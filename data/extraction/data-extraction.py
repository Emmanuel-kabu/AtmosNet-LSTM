"""
Data extraction module for the ATM Forecast project.

Responsible for pulling raw weather data from external sources (Kaggle, CSV,
API) and landing it into the data-lake raw layer as immutable, date-partitioned
Parquet files.  Every extraction run is:

    • Watermark-aware  — only fetches rows newer than ``last_success_at``.
    • Manifest-tracked — writes a JSON manifest to ``data/manifests/``.
    • Idempotent       — re-running with the same source is a safe no-op.

Usage
─────
    # From the project root:
    python data/extraction/data-extraction.py                        # defaults
    python data/extraction/data-extraction.py --dataset sumanthvrao/daily-climate-time-series-data
    python data/extraction/data-extraction.py --csv path/to/file.csv --date-col date_time
"""

from __future__ import annotations

import abc
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
import argparse

import pandas as pd

# ── resolve project root so we can import from src/ ───────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from atm_forecast.config import get_settings  # noqa: E402
from atm_forecast.data.lake import (  # noqa: E402
    LAYER_RAW,
    ensure_lake_dirs,
    write_partition,
)
from atm_forecast.data.manifest import Manifest  # noqa: E402
from atm_forecast.data.pipeline_state import (  # noqa: E402
    Watermark,
    get_engine,
    get_watermark,
    update_watermark,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Abstract base — any new source just subclasses this
# ═══════════════════════════════════════════════════════════════════════════

class BaseExtractor(abc.ABC):
    """Template for all data extractors.

    Subclasses must implement :meth:`_fetch_dataframe` which returns the
    raw ``pd.DataFrame`` from whatever source (Kaggle, REST API, S3, …).

    The base class handles:
    * watermark check  (skip already-ingested rows)
    * date-partitioned Parquet writes to ``data/lake/raw/``
    * watermark update in Postgres / SQLite
    * manifest creation in ``data/manifests/``
    """

    PIPELINE_NAME: str = "base_extractor"

    def __init__(
        self,
        *,
        date_column: str = "date",
        lake_root: Path | None = None,
        manifests_dir: Path | None = None,
        db_url: str | None = None,
    ) -> None:
        self._settings = get_settings()
        self.date_column = date_column
        self.lake_root = lake_root or self._settings.lake_root
        self.manifests_dir = manifests_dir or self._settings.manifests_dir
        self.db_url = db_url or self._settings.database_url

        self._engine = get_engine(self.db_url)
        ensure_lake_dirs(self.lake_root)
        logger.info(
            "%s initialised — lake=%s, db=%s",
            self.__class__.__name__, self.lake_root, self.db_url,
        )

    # ── abstract hook ─────────────────────────────────────────────────

    @abc.abstractmethod
    def _fetch_dataframe(self) -> pd.DataFrame:
        """Return the raw data as a DataFrame.

        The returned frame **must** contain a column whose name matches
        ``self.date_column`` and whose values are parseable as dates.
        """

    # ── public API ────────────────────────────────────────────────────

    def extract(self) -> list[date]:
        """Run the full extraction cycle.

        Returns
        -------
        list[date]
            Partition dates that were written to ``data/lake/raw/``.
        """
        manifest = Manifest(pipeline=self.PIPELINE_NAME)

        try:
            # 1. Read watermark ───────────────────────────────────────
            watermark = get_watermark(self._engine, self.PIPELINE_NAME)
            logger.info("Current watermark: %s", watermark)

            # 2. Fetch raw data ───────────────────────────────────────
            df = self._fetch_dataframe()
            df = self.ensure_date_column(df)
            logger.info("Fetched %d rows, %d columns", len(df), len(df.columns))

            # 3. Filter to new-only ───────────────────────────────────
            df = self.apply_watermark_filter(df, watermark)
            if df.empty:
                logger.info("No new data after watermark — nothing to ingest")
                manifest.mark_success()
                manifest.save(self.manifests_dir)
                return []

            # 4. Validate ─────────────────────────────────────────────
            df = self.validate(df)

            # 5. Partition & write ────────────────────────────────────
            written_dates = self.write_partitions(df, manifest)

            # 6. Update watermark ─────────────────────────────────────
            self._advance_watermark(df, written_dates)

            # 7. Save manifest ────────────────────────────────────────
            manifest.mark_success()
            manifest.save(self.manifests_dir)
            logger.info(
                "Extraction complete — %d partitions, %d rows",
                len(written_dates), len(df),
            )
            return written_dates

        except Exception as exc:
            manifest.mark_failure(str(exc))
            manifest.save(self.manifests_dir)
            logger.exception("Extraction failed")
            raise

    # ── internal helpers ──────────────────────────────────────────────

    def ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce the date column to ``datetime64`` if needed."""
        if self.date_column not in df.columns:
            raise KeyError(
                f"Date column {self.date_column!r} not found. "
                f"Available: {list(df.columns)}"
            )
        df[self.date_column] = pd.to_datetime(df[self.date_column], utc=False)
        return df

    def apply_watermark_filter(
        self, df: pd.DataFrame, watermark: Watermark,
    ) -> pd.DataFrame:
        """Keep only rows strictly after the watermark timestamp."""
        if watermark.last_success_at is None:
            return df  # first run — take everything

        cutoff = watermark.last_success_at
        col = df[self.date_column]

        # align tz-awareness for comparison
        if col.dt.tz is None and cutoff.tzinfo is not None:
            cutoff = cutoff.replace(tzinfo=None)
        elif col.dt.tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)

        before = len(df)
        df = df.loc[col > cutoff].copy()
        logger.info("Watermark filter: %d → %d rows (cutoff=%s)", before, len(df), cutoff)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic validation — override in subclasses for domain rules."""
        # Drop fully-empty rows
        df = df.dropna(how="all")

        # Report column-level nulls
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning("Null counts per column:\n%s", null_counts[null_counts > 0])

        logger.info("Validation passed — %d rows retained", len(df))
        return df

    def write_partitions(
        self, df: pd.DataFrame, manifest: Manifest,
    ) -> list[date]:
        """Partition by date and write Parquet files to the raw layer."""
        df["_partition_date"] = df[self.date_column].dt.date
        written: list[date] = []

        for part_date, group in df.groupby("_partition_date"):
            group = group.drop(columns=["_partition_date"])
            # Set date column as index for a clean Parquet layout
            if self.date_column in group.columns:
                group = group.set_index(self.date_column).sort_index()

            out_path = write_partition(
                group, self.lake_root, LAYER_RAW, part_date, overwrite=False,
            )
            manifest.add_partition(part_date, LAYER_RAW, len(group), out_path)
            written.append(part_date)
            logger.debug("Partition %s → %s (%d rows)", part_date, out_path, len(group))

        return written

    def _advance_watermark(
        self, df: pd.DataFrame, written_dates: list[date],
    ) -> None:
        """Set the watermark to the maximum timestamp we just ingested."""
        max_ts = df[self.date_column].max()

        # normalise to a tz-aware datetime
        if isinstance(max_ts, pd.Timestamp):
            max_ts = max_ts.to_pydatetime()
        if not isinstance(max_ts, datetime):
            max_ts = datetime.combine(max_ts, datetime.min.time())
        if max_ts.tzinfo is None:
            max_ts = max_ts.replace(tzinfo=timezone.utc)

        update_watermark(
            self._engine,
            self.PIPELINE_NAME,
            last_success_at=max_ts,
            last_partition_date=max(written_dates),
            rows_ingested=len(df),
        )


#  Concrete extractor — Kaggle

class KaggleExtractor(BaseExtractor):
    """Download a dataset from Kaggle via ``kagglehub`` and ingest it.

    Parameters
    ----------
    dataset_slug : str
        Kaggle dataset identifier, e.g.
        ``"sumanthvrao/daily-climate-time-series-data"``.
    filename : str | None
        If the dataset contains multiple files, specify which CSV to use.
        When *None*, the first ``.csv`` found is loaded.
    date_column : str
        Column in the CSV that holds observation dates.
    """

    PIPELINE_NAME = "kaggle_ingest"

    def __init__(
        self,
        dataset_slug: str = "nelgiriyewithana/global-weather-repository",
        *,
        filename: str | None = None,
        date_column: str = "last_updated",
        **kwargs: Any,
    ) -> None:
        super().__init__(date_column=date_column, **kwargs)
        self.dataset_slug = dataset_slug
        self.filename = filename

    def _fetch_dataframe(self) -> pd.DataFrame:
        """Download from Kaggle and read the CSV into a DataFrame."""
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required for Kaggle extraction. "
                "Install with: pip install kagglehub"
            ) from exc

        logger.info("Downloading dataset %r from Kaggle …", self.dataset_slug)
        download_path = Path(kagglehub.dataset_download(self.dataset_slug))
        logger.info("Kaggle download path: %s", download_path)

        csv_path = self.resolve_csv(download_path)
        logger.info("Reading CSV: %s", csv_path)
        return pd.read_csv(csv_path)

    def resolve_csv(self, download_path: Path) -> Path:
        """Find the target CSV inside the downloaded Kaggle directory."""
        if download_path.is_file() and download_path.suffix == ".csv":
            return download_path

        if not download_path.is_dir():
            raise FileNotFoundError(
                f"Kaggle download is neither a file nor a directory: {download_path}"
            )

        # User specified a filename
        if self.filename:
            target = download_path / self.filename
            if not target.exists():
                available = [f.name for f in download_path.rglob("*.csv")]
                raise FileNotFoundError(
                    f"{self.filename!r} not found in download. "
                    f"Available CSVs: {available}"
                )
            return target

        # Auto-detect: pick the first CSV
        csvs = sorted(download_path.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found in Kaggle download: {download_path}"
            )
        if len(csvs) > 1:
            logger.warning(
                "Multiple CSVs found — using %s. Pass --filename to override. "
                "All: %s", csvs[0].name, [c.name for c in csvs],
            )
        return csvs[0]

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kaggle-specific validation: check expected climate columns."""
        df = super().validate(df)

        expected = {"meantemp", "humidity", "wind_speed", "meanpressure"}
        present = expected & set(df.columns)
        missing = expected - present

        if present:
            logger.info("Found expected columns: %s", sorted(present))
        if missing:
            logger.warning(
                "Missing expected columns (may differ by dataset): %s",
                sorted(missing),
            )

        # Ensure numeric columns are actually numeric
        for col in present:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


#  Concrete extractor — local CSV file

class CSVExtractor(BaseExtractor):
    """Ingest from a local CSV file on disk.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.
    date_column : str
        Column that holds observation dates.
    """

    PIPELINE_NAME = "csv_ingest"

    def __init__(
        self,
        filepath: str | Path,
        *,
        date_column: str = "date",
        **kwargs: Any,
    ) -> None:
        super().__init__(date_column=date_column, **kwargs)
        self.filepath = Path(filepath)

    def _fetch_dataframe(self) -> pd.DataFrame:
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")

        logger.info("Reading local CSV: %s", self.filepath)
        return pd.read_csv(self.filepath)



#  Concrete extractor — REST API (template)

class APIExtractor(BaseExtractor):
    """Ingest from a REST API that returns JSON rows.

    This is a template — subclass and override ``_build_request_params``
    and ``_parse_response`` for your specific weather API.

    Parameters
    ----------
    base_url : str
        Base URL of the API.
    api_key : str
        Authentication token (set via env for production).
    date_column : str
        Column name to assign to the parsed dates.
    """

    PIPELINE_NAME = "api_ingest"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        date_column: str = "date",
        **kwargs: Any,
    ) -> None:
        super().__init__(date_column=date_column, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def build_request_params(self, watermark: Watermark) -> dict[str, Any]:
        """Build query parameters — override for your API schema."""
        params: dict[str, Any] = {}
        if watermark.last_success_at:
            params["start_date"] = watermark.last_success_at.isoformat()
        return params

    def parse_response(self, data: Any) -> pd.DataFrame:
        """Convert API JSON response into a DataFrame — override me."""
        return pd.DataFrame(data)

    def _fetch_dataframe(self) -> pd.DataFrame:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for API extraction. "
                "Install with: pip install httpx"
            ) from exc

        watermark = get_watermark(self._engine, self.PIPELINE_NAME)
        params = self.build_request_params(watermark)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        logger.info("Fetching from %s with params %s", self.base_url, params)

        with httpx.Client(timeout=60) as client:
            response = client.get(self.base_url, params=params, headers=headers)
            response.raise_for_status()

        data = response.json()
        df = self.parse_response(data)
        logger.info("API returned %d rows", len(df))
        return df


#  Factory — build the right extractor from CLI args or config

class ExtractorFactory:
    """Create the appropriate extractor from a source type string."""

    _registry: dict[str, type[BaseExtractor]] = {
        "kaggle": KaggleExtractor,
        "csv": CSVExtractor,
        "api": APIExtractor,
    }

    @classmethod
    def register(cls, name: str, extractor_cls: type[BaseExtractor]) -> None:
        """Register a custom extractor at runtime."""
        cls._registry[name] = extractor_cls

    @classmethod
    def create(cls, source_type: str, **kwargs: Any) -> BaseExtractor:
        """Instantiate an extractor by source type.

        Parameters
        ----------
        source_type : str
            One of ``"kaggle"``, ``"csv"``, ``"api"`` (or a custom
            registered name).
        **kwargs
            Forwarded to the extractor constructor.

        Returns
        -------
        BaseExtractor
        """
        extractor_cls = cls._registry.get(source_type)
        if extractor_cls is None:
            raise ValueError(
                f"Unknown source type {source_type!r}. "
                f"Available: {sorted(cls._registry)}"
            )
        return extractor_cls(**kwargs)



#  CLI entry-point

def build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract raw weather data into the data-lake raw layer.",
    )
    sub = parser.add_subparsers(dest="source", help="Data source type")

    # ── kaggle ────────────────────────────────────────────────────────
    kaggle_p = sub.add_parser("kaggle", help="Download from Kaggle")
    kaggle_p.add_argument(
        "--dataset",
        default="nelgiriyewithana/global-weather-repository",
        help="Kaggle dataset slug (default: nelgiriyewithana/global-weather-repository)",
    )
    kaggle_p.add_argument(
        "--filename", default=None,
        help="Specific CSV inside the Kaggle dataset (auto-detects if omitted)",
    )

    # ── csv ───────────────────────────────────────────────────────────
    csv_p = sub.add_parser("csv", help="Ingest from a local CSV file")
    csv_p.add_argument("filepath", help="Path to the CSV file")

    # ── common ────────────────────────────────────────────────────────
    for p in (kaggle_p, csv_p, parser):
        p.add_argument("--date-col", default="last_updated", help="Date column name")
        p.add_argument("--db-url", default=None, help="Watermark DB URL")
        p.add_argument(
            "--log-level", default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    source = args.source or "kaggle"  # default when no subcommand given
    common = {"date_column": args.date_col}
    if args.db_url:
        common["db_url"] = args.db_url

    if source == "kaggle":
        kwargs = {
            "dataset_slug": getattr(args, "dataset", "nelgiriyewithana/global-weather-repository"),
            "filename": getattr(args, "filename", None),
            **common,
        }
    elif source == "csv":
        kwargs = {"filepath": args.filepath, **common}
    else:
        parser.print_help()
        return 1

    extractor = ExtractorFactory.create(source, **kwargs)
    written = extractor.extract()

    if written:
        print(f"Extracted {len(written)} partition(s): {written}")
    else:
        print("No new data to extract.")

    return 0


if __name__ == "__main__":
    sys.exit(main())