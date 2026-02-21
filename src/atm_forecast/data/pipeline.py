"""Incremental data-lake pipeline stages.

Three stages mirror the lake layers:

    raw      – ingest from CSV / API, write immutable Parquet partitions
    clean    – types, missing values, dedup
    features – lags, rolling stats, cyclical time encodings

Each stage is watermark-aware and only processes *new* partitions.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from atm_forecast.config import get_settings
from atm_forecast.data.lake import (
    LAYER_CLEAN,
    LAYER_FEATURES,
    LAYER_RAW,
    ensure_lake_dirs,
    list_partition_dates,
    pending_partitions,
    read_partition,
    write_partition,
)
from atm_forecast.data.manifest import Manifest
from atm_forecast.data.pipeline_state import (
    Watermark,
    get_engine,
    get_watermark,
    update_watermark,
)
from atm_forecast.features.engineering import (
    add_cyclical_time_features,
    add_lag_features,
    add_rolling_features,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — Ingest raw data
# ═══════════════════════════════════════════════════════════════════════════

def ingest_raw(
    source_csv: str | Path,
    *,
    date_column: str = "date",
    lake_root: Path | None = None,
    db_url: str | None = None,
    pipeline_name: str = "weather_ingest",
) -> list[date]:
    """Read a CSV, partition rows by *date_column*, and write to ``raw/``.

    Only rows whose ``observed_at`` (or ``date_column``) is **after** the
    watermark stored in Postgres are ingested.  On the very first run the
    entire file is ingested.

    Returns the list of partition dates that were written.
    """
    settings = get_settings()
    lake_root = lake_root or settings.data_dir / "lake"
    db_url = db_url or settings.database_url
    ensure_lake_dirs(lake_root)

    # 1. Read watermark ---------------------------------------------------
    engine = get_engine(db_url)
    wm: Watermark = get_watermark(engine, pipeline_name)
    logger.info("Watermark for %s: %s", pipeline_name, wm)

    # 2. Load source ──────────────────────────────────────────────────────
    source_csv = Path(source_csv)
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    df = pd.read_csv(source_csv, parse_dates=[date_column])

    if date_column not in df.columns:
        raise KeyError(f"Column {date_column!r} not found in {list(df.columns)}")

    # 3. Filter to new data only ──────────────────────────────────────────
    if wm.last_success_at is not None:
        cutoff = wm.last_success_at
        # Ensure tz-aware comparison
        if df[date_column].dt.tz is None:
            cutoff = cutoff.replace(tzinfo=None) if cutoff.tzinfo else cutoff
        df = df[df[date_column] > cutoff]
        logger.info("Filtered to %d new rows (after %s)", len(df), cutoff)

    if df.empty:
        logger.info("No new data to ingest for %s", pipeline_name)
        return []

    # 4. Partition by date and write ──────────────────────────────────────
    df["_partition_date"] = df[date_column].dt.date
    manifest = Manifest(pipeline=pipeline_name)
    written_dates: list[date] = []

    for part_date, group in df.groupby("_partition_date"):
        group = group.drop(columns=["_partition_date"])
        # Set the date column as index before writing
        if date_column in group.columns:
            group = group.set_index(date_column).sort_index()
        out = write_partition(group, lake_root, LAYER_RAW, part_date)
        manifest.add_partition(part_date, LAYER_RAW, len(group), out)
        written_dates.append(part_date)

    # 5. Update watermark ─────────────────────────────────────────────────
    max_ts = df[date_column].max()
    if not isinstance(max_ts, datetime):
        max_ts = datetime.combine(max_ts, datetime.min.time(), tzinfo=timezone.utc)
    elif max_ts.tzinfo is None:
        max_ts = max_ts.replace(tzinfo=timezone.utc)

    update_watermark(
        engine, pipeline_name,
        last_success_at=max_ts,
        last_partition_date=max(written_dates),
        rows_ingested=len(df),
    )

    # 6. Save manifest ────────────────────────────────────────────────────
    manifests_dir = settings.data_dir / "manifests"
    manifest.mark_success()
    manifest.save(manifests_dir)

    logger.info("Ingested %d partitions (%d rows)", len(written_dates), len(df))
    return written_dates


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — Clean (types, missing values, dedup)
# ═══════════════════════════════════════════════════════════════════════════

def transform_clean(
    *,
    dates: list[date] | None = None,
    lake_root: Path | None = None,
) -> list[date]:
    """Apply cleaning rules to raw partitions and write to ``clean/``.

    If *dates* is ``None``, processes all raw partitions not yet present in
    the clean layer (incremental).

    Returns partition dates that were processed.
    """
    settings = get_settings()
    lake_root = lake_root or settings.data_dir / "lake"

    if dates is None:
        dates = pending_partitions(lake_root, LAYER_RAW, LAYER_CLEAN)
        if not dates:
            logger.info("Clean layer is up to date — nothing to process")
            return []

    manifest = Manifest(pipeline="clean_transform")
    processed: list[date] = []

    for d in dates:
        logger.info("Cleaning partition date=%s", d)
        df = read_partition(lake_root, LAYER_RAW, d)
        df = _clean_dataframe(df)
        out = write_partition(df, lake_root, LAYER_CLEAN, d, overwrite=True)
        manifest.add_partition(d, LAYER_CLEAN, len(df), out)
        processed.append(d)

    manifest.mark_success()
    manifest.save(settings.data_dir / "manifests")
    logger.info("Cleaned %d partitions", len(processed))
    return processed


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning: cast types, drop duplicates, handle missing."""
    # 1. Drop full-row duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.debug("Dropped %d duplicate rows", dropped)

    # 2. Numeric columns — fill small gaps with interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3)

    # 3. Drop remaining rows with any NaN
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        logger.debug("Dropped %d rows with remaining NaN", dropped)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — Feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def transform_features(
    *,
    dates: list[date] | None = None,
    target_column: str = "meantemp",
    lake_root: Path | None = None,
    lag_periods: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> list[date]:
    """Compute features from clean partitions and write to ``features/``.

    For correct lag/rolling computation we need a lookback window:
    we read some history from clean, compute features on the combined
    DataFrame, then slice back to only the target dates.

    Returns partition dates that were processed.
    """
    settings = get_settings()
    lake_root = lake_root or settings.data_dir / "lake"
    lag_periods = lag_periods or [1, 2, 3, 6, 12, 24]
    rolling_windows = rolling_windows or [6, 12, 24]
    max_lookback = max(max(lag_periods), max(rolling_windows))

    if dates is None:
        dates = pending_partitions(lake_root, LAYER_CLEAN, LAYER_FEATURES)
        if not dates:
            logger.info("Features layer is up to date — nothing to process")
            return []

    # We need lookback context from earlier clean partitions
    all_clean_dates = list_partition_dates(lake_root, LAYER_CLEAN)
    earliest_target = min(dates)

    # Load history for lookback
    context_dates = [d for d in all_clean_dates if d < earliest_target]
    # Keep only the last `max_lookback` days of history
    context_dates = context_dates[-max_lookback:] if len(context_dates) > max_lookback else context_dates

    load_dates = sorted(set(context_dates + dates))
    frames = []
    for d in load_dates:
        try:
            frames.append(read_partition(lake_root, LAYER_CLEAN, d))
        except FileNotFoundError:
            logger.warning("Clean partition missing for %s — skipping", d)

    if not frames:
        logger.warning("No clean data available for feature engineering")
        return []

    combined = pd.concat(frames).sort_index()

    # Determine numeric columns to engineer features on
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [target_column] if target_column in numeric_cols else numeric_cols[:1]

    # Apply feature engineering
    combined = add_lag_features(combined, columns=feat_cols, lags=lag_periods)
    combined = add_rolling_features(combined, columns=feat_cols, windows=rolling_windows)
    try:
        combined = add_cyclical_time_features(combined)
    except TypeError:
        logger.warning("Index is not DatetimeIndex — skipping cyclical features")

    # Drop NaN rows introduced by lags/rolling
    combined = combined.dropna()

    # Now slice and write only the target date partitions
    manifest = Manifest(pipeline="feature_transform")
    processed: list[date] = []

    if isinstance(combined.index, pd.DatetimeIndex):
        combined["_partition_date"] = combined.index.date
    else:
        # Fallback: use the dates we loaded
        logger.warning("Index is not DatetimeIndex — assigning partition dates from context")
        # Can't reliably slice, so write combined as single partition per target date
        for d in dates:
            out = write_partition(combined, lake_root, LAYER_FEATURES, d, overwrite=True)
            manifest.add_partition(d, LAYER_FEATURES, len(combined), out)
            processed.append(d)
        manifest.mark_success()
        manifest.save(settings.data_dir / "manifests")
        return processed

    target_set = set(dates)
    for part_date, group in combined.groupby("_partition_date"):
        if part_date not in target_set:
            continue
        group = group.drop(columns=["_partition_date"])
        out = write_partition(group, lake_root, LAYER_FEATURES, part_date, overwrite=True)
        manifest.add_partition(part_date, LAYER_FEATURES, len(group), out)
        processed.append(part_date)

    manifest.mark_success()
    manifest.save(settings.data_dir / "manifests")
    logger.info("Engineered features for %d partitions", len(processed))
    return processed
