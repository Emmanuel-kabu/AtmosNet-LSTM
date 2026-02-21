"""CLI entrypoint for the incremental data-lake pipeline.

Usage
─────
    # Run entire pipeline (ingest → clean → features)
    python -m atm_forecast.data.pipeline_cli --source data/temperature_data.csv

    # Run single stage
    python -m atm_forecast.data.pipeline_cli --source data/temperature_data.csv --stage ingest
    python -m atm_forecast.data.pipeline_cli --stage clean
    python -m atm_forecast.data.pipeline_cli --stage features

    # Or via the installed script:
    atm-pipeline --source data/temperature_data.csv
"""

from __future__ import annotations

import argparse
import sys

from atm_forecast.utils.logging import setup_logging


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental data-lake pipeline (raw → clean → features).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to source CSV file (required for 'ingest' and 'all' stages).",
    )
    parser.add_argument(
        "--stage",
        choices=["ingest", "clean", "features", "all"],
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--date-column",
        type=str,
        default="date",
        help="Name of the date column in the source CSV (default: 'date').",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="meantemp",
        help="Target column for feature engineering (default: 'meantemp').",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Database URL for watermark storage (default: from settings).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(level=args.log_level)

    from atm_forecast.data.pipeline import (
        ingest_raw,
        transform_clean,
        transform_features,
    )

    stage = args.stage
    new_dates: list = []

    # ── INGEST ────────────────────────────────────────────────────────
    if stage in ("ingest", "all"):
        if not args.source:
            print("ERROR: --source is required for the 'ingest' stage.", file=sys.stderr)
            return 1
        new_dates = ingest_raw(
            args.source,
            date_column=args.date_column,
            db_url=args.db_url,
        )
        print(f"[ingest] Wrote {len(new_dates)} raw partitions: {new_dates}")

    # ── CLEAN ─────────────────────────────────────────────────────────
    if stage in ("clean", "all"):
        # If running after ingest, process only the newly-ingested dates
        dates_to_clean = new_dates if (stage == "all" and new_dates) else None
        cleaned = transform_clean(dates=dates_to_clean)
        print(f"[clean]  Processed {len(cleaned)} partitions: {cleaned}")

    # ── FEATURES ──────────────────────────────────────────────────────
    if stage in ("features", "all"):
        dates_for_feat = new_dates if (stage == "all" and new_dates) else None
        featured = transform_features(
            dates=dates_for_feat,
            target_column=args.target,
        )
        print(f"[features] Processed {len(featured)} partitions: {featured}")

    print("Pipeline complete ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
