"""CLI entrypoint: ``python -m atm_forecast.training``."""

from __future__ import annotations

import argparse
import sys

from atm_forecast.training.train import run_training
from atm_forecast.utils.logging import setup_logging


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the atmospheric-forecast LSTM model.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the input CSV / Parquet data file (required unless --use-lake).",
    )
    parser.add_argument(
        "--use-lake",
        action="store_true",
        default=False,
        help="Load pre-engineered features from data/lake/features/ "
             "instead of a raw file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="temperature",
        help="Name of the target column (default: temperature).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for saved artefacts.",
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

    if not args.use_lake and args.data is None:
        print("ERROR: --data is required unless --use-lake is set.", file=sys.stderr)
        return 1

    overrides: dict = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    run_training(
        data_path=args.data,
        target_column=args.target,
        output_dir=args.output_dir,
        use_lake=args.use_lake,
        **overrides,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
