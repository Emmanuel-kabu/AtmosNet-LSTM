"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Literal


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    *,
    json_format: bool = False,
) -> None:
    """Configure the root logger with a consistent format.

    Parameters
    ----------
    level : str
        Logging level name.
    json_format : bool
        If True, emit structured JSON lines (useful in production / Docker).
    """
    handlers: list[logging.Handler] = []

    if json_format:
        try:
            import json_log_formatter  # type: ignore[import-untyped]

            formatter = json_log_formatter.JSONFormatter()
        except ImportError:
            formatter = logging.Formatter(
                '{"time":"%(asctime)s","level":"%(levelname)s",'
                '"logger":"%(name)s","message":"%(message)s"}'
            )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Suppress overly chatty third-party loggers
    for noisy in ("urllib3", "botocore", "tensorflow", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
