"""Data loading utilities for time-series weather data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


def load_csv_data(
    filepath: str | Path,
    date_column: str = "date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load weather data from a CSV file.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.
    date_column : str
        Column to parse as datetime index.
    parse_dates : bool
        Whether to parse the date column.

    Returns
    -------
    pd.DataFrame
        Loaded and indexed DataFrame.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info("Loading data from %s", filepath)

    df = pd.read_csv(filepath, parse_dates=[date_column] if parse_dates else False)

    if parse_dates and date_column in df.columns:
        df = df.set_index(date_column).sort_index()

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def load_data(
    source: str,
    *,
    query: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load data from various sources (CSV, SQL, Parquet).

    Parameters
    ----------
    source : str
        A file path or database connection string.
    query : str | None
        SQL query (required when source is a database URL).
    **kwargs
        Extra keyword arguments forwarded to the underlying reader.

    Returns
    -------
    pd.DataFrame
    """
    source_path = Path(source)

    if source_path.suffix == ".csv":
        return load_csv_data(source_path, **kwargs)

    if source_path.suffix == ".parquet":
        logger.info("Loading parquet from %s", source)
        return pd.read_parquet(source_path, **kwargs)

    # Assume database connection string
    if query is None:
        raise ValueError("A SQL `query` is required when loading from a database.")

    logger.info("Loading data from database")
    engine = create_engine(source)
    return pd.read_sql(query, engine, **kwargs)
