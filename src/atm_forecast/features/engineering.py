"""Feature engineering utilities for atmospheric time-series data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged versions of specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input time-indexed DataFrame.
    columns : list[str]
        Columns to create lag features for.
    lags : list[int] | None
        Lag periods. Defaults to [1, 2, 3, 6, 12, 24].

    Returns
    -------
    pd.DataFrame
        DataFrame with additional lag columns.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]

    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    logger.info("Added lag features for %s with lags %s", columns, lags)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and standard deviation features.

    Parameters
    ----------
    df : pd.DataFrame
        Input time-indexed DataFrame.
    columns : list[str]
        Columns to create rolling features for.
    windows : list[int] | None
        Window sizes. Defaults to [6, 12, 24].

    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling-statistic columns.
    """
    if windows is None:
        windows = [6, 12, 24]

    df = df.copy()
    for col in columns:
        for w in windows:
            df[f"{col}_rolling_mean_{w}"] = df[col].rolling(window=w).mean()
            df[f"{col}_rolling_std_{w}"] = df[col].rolling(window=w).std()

    logger.info("Added rolling features for %s with windows %s", columns, windows)
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode time-of-day and day-of-year as cyclical sin/cos features.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with cyclical time columns appended.
    """
    df = df.copy()
    idx = df.index

    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    # Hour of day (period = 24)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)

    # Day of year (period = 365.25)
    df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    # Day of week (period = 7)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # Month of year (period = 12)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    logger.info("Added cyclical time features")
    return df
