"""
Feature Engineering for Atmospheric / Air-Quality Time-Series
=============================================================

All feature engineering used in the AtmosNet pipeline.  Every function
takes a DataFrame, adds columns, and returns the augmented DataFrame
(no in-place mutation).

Features produced
-----------------
- **Daylight hours** — from sunrise / sunset strings
- **Wind direction encoding** — sin / cos of ``wind_degree``
- **Pressure tendency** — 1-day and 3-day diffs
- **Interaction features** — temp×humidity, wind×temp, precip×humidity
- **Location encoding** — category codes from ``location_name``
- **Cyclical time features** — sin / cos for day-of-year, month, day-of-week
- **Lag features** — per-location shifted values at configurable lags
- **Rolling features** — per-location rolling mean & std at configurable windows

The convenience function ``run_feature_engineering()`` chains everything
together and returns a model-ready DataFrame (no text columns, no NaN).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Individual feature transformations
# =====================================================================

def add_daylight_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daylight duration from sunrise / sunset strings.

    Expects columns ``sunrise``, ``sunset``, and ``date`` (datetime).
    Drops the raw sunrise/sunset columns after conversion.
    """
    df = df.copy()
    if "sunrise" not in df.columns or "sunset" not in df.columns:
        logger.warning("sunrise/sunset columns not found — skipping daylight_hours")
        return df

    date_str = df["date"].dt.strftime("%Y-%m-%d")
    sunrise_dt = pd.to_datetime(date_str + " " + df["sunrise"], errors="coerce")
    sunset_dt = pd.to_datetime(date_str + " " + df["sunset"], errors="coerce")
    df["daylight_hours"] = ((sunset_dt - sunrise_dt).dt.total_seconds() / 3600).abs()
    df = df.drop(columns=["sunrise", "sunset"], errors="ignore")

    logger.info(
        "daylight_hours — mean: %.1fh, range: [%.1f, %.1f]",
        df["daylight_hours"].mean(),
        df["daylight_hours"].min(),
        df["daylight_hours"].max(),
    )
    return df


def add_wind_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ``wind_degree`` as sin / cos and drop the raw column."""
    df = df.copy()
    if "wind_degree" not in df.columns:
        logger.warning("wind_degree not found — skipping wind encoding")
        return df

    df["wind_dir_sin"] = np.sin(np.radians(df["wind_degree"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_degree"]))
    df = df.drop(columns=["wind_degree"])
    logger.info("Created wind_dir_sin, wind_dir_cos (dropped wind_degree)")
    return df


def add_pressure_change(
    df: pd.DataFrame,
    location_col: str = "location_name",
) -> pd.DataFrame:
    """1-day and 3-day pressure differences per location."""
    df = df.copy()
    if "pressure_mb" not in df.columns:
        logger.warning("pressure_mb not found — skipping pressure change")
        return df

    grp = location_col if location_col in df.columns else None
    if grp:
        df["pressure_change_1d"] = df.groupby(grp)["pressure_mb"].diff(1)
        df["pressure_change_3d"] = df.groupby(grp)["pressure_mb"].diff(3)
    else:
        df["pressure_change_1d"] = df["pressure_mb"].diff(1)
        df["pressure_change_3d"] = df["pressure_mb"].diff(3)
    logger.info("Created pressure_change_1d, pressure_change_3d")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative interactions between key weather variables."""
    df = df.copy()
    interactions = {
        "temp_x_humidity": ("temperature_celsius", "humidity"),
        "wind_x_temp": ("wind_kph", "temperature_celsius"),
        "precip_x_humidity": ("precip_mm", "humidity"),
    }
    created = []
    for name, (a, b) in interactions.items():
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]
            created.append(name)
    logger.info("Interaction features: %s", created)
    return df


def add_location_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ``location_name`` as integer codes and drop text columns."""
    df = df.copy()
    if "location_name" in df.columns:
        df["location_id"] = df["location_name"].astype("category").cat.codes
        logger.info("location_id — %d unique locations", df["location_id"].nunique())
    df = df.drop(columns=["country", "location_name"], errors="ignore")
    logger.info("Dropped text columns: country, location_name")
    return df


def add_cyclical_time_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Sin / cos encoding for day-of-year, month, and day-of-week.

    Works with either a ``date`` column or a DatetimeIndex.
    """
    df = df.copy()

    if date_col in df.columns:
        dt = df[date_col]
    elif isinstance(df.index, pd.DatetimeIndex):
        dt = df.index.to_series()
    else:
        raise TypeError(
            "Provide a date column or set a DatetimeIndex to use cyclical features."
        )

    df["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    logger.info("Cyclical features: doy, month, dow (sin+cos)")
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] | None = None,
    group_col: str = "location_id",
) -> pd.DataFrame:
    """Add lagged versions of specified columns, grouped per location.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Columns to lag.
    lags : list[int]
        Lag periods (default [1, 3, 7]).
    group_col : str
        Column to group by when shifting (default ``location_id``).
    """
    if lags is None:
        lags = [1, 3, 7]

    df = df.copy()
    grp = group_col if group_col in df.columns else None

    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            col_name = f"{col}_lag_{lag}"
            if grp:
                df[col_name] = df.groupby(grp)[col].shift(lag)
            else:
                df[col_name] = df[col].shift(lag)

    n_created = sum(1 for c in columns if c in df.columns) * len(lags)
    logger.info("Lag features: %d columns × %s lags = %d new", len(columns), lags, n_created)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] | None = None,
    group_col: str = "location_id",
) -> pd.DataFrame:
    """Add rolling mean and std per location for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Columns to compute rolling stats on.
    windows : list[int]
        Window sizes (default [3, 7]).
    group_col : str
        Column to group by (default ``location_id``).
    """
    if windows is None:
        windows = [3, 7]

    df = df.copy()
    grp = group_col if group_col in df.columns else None

    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            if grp:
                df[f"{col}_rmean_{w}"] = df.groupby(grp)[col].transform(
                    lambda x: x.rolling(window=w, min_periods=1).mean()
                )
                df[f"{col}_rstd_{w}"] = df.groupby(grp)[col].transform(
                    lambda x: x.rolling(window=w, min_periods=1).std()
                )
            else:
                df[f"{col}_rmean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
                df[f"{col}_rstd_{w}"] = df[col].rolling(window=w, min_periods=1).std()

    n_created = sum(1 for c in columns if c in df.columns) * len(windows) * 2
    logger.info("Rolling features: %d columns × %s windows × 2 = %d new",
                len(columns), windows, n_created)
    return df


# =====================================================================
# Convenience: full feature-engineering pipeline
# =====================================================================

DEFAULT_TARGETS: List[str] = [
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "temperature_celsius",
]

DEFAULT_LAG_COLS: List[str] = DEFAULT_TARGETS + [
    "humidity", "pressure_mb", "wind_kph", "precip_mm", "uv_index",
]

DEFAULT_ROLL_COLS: List[str] = DEFAULT_TARGETS + [
    "humidity", "pressure_mb", "wind_kph",
]


def run_feature_engineering(
    df: pd.DataFrame,
    targets: List[str] | None = None,
    lag_cols: List[str] | None = None,
    roll_cols: List[str] | None = None,
    lags: List[int] | None = None,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """Run the complete feature-engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of ``PreprocessingPipeline.clean``).
        Must contain ``date`` (datetime), ``location_name``, and numeric
        weather columns.
    targets : list[str]
        Target column names (used to seed lag/rolling defaults).
    lag_cols / roll_cols : list[str]
        Override which columns get lag / rolling features.
    lags / windows : list[int]
        Override lag periods / rolling window sizes.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame with no text columns and no NaN rows.
    """
    targets = targets or DEFAULT_TARGETS
    lag_cols = lag_cols or DEFAULT_LAG_COLS
    roll_cols = roll_cols or DEFAULT_ROLL_COLS
    lags = lags or [1, 3, 7]
    windows = windows or [3, 7]

    logger.info("Starting feature engineering — input shape: %s", df.shape)

    # 1. Daylight hours
    df = add_daylight_hours(df)

    # 2. Wind encoding
    df = add_wind_encoding(df)

    # 3. Pressure change
    df = add_pressure_change(df)

    # 4. Interaction features
    df = add_interaction_features(df)

    # 5. Location encoding (must come before lags that use location_id)
    df = add_location_encoding(df)

    # 6. Cyclical time features
    df = add_cyclical_time_features(df)

    # 7. Lag features
    df = add_lag_features(df, columns=lag_cols, lags=lags)

    # 8. Rolling features
    df = add_rolling_features(df, columns=roll_cols, windows=windows)

    # 9. Drop NaN rows created by lags and drop date/temp EDA columns
    rows_before = len(df)
    df = df.dropna()
    df = df.drop(columns=["date", "month", "day_of_year"], errors="ignore")
    logger.info(
        "Dropped %d NaN rows + date columns — final shape: %s",
        rows_before - len(df), df.shape,
    )

    return df
