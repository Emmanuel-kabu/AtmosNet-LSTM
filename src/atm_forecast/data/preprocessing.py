"""Data preprocessing and sequence creation for time-series models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    test_ratio: float = 0.1,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically into train / val / test.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by time.
    test_ratio : float
        Fraction of data for the test set.
    val_ratio : float
        Fraction of data for the validation set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) DataFrames.
    """
    n = len(df)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size

    train = df.iloc[:train_size]
    val = df.iloc[train_size : train_size + val_size]
    test = df.iloc[train_size + val_size :]

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def prepare_pipeline(
    scaler_type: str = "minmax",
) -> Pipeline:
    """Create a sklearn preprocessing pipeline.

    Parameters
    ----------
    scaler_type : str
        'minmax' for MinMaxScaler or 'standard' for StandardScaler.

    Returns
    -------
    Pipeline
        A fitted-ready sklearn pipeline.
    """
    scalers = {
        "minmax": MinMaxScaler(feature_range=(0, 1)),
        "standard": StandardScaler(),
    }

    scaler = scalers.get(scaler_type)
    if scaler is None:
        raise ValueError(f"Unknown scaler type: {scaler_type!r}. Use 'minmax' or 'standard'.")

    return Pipeline([("scaler", scaler)])


def create_sequences(
    data: NDArray[Any],
    sequence_length: int,
    forecast_horizon: int = 1,
    target_col_idx: int = 0,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Slide a window over the data to create (X, y) sequence pairs.

    Parameters
    ----------
    data : NDArray
        2-D array of shape (timesteps, features).
    sequence_length : int
        Number of past time-steps in each input window.
    forecast_horizon : int
        Number of time-steps to predict ahead.
    target_col_idx : int
        Column index of the target variable.

    Returns
    -------
    tuple[NDArray, NDArray]
        X of shape (samples, sequence_length, features) and
        y of shape (samples,).
    """
    xs: list[NDArray[Any]] = []
    ys: list[float] = []

    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        xs.append(data[i : i + sequence_length])
        ys.append(data[i + sequence_length + forecast_horizon - 1, target_col_idx])

    return np.array(xs), np.array(ys)
