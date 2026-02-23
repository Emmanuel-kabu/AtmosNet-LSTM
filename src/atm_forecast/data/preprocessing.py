"""
Data Preprocessing Pipeline for Atmospheric / Air-Quality Forecasting
=====================================================================

End-to-end preprocessing for multi-target time-series data ingested from
a Hive-partitioned Parquet data lake.

Pipeline steps
--------------
1. Load raw data from ``data/lake/raw/``
2. Parse the ``date`` column (Categorical → datetime)
3. Sort by ``(location_name, date)``
4. Drop redundant / leakage columns
5. Forward-fill + back-fill missing values per location
6. IQR-based outlier clipping on targets (3× IQR)
7. Auto target transforms (log1p / asinh) based on skew & kurtosis
8. Scaling — RobustScaler for features, StandardScaler for targets
9. Persist fitted scalers & transform metadata as artefacts

The module exposes:
* ``PreprocessingPipeline`` — stateful class that fits on train data and
  can transform / inverse-transform any split consistently.
* ``run_preprocessing()`` — convenience function that loads raw lake data,
  runs the full pipeline, and returns the cleaned DataFrame + artefacts.
* ``create_sequences()`` / ``split_data()`` — helpers for windowing &
  chronological splitting.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

# =====================================================================
# Constants
# =====================================================================

TARGETS: List[str] = [
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "temperature_celsius",
]

REDUNDANT_COLUMNS: List[str] = [
    # Unit duplicates (keep metric)
    "temperature_fahrenheit", "feels_like_fahrenheit",
    "pressure_in", "precip_in", "visibility_miles",
    "gust_mph", "wind_mph",
    # Data-leakage indices derived from targets
    "air_quality_us-epa-index", "air_quality_gb-defra-index",
    # Low predictive value
    "moonrise", "moonset", "moon_phase", "moon_illumination",
    # Text / replaced by engineered features
    "condition_text",
    "last_updated_epoch",
    "wind_direction",   # → sin/cos of wind_degree
    "timezone",         # → lat/lon + location_id
]


# Utility helpers

def _ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _skewness(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").skew())


def _kurtosis_fisher(x: pd.Series) -> float:
    """Fisher kurtosis (normal ~ 0)."""
    return float(pd.to_numeric(x, errors="coerce").kurt())


def _kurtosis_pearson(x: pd.Series) -> float:
    """Pearson kurtosis (normal ~ 3)."""
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(stats.kurtosis(s, fisher=False, bias=False))


def choose_transform(skew: float, kurt_f: float, min_val: float) -> str:
    """Pick a variance-stabilising transform based on distribution shape.

    Returns one of ``'none'``, ``'log1p'``, ``'asinh'``.
    """
    heavy = (abs(skew) > 1.0) or (kurt_f is not None and kurt_f > 2.0)
    if not heavy:
        return "none"
    if min_val >= 0 and skew >= 0:
        return "log1p"
    return "asinh"


def apply_transform(series: pd.Series | np.ndarray, t: str) -> np.ndarray:
    """Forward transform a series."""
    arr = np.asarray(series, dtype=np.float64)
    if t == "none":
        return arr
    if t == "log1p":
        return np.log1p(np.clip(arr, 0, None))
    if t == "asinh":
        return np.arcsinh(arr)
    raise ValueError(f"Unknown transform: {t}")


def inverse_transform_values(arr: np.ndarray, t: str) -> np.ndarray:
    """Inverse of ``apply_transform``."""
    if t == "none":
        return arr
    if t == "log1p":
        return np.expm1(arr)
    if t == "asinh":
        return np.sinh(arr)
    raise ValueError(f"Unknown transform: {t}")


# PreprocessingPipeline

class PreprocessingPipeline:
    """Stateful preprocessing pipeline fitted on training data.

    Parameters
    ----------
    targets : list[str]
        Target column names.
    iqr_multiplier : float
        Multiplier for IQR-based outlier clipping (default 3.0).
    """

    def __init__(
        self,
        targets: List[str] | None = None,
        iqr_multiplier: float = 3.0,
    ):
        self.targets = targets or TARGETS
        self.iqr_multiplier = iqr_multiplier

        # Fitted state
        self.feature_scaler: RobustScaler | None = None
        self.target_scaler: StandardScaler | None = None
        self.transforms_applied: Dict[str, str] = {}
        self.feature_cols: List[str] = []
        self.clip_bounds: Dict[str, Tuple[float, float]] = {}
        self._is_fitted = False

    # ── public API ────────────────────────────────────────────────

    def load_raw(self, raw_dir: str | Path) -> pd.DataFrame:
        """Read Hive-partitioned Parquet from the raw layer."""
        raw_dir = Path(raw_dir)
        logger.info("Loading raw data from %s", raw_dir)
        df = pd.read_parquet(raw_dir)
        logger.info("Raw shape: %s", df.shape)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Steps 1-4: parse dates, drop columns, fill NaN, clip outliers."""
        df = df.copy()

        # 1. Parse date (Hive partition → Categorical)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"].astype(str))
        df = df.sort_values(["location_name", "date"]).reset_index(drop=True)

        # 2. Drop redundant columns
        existing_drops = [c for c in REDUNDANT_COLUMNS if c in df.columns]
        df = df.drop(columns=existing_drops)
        logger.info("Dropped %d redundant columns", len(existing_drops))

        # 3. Forward-fill + back-fill per location, then median fallback
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df[numeric_cols] = df.groupby("location_name")[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # 4. Outlier clipping on targets (3× IQR)
        for col in self.targets:
            if col not in df.columns:
                continue
            s = df[col]
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self.clip_bounds[col] = (float(lower), float(upper))
            clipped = ((s < lower) | (s > upper)).sum()
            df[col] = s.clip(lower=lower, upper=upper)
            if clipped > 0:
                logger.info("Clipped %d outliers in %s", clipped, col)

        logger.info("Cleaned shape: %s  |  NaN remaining: %d",
                     df.shape, df.isnull().sum().sum())
        return df

    def fit_transform_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-transform targets (log1p / asinh) and fit StandardScaler."""
        df = df.copy()
        self.transforms_applied = {}

        for col in self.targets:
            if col not in df.columns:
                continue
            s = df[col]
            sk = _skewness(s)
            kf = _kurtosis_fisher(s)
            min_val = float(s.min())
            t = choose_transform(sk, kf, min_val)
            df[col] = apply_transform(s, t)
            self.transforms_applied[col] = t
            new_sk = _skewness(df[col])
            logger.info(
                "Target %-40s  transform=%-6s  skew: %+.2f → %+.2f",
                col, t, sk, new_sk,
            )

        return df

    def fit_scalers(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Fit RobustScaler on features and StandardScaler on targets.

        Returns the scaled DataFrame and the list of feature column names.
        """
        self.feature_cols = [c for c in df.columns if c not in self.targets]
        X = df[self.feature_cols]
        Y = df[self.targets]

        self.feature_scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            columns=self.feature_cols,
            index=X.index,
        )

        self.target_scaler = StandardScaler()
        Y_scaled = pd.DataFrame(
            self.target_scaler.fit_transform(Y),
            columns=self.targets,
            index=Y.index,
        )

        df_scaled = pd.concat([X_scaled, Y_scaled], axis=1)
        self._is_fitted = True

        logger.info(
            "Fitted scalers — %d features (Robust), %d targets (Standard)",
            len(self.feature_cols), len(self.targets),
        )
        return df_scaled, self.feature_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a new split using already-fitted scalers."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_scalers first.")

        df = df.copy()
        # Apply target transforms
        for col, t in self.transforms_applied.items():
            if col in df.columns:
                df[col] = apply_transform(df[col], t)

        X = df[self.feature_cols]
        Y = df[self.targets]

        X_scaled = pd.DataFrame(
            self.feature_scaler.transform(X),
            columns=self.feature_cols,
            index=X.index,
        )
        Y_scaled = pd.DataFrame(
            self.target_scaler.transform(Y),
            columns=self.targets,
            index=Y.index,
        )
        return pd.concat([X_scaled, Y_scaled], axis=1)

    def inverse_transform_targets(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-scale and inverse-transform target predictions.

        Parameters
        ----------
        arr : np.ndarray, shape (n_samples, n_targets)

        Returns
        -------
        np.ndarray  — predictions in original units.
        """
        if self.target_scaler is None:
            raise RuntimeError("Pipeline not fitted.")

        inv = self.target_scaler.inverse_transform(arr)
        for i, col in enumerate(self.targets):
            t = self.transforms_applied.get(col, "none")
            inv[:, i] = inverse_transform_values(inv[:, i], t)
        return inv

    # ── Persistence ───────────────────────────────────────────────

    def save(self, artifacts_dir: str | Path) -> Path:
        """Persist pipeline state to disk."""
        d = _ensure_dir(artifacts_dir)

        with open(d / "feature_scaler.pkl", "wb") as f:
            pickle.dump(self.feature_scaler, f)
        with open(d / "target_scaler.pkl", "wb") as f:
            pickle.dump(self.target_scaler, f)

        meta = {
            "targets": self.targets,
            "feature_cols": self.feature_cols,
            "transforms_applied": self.transforms_applied,
            "clip_bounds": self.clip_bounds,
            "iqr_multiplier": self.iqr_multiplier,
        }
        with open(d / "preprocessing_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved preprocessing artefacts to %s", d)
        return d

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> "PreprocessingPipeline":
        """Restore a fitted pipeline from disk."""
        d = Path(artifacts_dir)

        with open(d / "preprocessing_meta.json") as f:
            meta = json.load(f)

        pipe = cls(targets=meta["targets"], iqr_multiplier=meta.get("iqr_multiplier", 3.0))
        pipe.feature_cols = meta["feature_cols"]
        pipe.transforms_applied = meta["transforms_applied"]
        pipe.clip_bounds = meta.get("clip_bounds", {})

        with open(d / "feature_scaler.pkl", "rb") as fh:
            pipe.feature_scaler = pickle.load(fh)
        with open(d / "target_scaler.pkl", "rb") as fh:
            pipe.target_scaler = pickle.load(fh)

        pipe._is_fitted = True
        logger.info("Loaded preprocessing pipeline from %s", d)
        return pipe


# Sequence / split helpers

def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70 / 15 / 15 split (no shuffle)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    logger.info("Split — train: %d  val: %d  test: %d", len(train), len(val), len(test))
    return train, val, test


def create_sequences(
    data: pd.DataFrame | np.ndarray,
    seq_len: int,
    forecast_h: int,
    feature_cols: List[str] | None = None,
    target_cols: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window (X, y) arrays for time-series models.

    Parameters
    ----------
    data : DataFrame or ndarray
        If DataFrame, *feature_cols* and *target_cols* select columns.
        If ndarray, all columns are features and last ``len(target_cols)``
        are targets.
    seq_len : int
        Look-back window length.
    forecast_h : int
        Steps ahead to predict (usually 1).

    Returns
    -------
    X : np.ndarray, shape (n_samples, seq_len, n_features)
    y : np.ndarray, shape (n_samples, n_targets)
    """
    if isinstance(data, pd.DataFrame):
        if feature_cols is None or target_cols is None:
            raise ValueError("feature_cols and target_cols required for DataFrame input")
        X_all = data[feature_cols].values.astype(np.float32)
        Y_all = data[target_cols].values.astype(np.float32)
    else:
        # assume last n cols are targets
        n_targets = len(target_cols) if target_cols else 1
        X_all = data[:, :-n_targets].astype(np.float32)
        Y_all = data[:, -n_targets:].astype(np.float32)

    xs, ys = [], []
    for i in range(len(X_all) - seq_len - forecast_h + 1):
        xs.append(X_all[i : i + seq_len])
        ys.append(Y_all[i + seq_len + forecast_h - 1])

    return np.array(xs), np.array(ys)


def prepare_pipeline(scaler_type: str = "robust") -> RobustScaler | StandardScaler:
    """Return an unfitted scaler instance (legacy compat)."""
    if scaler_type == "robust":
        return RobustScaler()
    return StandardScaler()


# Convenience entry-point

def run_preprocessing(
    raw_dir: str | Path,
    artifacts_dir: str | Path | None = None,
) -> Tuple[pd.DataFrame, PreprocessingPipeline, List[str]]:
    """Load raw lake data, clean, engineer targets, scale, and return.

    Returns
    -------
    df_final : pd.DataFrame
        Fully preprocessed & scaled DataFrame (features + targets).
    pipeline : PreprocessingPipeline
        Fitted pipeline (for later inverse-transforms / saving).
    feature_cols : list[str]
        List of feature column names.
    """
    pipe = PreprocessingPipeline()
    df = pipe.load_raw(raw_dir)
    df = pipe.clean(df)
    # Note: feature engineering is a separate step — see feature_engineering.py
    return df, pipe

