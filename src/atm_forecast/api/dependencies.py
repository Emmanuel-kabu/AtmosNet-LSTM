"""Dependency injection for FastAPI endpoints.

Provides:
- ``PipelineService``  — wraps the fitted PreprocessingPipeline.
- ``ModelRegistry``    — lazily loads / caches multiple model architectures.
- ``ModelService``     — backward-compat wrapper used by legacy ``/predict``.
- ``DataService``      — location / raw-data helpers for data endpoints.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from atm_forecast.config import get_settings
from atm_forecast.data.preprocessing import (
    PreprocessingPipeline,
    TARGETS,
    create_sequences,
)
from atm_forecast.models.registry import load_model as _load_model_from_disk

logger = logging.getLogger(__name__)

# Display-name lookup (mirrors frontend constants)
MODEL_NAMES = {"bilstm": "Bi-LSTM", "tcn": "TCN", "tft": "TFT"}

TARGET_DISPLAY = {
    "air_quality_Carbon_Monoxide": "Carbon Monoxide (CO)",
    "air_quality_Ozone": "Ozone (O\u2083)",
    "air_quality_Nitrogen_dioxide": "Nitrogen Dioxide (NO\u2082)",
    "air_quality_Sulphur_dioxide": "Sulphur Dioxide (SO\u2082)",
    "air_quality_PM2.5": "PM 2.5",
    "air_quality_PM10": "PM 10",
    "temperature_celsius": "Temperature (\u00b0C)",
}

TARGET_UNITS = {
    "air_quality_Carbon_Monoxide": "\u00b5g/m\u00b3",
    "air_quality_Ozone": "\u00b5g/m\u00b3",
    "air_quality_Nitrogen_dioxide": "\u00b5g/m\u00b3",
    "air_quality_Sulphur_dioxide": "\u00b5g/m\u00b3",
    "air_quality_PM2.5": "\u00b5g/m\u00b3",
    "air_quality_PM10": "\u00b5g/m\u00b3",
    "temperature_celsius": "\u00b0C",
}


# =====================================================================
# Pipeline Service
# =====================================================================

class PipelineService:
    """Wraps the fitted ``PreprocessingPipeline`` for inference.

    Handles: scaling, inverse-transform, sequence creation.
    """

    def __init__(self, pipeline: PreprocessingPipeline) -> None:
        self.pipeline = pipeline

    @property
    def feature_cols(self) -> List[str]:
        return self.pipeline.feature_cols

    @property
    def targets(self) -> List[str]:
        return self.pipeline.targets

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.transform(df)

    def inverse_transform_targets(self, arr: np.ndarray) -> np.ndarray:
        return self.pipeline.inverse_transform_targets(arr)

    def create_sequences(
        self,
        df_scaled: pd.DataFrame,
        seq_len: int,
        forecast_h: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return create_sequences(
            df_scaled,
            seq_len,
            forecast_h=forecast_h,
            feature_cols=self.feature_cols,
            target_cols=self.targets,
        )


@lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
    """Load and cache the preprocessing pipeline singleton."""
    settings = get_settings()
    preprocess_dir = settings.artifacts_dir / "preprocessing"

    if not preprocess_dir.exists():
        raise FileNotFoundError(
            f"Preprocessing artefacts not found at {preprocess_dir}. "
            "Run the training pipeline first."
        )

    pipeline = PreprocessingPipeline.load(preprocess_dir)
    logger.info("PipelineService loaded from %s", preprocess_dir)
    return PipelineService(pipeline)


# =====================================================================
# Model Registry (multi-model)
# =====================================================================

class ModelRegistry:
    """Discover and lazily load trained model architectures.

    Scans ``artifacts/models/`` for sub-directories containing
    ``model.keras`` and caches them on first access.
    """

    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self._cache: Dict[str, Tuple[Any, dict]] = {}

    # ── discovery ─────────────────────────────────────────────────

    def available_models(self) -> Dict[str, Path]:
        """Return {name: dir} of models that have model.keras on disk."""
        found: Dict[str, Path] = {}
        if self.models_root.exists():
            for child in self.models_root.iterdir():
                if child.is_dir() and (child / "model.keras").exists():
                    found[child.name] = child
        return found

    def list_metadata(self) -> Dict[str, dict]:
        """Return {name: metadata_dict} for every available model."""
        out: Dict[str, dict] = {}
        for name, path in self.available_models().items():
            meta_path = path / "metadata.json"
            if meta_path.exists():
                out[name] = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                out[name] = {}
        return out

    def best_model_name(self) -> Optional[str]:
        """Return the model with the highest avg_r2, or None."""
        best, best_r2 = None, -float("inf")
        for name, meta in self.list_metadata().items():
            r2 = meta.get("avg_r2")
            if r2 is not None and r2 > best_r2:
                best, best_r2 = name, r2
        return best

    # ── loading (lazy + cached) ───────────────────────────────────

    def get(self, name: str) -> Tuple[Any, dict]:
        """Return ``(keras_model, metadata)`` for *name*.

        Raises ``KeyError`` if the model name is not available.
        """
        if name in self._cache:
            return self._cache[name]

        available = self.available_models()
        if name not in available:
            raise KeyError(
                f"Model '{name}' not found.  Available: {list(available.keys())}"
            )

        model, _scaler, metadata = _load_model_from_disk(
            available[name], load_scaler=False,
        )
        self._cache[name] = (model, metadata)
        logger.info("ModelRegistry: loaded '%s' from %s", name, available[name])
        return model, metadata

    def get_default(self) -> Tuple[str, Any, dict]:
        """Return ``(name, model, metadata)`` for the best (or first) model."""
        available = self.available_models()
        if not available:
            raise FileNotFoundError("No trained models in artifacts/models/")

        name = self.best_model_name() or next(iter(available))
        model, metadata = self.get(name)
        return name, model, metadata


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """Return a cached ModelRegistry singleton."""
    settings = get_settings()
    models_root = settings.artifacts_dir / "models"
    registry = ModelRegistry(models_root)
    logger.info(
        "ModelRegistry initialised — %d models available",
        len(registry.available_models()),
    )
    return registry


# =====================================================================
# Data Service (locations, raw data)
# =====================================================================

class DataService:
    """Read-only helpers for the data lake (locations, summaries)."""

    def __init__(self, lake_raw: Path) -> None:
        self.lake_raw = lake_raw
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_parquet(self.lake_raw)
            if "date" in self._df.columns:
                self._df["date"] = pd.to_datetime(self._df["date"].astype(str))
        return self._df

    def locations(self) -> pd.DataFrame:
        """Return unique locations with lat/lon and country."""
        df = self._load()
        if "location_name" not in df.columns:
            return pd.DataFrame(
                columns=["location_name", "country", "latitude", "longitude"]
            )
        return (
            df.groupby("location_name")
            .agg(
                country=("country", "first"),
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
            )
            .reset_index()
        )

    def countries(self) -> List[str]:
        df = self._load()
        if "country" not in df.columns:
            return []
        return sorted(df["country"].unique().tolist())

    def summary(self) -> dict:
        df = self._load()
        date_col = df["date"] if "date" in df.columns else None
        target_stats = {}
        for tgt in TARGETS:
            if tgt in df.columns:
                s = df[tgt].dropna()
                target_stats[tgt] = {
                    "mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                    "min": round(float(s.min()), 4),
                    "max": round(float(s.max()), 4),
                }
        return {
            "total_rows": len(df),
            "date_range": (
                [str(date_col.min().date()), str(date_col.max().date())]
                if date_col is not None
                else []
            ),
            "n_countries": df["country"].nunique() if "country" in df.columns else 0,
            "n_locations": df["location_name"].nunique() if "location_name" in df.columns else 0,
            "columns": df.columns.tolist(),
            "target_stats": target_stats,
        }

    def raw_dataframe(self) -> pd.DataFrame:
        """Return the full raw DataFrame (for Evidently, etc.)."""
        return self._load()


@lru_cache(maxsize=1)
def get_data_service() -> DataService:
    """Return a cached DataService singleton."""
    settings = get_settings()
    lake_raw = settings.lake_root / "raw"
    return DataService(lake_raw)


# =====================================================================
# Inference helper
# =====================================================================

def run_forecast(
    model,
    pipeline_svc: PipelineService,
    df_engineered: pd.DataFrame,
    seq_len: int,
    n_days: int,
) -> pd.DataFrame:
    """Full inference: scale → sequence → predict → inverse-transform.

    Returns a DataFrame with columns = TARGETS and rows = forecast steps.
    """
    df_scaled = pipeline_svc.transform(df_engineered)
    X, _ = pipeline_svc.create_sequences(df_scaled, seq_len)

    if len(X) == 0:
        return pd.DataFrame(columns=TARGETS)

    X_last = X[-n_days:]
    preds_scaled = model.predict(X_last, verbose=0)
    preds_original = pipeline_svc.inverse_transform_targets(preds_scaled)

    return pd.DataFrame(preds_original, columns=TARGETS)


def prepare_engineered_data() -> pd.DataFrame:
    """Load raw data, clean, and run feature engineering.

    Returns the fully-engineered DataFrame ready for scaling/sequencing.
    """
    from atm_forecast.features.feature_engineering import run_feature_engineering

    settings = get_settings()
    lake_raw = settings.lake_root / "raw"

    pipe = PreprocessingPipeline(targets=TARGETS)
    df = pipe.load_raw(lake_raw)
    df_clean = pipe.clean(df)
    df_engineered = run_feature_engineering(df_clean.copy(), targets=TARGETS)
    return df_engineered


# ---- cached version for repeated calls within a process ----
_engineered_cache: Optional[pd.DataFrame] = None


def get_engineered_data() -> pd.DataFrame:
    """Return cached engineered data (computed once per process)."""
    global _engineered_cache
    if _engineered_cache is None:
        _engineered_cache = prepare_engineered_data()
    return _engineered_cache


# =====================================================================
# Legacy compat: ModelService + get_model_service
# =====================================================================

class ModelService:
    """Backward-compatible wrapper — now delegates to ModelRegistry.

    Kept so existing ``/predict`` consumers do not break.
    """

    def __init__(
        self,
        model,
        metadata: dict | None = None,
        model_name: str = "unknown",
    ) -> None:
        self.model = model
        self.metadata = metadata or {}
        self.model_version: str | None = self.metadata.get("saved_at")
        self.model_name = model_name

    def predict(self, features: NDArray) -> NDArray:
        """Run raw inference (no pipeline scaling — caller handles that)."""
        if features.ndim == 2:
            features = features[np.newaxis, ...]
        preds = self.model.predict(features, verbose=0)
        return preds


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Load the default (best) model as a legacy ModelService."""
    registry = get_model_registry()
    name, model, metadata = registry.get_default()
    logger.info("Legacy ModelService using '%s'", name)
    return ModelService(model=model, metadata=metadata, model_name=name)
