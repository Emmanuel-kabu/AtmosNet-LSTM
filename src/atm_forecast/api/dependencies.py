"""Dependency injection for FastAPI endpoints."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
import tensorflow as tf
from tensorflow import keras
from atm_forecast.config import get_settings

logger = logging.getLogger(__name__)


class ModelService:
    """Wrapper around a trained model + scaler for inference."""

    def __init__(
        self,
        model: keras.Model,
        scaler: Any | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.metadata = metadata or {}
        self.model_version: str | None = self.metadata.get("saved_at")

    def predict(self, features: NDArray) -> NDArray:
        """Run inference on a feature array.

        Parameters
        ----------
        features : NDArray
            2-D array (timesteps, n_features). Will be scaled and reshaped.

        Returns
        -------
        NDArray
            1-D prediction array.
        """
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Model expects (batch, seq_len, features)
        if features.ndim == 2:
            features = features[np.newaxis, ...]

        preds = self.model.predict(features, verbose=0)
        return preds.flatten()


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Load and cache the model service singleton."""
    settings = get_settings()
    model_dir = Path(settings.model_dir)

    model_path = model_dir / "model.keras"
    if not model_path.exists():
        logger.warning("No model found at %s — serving will fail on prediction calls", model_path)
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = keras.models.load_model(model_path)

    scaler = None
    scaler_path = model_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    import json

    metadata: dict = {}
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    logger.info("Model service loaded from %s", model_dir)
    return ModelService(model=model, scaler=scaler, metadata=metadata)
