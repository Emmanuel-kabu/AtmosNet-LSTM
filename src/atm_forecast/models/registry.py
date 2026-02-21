"""Model persistence and registry utilities."""

from __future__ import annotations

import json
import joblib
import logging
from datetime import datetime, timezone
from pathlib import Path
from tensorflow import keras
logger = logging.getLogger(__name__)


def save_model(
    model: keras.Model,
    output_dir: str | Path,
    *,
    scaler=None,
    metadata: dict | None = None,
) -> Path:
    """Persist a trained model, scaler, and metadata to disk.

    Parameters
    ----------
    model : keras.Model
        Trained Keras model.
    output_dir : str | Path
        Directory to save artefacts into.
    scaler : sklearn transformer, optional
        Fitted scaler to persist alongside the model.
    metadata : dict, optional
        Additional metadata to store as JSON.

    Returns
    -------
    Path
        The output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Keras model
    model_path = output_dir / "model.keras"
    model.save(model_path)
    logger.info("Saved Keras model to %s", model_path)

    # Save scaler
    if scaler is not None:
        scaler_path = output_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info("Saved scaler to %s", scaler_path)

    # Save metadata
    meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model.name,
        "total_params": model.count_params(),
        **(metadata or {}),
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Saved metadata to %s", meta_path)

    return output_dir


def load_model(
    model_dir: str | Path,
    *,
    load_scaler: bool = True,
) -> tuple[keras.Model, object | None, dict]:
    """Load a model, scaler, and metadata from disk.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing saved artefacts.
    load_scaler : bool
        Whether to attempt loading the scaler.

    Returns
    -------
    tuple[keras.Model, object | None, dict]
        (model, scaler_or_None, metadata_dict)
    """
    model_dir = Path(model_dir)

    model = keras.models.load_model(model_dir / "model.keras")
    logger.info("Loaded model from %s", model_dir)

    scaler = None
    scaler_path = model_dir / "scaler.joblib"
    if load_scaler and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler from %s", scaler_path)

    meta_path = model_dir / "metadata.json"
    metadata: dict = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    return model, scaler, metadata
