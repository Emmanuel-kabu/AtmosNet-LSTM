"""Model evaluation utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

logger = logging.getLogger(__name__)


def evaluate_model(
    model: keras.Model,
    X_test: NDArray[Any],
    y_test: NDArray[Any],
    scaler=None,
    target_col_idx: int = 0,
) -> dict[str, float]:
    """Evaluate a trained model and return standard regression metrics.

    Parameters
    ----------
    model : keras.Model
        Trained model.
    X_test : NDArray
        Test input sequences.
    y_test : NDArray
        True target values (scaled).
    scaler : sklearn pipeline, optional
        If provided, inverse-transform predictions for real-scale metrics.
    target_col_idx : int
        Index of the target column in the scaler's feature set.

    Returns
    -------
    dict[str, float]
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Compute scaled metrics
    metrics: dict[str, float] = {
        "test_mse": float(mean_squared_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
    }

    # Inverse-transform for real-scale metrics if scaler available
    if scaler is not None:
        try:
            n_features = scaler.transform(np.zeros((1, scaler.n_features_in_))).shape[1]
            dummy_pred = np.zeros((len(y_pred), n_features))
            dummy_true = np.zeros((len(y_test), n_features))

            dummy_pred[:, target_col_idx] = y_pred
            dummy_true[:, target_col_idx] = y_test

            y_pred_inv = scaler.inverse_transform(dummy_pred)[:, target_col_idx]
            y_true_inv = scaler.inverse_transform(dummy_true)[:, target_col_idx]

            metrics["test_mae_original_scale"] = float(mean_absolute_error(y_true_inv, y_pred_inv))
            metrics["test_rmse_original_scale"] = float(
                np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
            )
        except Exception:
            logger.warning("Could not inverse-transform predictions", exc_info=True)

    logger.info("Evaluation metrics: %s", metrics)
    return metrics
