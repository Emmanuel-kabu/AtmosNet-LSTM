"""Model evaluation utilities.

Supports both single-target (legacy) and multi-target evaluation
using the new ``PreprocessingPipeline`` for inverse-transformation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

from atm_forecast.data.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


# ── Multi-target evaluation (primary) ────────────────────────────────

def evaluate_multi_target(
    model: keras.Model,
    X_test: NDArray[Any],
    y_test: NDArray[Any],
    pipeline: PreprocessingPipeline,
    targets: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a model on multi-target outputs.

    Parameters
    ----------
    model : keras.Model
        Trained model with ``n_targets`` outputs.
    X_test : NDArray
        Test input sequences ``(n_samples, seq_len, n_features)``.
    y_test : NDArray
        True targets ``(n_samples, n_targets)`` (scaled).
    pipeline : PreprocessingPipeline
        Fitted pipeline — used for inverse-transform.
    targets : list[str], optional
        Target column names.  Defaults to ``pipeline.targets``.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{target_name: {"MAE": …, "RMSE": …, "R2": …}}``.
    """
    if targets is None:
        targets = pipeline.targets

    preds = model.predict(X_test, verbose=0)
    preds_inv = pipeline.inverse_transform_targets(preds)
    trues_inv = pipeline.inverse_transform_targets(y_test)

    results: Dict[str, Dict[str, float]] = {}
    for i, col in enumerate(targets):
        mae = float(mean_absolute_error(trues_inv[:, i], preds_inv[:, i]))
        rmse = float(np.sqrt(mean_squared_error(trues_inv[:, i], preds_inv[:, i])))
        r2 = float(r2_score(trues_inv[:, i], preds_inv[:, i]))
        results[col] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    log_results_table(results)
    return results


def summarise_metrics(
    per_target: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute average MAE, RMSE, and R² across all targets."""
    maes = [m["MAE"] for m in per_target.values()]
    rmses = [m["RMSE"] for m in per_target.values()]
    r2s = [m["R2"] for m in per_target.values()]
    return {
        "avg_MAE": float(np.mean(maes)),
        "avg_RMSE": float(np.mean(rmses)),
        "avg_R2": float(np.mean(r2s)),
    }


def metrics_to_dataframe(
    per_target: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Convert per-target metrics dict to a tidy DataFrame."""
    rows = []
    for target, vals in per_target.items():
        rows.append({"target": target, **vals})
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


# ── Legacy single-target evaluation ─────────────────────────────────

def evaluate_model(
    model: keras.Model,
    X_test: NDArray[Any],
    y_test: NDArray[Any],
    scaler=None,
    target_col_idx: int = 0,
) -> dict[str, float]:
    """Evaluate a trained model (single-target, legacy API).

    Parameters
    ----------
    model : keras.Model
        Trained model.
    X_test : NDArray
        Test input sequences.
    y_test : NDArray
        True target values (scaled).
    scaler : sklearn transformer, optional
        If provided, inverse-transform predictions for real-scale metrics.
    target_col_idx : int
        Index of the target column in the scaler's feature set.

    Returns
    -------
    dict[str, float]
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test, verbose=0).flatten()

    metrics: dict[str, float] = {
        "test_mse": float(mean_squared_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
    }

    if scaler is not None:
        try:
            n_features = scaler.transform(
                np.zeros((1, scaler.n_features_in_))
            ).shape[1]
            dummy_pred = np.zeros((len(y_pred), n_features))
            dummy_true = np.zeros((len(y_test), n_features))

            dummy_pred[:, target_col_idx] = y_pred
            dummy_true[:, target_col_idx] = y_test

            y_pred_inv = scaler.inverse_transform(dummy_pred)[:, target_col_idx]
            y_true_inv = scaler.inverse_transform(dummy_true)[:, target_col_idx]

            metrics["test_mae_original_scale"] = float(
                mean_absolute_error(y_true_inv, y_pred_inv)
            )
            metrics["test_rmse_original_scale"] = float(
                np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
            )
        except Exception:
            logger.warning(
                "Could not inverse-transform predictions", exc_info=True
            )

    logger.info("Evaluation metrics: %s", metrics)
    return metrics


# ── Pretty-print helper ─────────────────────────────────────────────

def log_results_table(
    results: Dict[str, Dict[str, float]],
) -> None:
    """Log a formatted table of per-target metrics."""
    header = f"  {'Target':<45s}  {'MAE':>10s}  {'RMSE':>10s}  {'R²':>10s}"
    logger.info(header)
    logger.info("  " + "-" * len(header))
    for target, vals in results.items():
        logger.info(
            "  %-45s  %10.4f  %10.4f  %10.4f",
            target, vals["MAE"], vals["RMSE"], vals["R2"],
        )
