"""
End-to-end Training Pipeline — AtmosNet
========================================

Orchestrates the full workflow:

1. Load raw data from the data lake (``data/lake/raw/``)
2. Preprocess (clean, clip, fill NaN)
3. Feature engineering (lags, rolling, cyclical, interactions, …)
4. Target transforms + scaling (fit on train split only)
5. Sequence creation (sliding window)
6. Build & train three models (Bi-LSTM, TCN, TFT)
7. Evaluate on the test split
8. Log everything to **MLflow** (params, per-target metrics, datasets,
   models, artefacts)
9. Log to **W&B** — gradient/weight monitoring via ``wandb.watch``,
   per-epoch metrics, prediction tables, and model artefacts
10. Persist pipeline artefacts & trained models to disk

Data modes
----------
* **Data-lake** (default) — reads ``data/lake/raw/``, runs full
  preprocessing + feature engineering.
* **Legacy** — pass ``data_path`` to a CSV/Parquet file.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from atm_forecast.config import get_settings
from atm_forecast.data.preprocessing import (
    PreprocessingPipeline,
    create_sequences,
    split_data,
    TARGETS,
)
from atm_forecast.features.feature_engineering import run_feature_engineering
from atm_forecast.models.Model_initialiazatin import build_model
from atm_forecast.models.registry import save_model
from atm_forecast.monitoring.mlflow_tracker import MLflowTracker
from atm_forecast.monitoring.wandb_tracker import (
    finish_wandb,
    init_wandb,
    log_artifact,
    log_metrics as wandb_log_metrics,
    log_model_summary,
    log_predictions,
    log_training_history as wandb_log_history,
)

logger = logging.getLogger(__name__)


# ── Evaluation helper ─────────────────────────────────────────────────

def evaluate_on_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pipeline: PreprocessingPipeline,
    targets: List[str],
) -> Dict[str, Dict[str, float]]:
    """Evaluate a trained model on the test split.

    Returns per-target metrics dict:
    ``{target: {MAE, RMSE, R2}}`` in original-scale units.
    """
    preds = model.predict(X_test, verbose=0)
    preds_inv = pipeline.inverse_transform_targets(preds)
    trues_inv = pipeline.inverse_transform_targets(y_test)

    metrics: Dict[str, Dict[str, float]] = {}
    for i, t in enumerate(targets):
        mae = float(mean_absolute_error(trues_inv[:, i], preds_inv[:, i]))
        rmse = float(np.sqrt(mean_squared_error(trues_inv[:, i], preds_inv[:, i])))
        r2 = float(r2_score(trues_inv[:, i], preds_inv[:, i]))
        metrics[t] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return metrics


# ── Main entry-point ──────────────────────────────────────────────────

def run_training(
    data_path: str | None = None,
    output_dir: str | None = None,
    *,
    models_to_train: List[Literal["bilstm", "tcn", "tft"]] | None = None,
    use_lake: bool = True,
    mlflow_experiment: str = "atm-forecast",
    mlflow_tracking_uri: str = "mlruns",
    **overrides,
) -> Path:
    """Execute the full training pipeline for all models.

    Parameters
    ----------
    data_path : str | None
        Path to raw CSV/Parquet.  Ignored when ``use_lake=True``.
    output_dir : str | None
        Where to save artefacts.  Falls back to ``settings.artifacts_dir``.
    models_to_train : list[str]
        Which models to train.  Default: all three.
    use_lake : bool
        If True, load from ``data/lake/raw/``.
    mlflow_experiment / mlflow_tracking_uri : str
        MLflow configuration.
    **overrides
        Override ``Settings`` fields (e.g. ``epochs=100``).

    Returns
    -------
    Path
        Artefacts output directory.
    """
    settings = get_settings()
    seq_len = overrides.get("sequence_length", settings.sequence_length)
    forecast_h = overrides.get("forecast_horizon", settings.forecast_horizon)
    epochs = overrides.get("epochs", settings.epochs)
    batch_size = overrides.get("batch_size", settings.batch_size)
    lr = overrides.get("learning_rate", settings.learning_rate)
    out_dir = Path(output_dir) if output_dir else settings.artifacts_dir

    if models_to_train is None:
        models_to_train = ["bilstm", "tcn", "tft"]

    targets = TARGETS

    # ── 1. Initialise MLflow tracker ─────────────────────────────
    tracker = MLflowTracker(
        experiment_name=mlflow_experiment,
        tracking_uri=mlflow_tracking_uri,
    )
    tracker.start_run(
        run_name=f"train-{'_'.join(models_to_train)}",
        tags={
            "models": ",".join(models_to_train),
            "data_source": "lake" if use_lake else "file",
        },
    )

    # ── 1b. Initialise W&B (weight & gradient monitoring) ────────
    wandb_run = None
    if settings.wandb_enabled:
        wandb_run = init_wandb(
            project=settings.wandb_project,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "forecast_horizon": forecast_h,
                "learning_rate": lr,
                "models": models_to_train,
                "n_targets": len(targets),
            },
            run_name=f"train-{'_'.join(models_to_train)}-ep{epochs}",
            tags=["training", "multi-target"]
            + (["lake"] if use_lake else [])
            + models_to_train,
        )

    try:
        # ── 2. Load & preprocess ─────────────────────────────────
        pipeline = PreprocessingPipeline(targets=targets)

        if use_lake:
            raw_dir = settings.lake_root / "raw"
            logger.info("Loading data lake from %s", raw_dir)
            df = pipeline.load_raw(raw_dir)
        else:
            if data_path is None:
                raise ValueError("data_path required when use_lake=False")
            logger.info("Loading data from %s", data_path)
            df = (
                pd.read_parquet(data_path)
                if str(data_path).endswith(".parquet")
                else pd.read_csv(data_path)
            )

        df = pipeline.clean(df)

        # ── 3. Feature engineering ───────────────────────────────
        df = run_feature_engineering(df, targets=targets)

        # ── 4. Target transforms + scaling (fit on train) ────────
        train_raw, val_raw, test_raw = split_data(df)

        # Fit transforms & scalers on TRAIN only
        train_transformed = pipeline.fit_transform_targets(train_raw)
        train_scaled, feature_cols = pipeline.fit_scalers(train_transformed)

        # Apply same transform to val & test
        val_scaled = pipeline.transform(val_raw)
        test_scaled = pipeline.transform(test_raw)

        # Save pipeline artefacts
        pipeline.save(out_dir / "preprocessing")

        # Log dataset to MLflow
        tracker.log_dataset(train_scaled, name="train_features", context="training")
        tracker.log_dataset(test_scaled, name="test_features", context="testing")
        tracker.log_artifacts_dir(
            str(out_dir / "preprocessing"), artifact_path="preprocessing",
        )

        # ── 5. Sequence creation ─────────────────────────────────
        X_train, y_train = create_sequences(
            train_scaled, seq_len, forecast_h, feature_cols, targets,
        )
        X_val, y_val = create_sequences(
            val_scaled, seq_len, forecast_h, feature_cols, targets,
        )
        X_test, y_test = create_sequences(
            test_scaled, seq_len, forecast_h, feature_cols, targets,
        )

        n_features = X_train.shape[2]
        n_targets = y_train.shape[1]

        logger.info(
            "Sequences — train: %s, val: %s, test: %s",
            X_train.shape, X_val.shape, X_test.shape,
        )

        # Log common hyper-parameters
        tracker.log_params({
            "seq_len": seq_len,
            "forecast_horizon": forecast_h,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "n_features": n_features,
            "n_targets": n_targets,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "targets": targets,
            "transforms": pipeline.transforms_applied,
        })

        # Log data stats to W&B
        if settings.wandb_enabled:
            wandb_log_metrics({
                "data/train_samples": len(X_train),
                "data/val_samples": len(X_val),
                "data/test_samples": len(X_test),
                "data/n_features": n_features,
                "data/n_targets": n_targets,
            })

        # ── 6. Train each model ──────────────────────────────────
        callbacks_factory = lambda: [
            EarlyStopping(
                monitor="val_loss", patience=8,
                restore_best_weights=True, verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", patience=3,
                factor=0.5, min_lr=1e-6, verbose=1,
            ),
        ]

        all_results: Dict[str, Dict] = {}

        for model_name in models_to_train:
            logger.info("=" * 60)
            logger.info("Training model: %s", model_name.upper())
            logger.info("=" * 60)

            model = build_model(
                model_name, seq_len, n_features, n_targets,
                learning_rate=lr,
            )

            # W&B: watch weights, biases, and gradients
            if settings.wandb_enabled:
                log_model_summary(model)

            # Build callbacks — include WandbMetricsLogger if available
            cbs = callbacks_factory()
            if settings.wandb_enabled:
                try:
                    from wandb.integration.keras import WandbMetricsLogger
                    cbs.append(WandbMetricsLogger(log_freq="epoch"))
                except ImportError:
                    logger.debug(
                        "WandbMetricsLogger not available — using manual logging"
                    )

            t0 = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=cbs,
                verbose=1,
            )
            train_time = time.time() - t0

            # Evaluate
            metrics = evaluate_on_test(model, X_test, y_test, pipeline, targets)

            # Compute summary
            avg_r2 = float(np.mean([m["R2"] for m in metrics.values()]))
            avg_mae = float(np.mean([m["MAE"] for m in metrics.values()]))
            avg_rmse = float(np.mean([m["RMSE"] for m in metrics.values()]))

            # Log to MLflow
            tracker.log_per_target_metrics(model_name, metrics)
            tracker.log_metrics({
                f"{model_name}/avg_R2": avg_r2,
                f"{model_name}/avg_MAE": avg_mae,
                f"{model_name}/avg_RMSE": avg_rmse,
                f"{model_name}/train_time_s": train_time,
                f"{model_name}/epochs_trained": len(history.history["loss"]),
                f"{model_name}/best_val_loss": float(
                    min(history.history["val_loss"])
                ),
            })
            tracker.log_training_history(history, prefix=f"{model_name}/")
            tracker.log_model(model, model_name, X_sample=X_train, register=True)

            # ── W&B: log metrics, history, predictions, artefact ─
            if settings.wandb_enabled:
                wandb_log_metrics({
                    f"{model_name}/avg_R2": avg_r2,
                    f"{model_name}/avg_MAE": avg_mae,
                    f"{model_name}/avg_RMSE": avg_rmse,
                    f"{model_name}/train_time_s": train_time,
                    f"{model_name}/epochs_trained": len(history.history["loss"]),
                    f"{model_name}/best_val_loss": float(
                        min(history.history["val_loss"])
                    ),
                })
                # Per-target metrics to W&B
                for tgt, vals in metrics.items():
                    for metric_key, metric_val in vals.items():
                        short = tgt.replace("air_quality_", "")
                        wandb_log_metrics(
                            {f"{model_name}/{short}/{metric_key}": metric_val}
                        )
                wandb_log_history(history)
                # Prediction table (subset)
                preds_test = model.predict(X_test, verbose=0)
                preds_inv = pipeline.inverse_transform_targets(preds_test)
                trues_inv = pipeline.inverse_transform_targets(y_test)
                log_predictions(
                    trues_inv[:, 0], preds_inv[:, 0],
                    table_name=f"{model_name}_predictions",
                )

            # Save model to disk
            model_out = out_dir / "models" / model_name
            save_model(
                model, model_out,
                metadata={
                    "model_name": model_name,
                    "targets": targets,
                    "feature_cols": feature_cols,
                    "seq_len": seq_len,
                    "forecast_horizon": forecast_h,
                    "avg_r2": avg_r2,
                    "avg_mae": avg_mae,
                    "avg_rmse": avg_rmse,
                    "train_time_s": train_time,
                    "per_target": metrics,
                },
            )

            all_results[model_name] = {
                "model": model,
                "history": history,
                "metrics": metrics,
                "avg_r2": avg_r2,
                "train_time": train_time,
            }

            logger.info(
                "%s done — avg R²=%.4f, avg MAE=%.4f, time=%.1fs",
                model_name.upper(), avg_r2, avg_mae, train_time,
            )

        # ── 7. Summary ───────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE — SUMMARY")
        logger.info("=" * 60)
        best_model = max(all_results, key=lambda k: all_results[k]["avg_r2"])
        for name, res in all_results.items():
            marker = " *" if name == best_model else ""
            logger.info(
                "  %-8s  avg R2=%.4f  time=%.1fs%s",
                name, res["avg_r2"], res["train_time"], marker,
            )

        tracker.log_metrics(
            {"best_model_avg_r2": all_results[best_model]["avg_r2"]}
        )
        tracker.log_params({"best_model": best_model})

        # Log best model artefact to W&B
        if settings.wandb_enabled:
            best_dir = out_dir / "models" / best_model / "model.keras"
            if best_dir.exists():
                log_artifact(
                    filepath=str(best_dir),
                    name=f"{best_model}-best-model",
                    artifact_type="model",
                )
            wandb_log_metrics(
                {"best_model_avg_r2": all_results[best_model]["avg_r2"]}
            )

        return out_dir

    finally:
        tracker.end_run()
        if settings.wandb_enabled:
            finish_wandb()
