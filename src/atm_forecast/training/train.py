"""End-to-end training pipeline.

Supports two data modes:

1. **Legacy** — pass a CSV/Parquet path via ``data_path``.
2. **Data-lake** — read pre-engineered features from
   ``data/lake/features/`` by setting ``use_lake=True``.  Features are
   already computed, so the pipeline skips feature engineering and goes
   straight to split → scale → sequence → train.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from atm_forecast.config import get_settings
from atm_forecast.data.ingestion import load_data
from atm_forecast.data.preprocessing import create_sequences, prepare_pipeline, split_data
from atm_forecast.features.engineering import (
    add_cyclical_time_features,
    add_lag_features,
    add_rolling_features,
)
from atm_forecast.models.lstm import build_lstm_model
from atm_forecast.models.registry import save_model
from atm_forecast.monitoring.wandb_tracker import (
    finish_wandb,
    init_wandb,
    log_artifact,
    log_metrics,
    log_model_summary,
    log_predictions,
    log_training_history,
)
from atm_forecast.monitoring.evidently_monitor import (
    generate_data_drift_report,
    generate_model_performance_report,
)
from atm_forecast.training.evaluate import evaluate_model

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_lake_features(settings) -> pd.DataFrame:
    """Load the full features layer from the data lake."""
    from atm_forecast.data.lake import read_all_partitions, LAYER_FEATURES

    lake_root = settings.lake_root
    logger.info("Loading features from data lake at %s", lake_root)
    return read_all_partitions(lake_root, LAYER_FEATURES)


def run_training(
    data_path: str | None = None,
    target_column: str = "temperature",
    output_dir: str | None = None,
    *,
    use_lake: bool = False,
    **overrides,
) -> Path:
    """Execute the full training pipeline.

    Parameters
    ----------
    data_path : str | None
        Path to a raw CSV/Parquet file.  Ignored when *use_lake* is True.
    target_column : str
        Name of the column to forecast.
    output_dir : str | None
        Where to save artefacts.  Falls back to ``settings.model_dir``.
    use_lake : bool
        If *True*, load pre-engineered features from the data lake
        (``data/lake/features/``) and skip the feature-engineering step.
    **overrides
        Any ``Settings`` fields to override (e.g. ``epochs=100``).

    Returns
    -------
    Path
        Directory where the trained model was saved.
    """
    settings = get_settings()
    epochs = overrides.get("epochs", settings.epochs)
    batch_size = overrides.get("batch_size", settings.batch_size)
    seq_len = overrides.get("sequence_length", settings.sequence_length)
    horizon = overrides.get("forecast_horizon", settings.forecast_horizon)
    out = Path(output_dir) if output_dir else settings.model_dir

    # ── W&B initialisation ───────────────────────────────────────────
    wandb_run = None
    if settings.wandb_enabled:
        wandb_run = init_wandb(
            project=settings.wandb_project,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "forecast_horizon": horizon,
                "lstm_units": settings.lstm_units,
                "dropout_rate": settings.dropout_rate,
                "learning_rate": settings.learning_rate,
                "target_column": target_column,
                "data_path": data_path,
            },
            run_name=f"train-{target_column}-ep{epochs}",
            tags=["training", target_column] + (["lake"] if use_lake else []),
        )

    try:
        # 1. Load data — either from the lake or a raw file ───────────
        if use_lake:
            logger.info("Loading pre-engineered features from data lake")
            df = _load_lake_features(settings)
            if target_column not in df.columns:
                raise KeyError(
                    f"Target column {target_column!r} not in lake features: "
                    f"{list(df.columns)}"
                )
            # Lake features already include lags/rolling/cyclical → skip FE
        else:
            if data_path is None:
                raise ValueError(
                    "data_path is required when use_lake=False"
                )
            logger.info("Loading data from %s", data_path)
            df = load_data(data_path)

            # 2. Feature engineering (only in legacy mode)
            logger.info("Engineering features")
            df = add_lag_features(
                df, columns=[target_column], lags=[1, 2, 3, 6, 12, 24]
            )
            df = add_rolling_features(
                df, columns=[target_column], windows=[6, 12, 24]
            )
            try:
                df = add_cyclical_time_features(df)
            except TypeError:
                logger.warning(
                    "Index is not DatetimeIndex — skipping cyclical features"
                )
            df = df.dropna()

        # 3. Split data
        train_df, val_df, test_df = split_data(
            df, test_ratio=settings.test_split, val_ratio=settings.validation_split
        )

        # ── Evidently: baseline data drift between train and val ─────
        logger.info("Running Evidently data drift report (train vs val)")
        drift_report = generate_data_drift_report(
            reference=train_df,
            current=val_df,
            output_path=settings.artifacts_dir / "reports" / "train_val_data_drift.html",
        )

        # 4. Scale
        pipeline = prepare_pipeline(scaler_type="minmax")

        feature_cols = list(train_df.columns)
        target_idx = feature_cols.index(target_column)

        train_scaled = pipeline.fit_transform(train_df[feature_cols].values)
        val_scaled = pipeline.transform(val_df[feature_cols].values)
        test_scaled = pipeline.transform(test_df[feature_cols].values)

        # 5. Create sequences
        X_train, y_train = create_sequences(train_scaled, seq_len, horizon, target_idx)
        X_val, y_val = create_sequences(val_scaled, seq_len, horizon, target_idx)
        X_test, y_test = create_sequences(test_scaled, seq_len, horizon, target_idx)

        logger.info(
            "Sequence shapes — X_train: %s, X_val: %s, X_test: %s",
            X_train.shape,
            X_val.shape,
            X_test.shape,
        )

        if settings.wandb_enabled:
            log_metrics({
                "data/train_samples": len(X_train),
                "data/val_samples": len(X_val),
                "data/test_samples": len(X_test),
                "data/n_features": X_train.shape[2],
            })

        # 6. Build model
        model = build_lstm_model(
            input_shape=(seq_len, X_train.shape[2]),
            lstm_units=settings.lstm_units,
            dropout_rate=settings.dropout_rate,
            learning_rate=settings.learning_rate,
            forecast_horizon=horizon,
        )

        if settings.wandb_enabled:
            log_model_summary(model)

        # 7. Train
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        # Add W&B callback if available
        if settings.wandb_enabled:
            try:
                from wandb.integration.keras import WandbMetricsLogger
                callbacks.append(WandbMetricsLogger(log_freq="epoch"))
            except ImportError:
                logger.debug("WandbMetricsLogger not available, using manual logging")

        logger.info("Starting training — epochs=%d, batch_size=%d", epochs, batch_size)
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Log training history to W&B
        if settings.wandb_enabled:
            log_training_history(history)

        # 8. Evaluate
        metrics = evaluate_model(model, X_test, y_test, pipeline, target_idx)

        if settings.wandb_enabled:
            log_metrics(metrics)

            # Log prediction vs actual table
            y_pred_test = model.predict(X_test, verbose=0).flatten()
            log_predictions(y_test, y_pred_test, table_name="test_predictions")

        # ── Evidently: model performance on test set ─────────────────
        logger.info("Running Evidently model performance report")
        y_pred_all = model.predict(X_test, verbose=0).flatten()
        perf_df = pd.DataFrame({
            "target": y_test.flatten(),
            "prediction": y_pred_all,
        })
        # Use train predictions as reference
        y_pred_train = model.predict(X_train, verbose=0).flatten()
        ref_df = pd.DataFrame({
            "target": y_train.flatten(),
            "prediction": y_pred_train,
        })
        generate_model_performance_report(
            reference=ref_df,
            current=perf_df,
            target_column="target",
            prediction_column="prediction",
            output_path=settings.artifacts_dir / "reports" / "model_performance.html",
        )

        # 9. Save
        metadata = {
            "target_column": target_column,
            "sequence_length": seq_len,
            "forecast_horizon": horizon,
            "feature_columns": feature_cols,
            "epochs_trained": len(history.history["loss"]),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            **metrics,
        }

        save_dir = save_model(model, out, scaler=pipeline, metadata=metadata)
        logger.info("Training complete — artefacts saved to %s", save_dir)

        # Log model artefact to W&B
        if settings.wandb_enabled:
            log_artifact(
                filepath=str(save_dir / "model.keras"),
                name="lstm-forecast-model",
                artifact_type="model",
            )

        return save_dir

    finally:
        # Always finish the W&B run
        if settings.wandb_enabled:
            finish_wandb()
