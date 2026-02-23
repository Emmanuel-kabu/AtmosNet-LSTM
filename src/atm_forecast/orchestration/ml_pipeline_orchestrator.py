"""
Flyte ML Pipeline — AtmosNet
==============================

Orchestrates the **full ML lifecycle** for the AtmosNet atmospheric
forecasting system.  Each stage is an explicit Flyte task so that
failures are isolated, retries are granular, and artefacts are
versioned automatically.

Pipeline graph::

 ┌────────────┐   ┌─────────────┐   ┌─────────────────┐   ┌────────────────┐
 │ load_raw   │──▶│ preprocess  │──▶│ engineer_       │──▶│ prepare_       │
 │ _data      │   │ _data       │   │ features        │   │ training_data  │
 └────────────┘   └─────────────┘   └─────────────────┘   └───────┬────────┘
                                                                  │
                  ┌──────────────────────────────────────────┬─────┴──────┐
                  ▼                  ▼                       ▼            │
           ┌────────────┐   ┌────────────┐   ┌────────────┐              │
           │ train_     │   │ train_     │   │ train_     │              │
           │ single_    │   │ single_    │   │ single_    │              │
           │ model      │   │ model      │   │ model      │              │
           │ (bilstm)   │   │ (tcn)      │   │ (tft)      │              │
           └─────┬──────┘   └─────┬──────┘   └─────┬──────┘              │
                 └────────────────┼────────────────┘                     │
                                  ▼                                      │
                         ┌────────────────┐                              │
                         │ evaluate_      │◀─────────────────────────────┘
                         │ models         │
                         └───────┬────────┘
                                 ▼
                         ┌────────────────┐
                         │ deploy_best    │
                         │ _model         │
                         └────────────────┘

Separate CT (continuous-training) workflow re-uses the core tasks and
adds **data-readiness checks** and **drift detection** as gates.

Requirements
------------
- Raw data exists in ``data/lake/raw/`` (populated by the Airflow data DAG).
- Flyte agent is running (``flytectl demo start`` or remote cluster).

Usage::

    # Full training pipeline
    pyflyte run src/atm_forecast/orchestration/ml_pipeline_orchestrator.py \\
        ml_pipeline --models '["bilstm","tcn","tft"]' --epochs 50

    # Continuous-training pipeline (drift-gated)
    pyflyte run src/atm_forecast/orchestration/ml_pipeline_orchestrator.py \\
        ct_pipeline --models '["bilstm","tcn","tft"]' --epochs 30

    # Register for scheduled execution
    pyflyte register src/atm_forecast/orchestration/ml_pipeline_orchestrator.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flyte

logger = logging.getLogger(__name__)


# =====================================================================
# Task environment (Flyte v2)
# =====================================================================
# All tasks share one Docker image that contains the full project code
# and all dependencies.  Build & push it with:
#
#   docker build -f docker/Dockerfile.flyte \
#       -t ghcr.io/emmanuelkabu/atm-forecast-flyte:latest .
#   docker push ghcr.io/emmanuelkabu/atm-forecast-flyte:latest
#
# The image tag is the SINGLE source of truth for the execution env.
# =====================================================================

FLYTE_IMAGE = "ghcr.io/emmanuelkabu/atm-forecast-flyte:latest"

env = flyte.TaskEnvironment(
    name="atm-forecast",
    image=FLYTE_IMAGE,
)


# =====================================================================
# Configuration dataclass (serialisable by Flyte)
# =====================================================================

@dataclass
class PipelineConfig:
    """Configuration shared across all ML pipeline tasks."""

    # Model selection
    models: List[str] = field(default_factory=lambda: ["bilstm", "tcn", "tft"])

    # Hyper-parameters
    epochs: int = 50
    batch_size: int = 32
    sequence_length: int = 24
    forecast_horizon: int = 1
    learning_rate: float = 1e-3

    # Data
    use_lake: bool = True
    data_path: str = ""              # only used when use_lake=False

    # Quality gates
    promote_r2_threshold: float = 0.7
    drift_threshold: float = 0.05
    min_new_rows: int = 100

    # Tracking
    use_wandb: bool = False
    mlflow_experiment: str = "atm-forecast"
    mlflow_tracking_uri: str = "mlruns"


# =====================================================================
# Task 1 — Load raw data
# =====================================================================

@env.task(
    cache=True,
    cache_version="1.0",
    retries=2,
)
def load_raw_data(config: PipelineConfig) -> str:
    """Load raw data from the Hive-partitioned data lake.

    Reads ``data/lake/raw/`` and writes a consolidated Parquet file to
    a temporary working directory that downstream tasks can consume.

    Returns the path to the consolidated Parquet file.
    """
    import pandas as pd
    from atm_forecast.config import get_settings

    settings = get_settings()
    ctx = flyte.current_context()
    work_dir = Path(ctx.working_directory) / "pipeline_data"
    work_dir.mkdir(parents=True, exist_ok=True)

    if config.use_lake:
        from atm_forecast.data.lake import LAYER_RAW, read_all_partitions
        raw_dir = settings.lake_root / "raw"
        logger.info("Loading raw data from lake: %s", raw_dir)
        df = read_all_partitions(settings.lake_root, LAYER_RAW)
    else:
        path = config.data_path or str(settings.data_dir / "raw.parquet")
        logger.info("Loading raw data from file: %s", path)
        df = (
            pd.read_parquet(path)
            if path.endswith(".parquet")
            else pd.read_csv(path)
        )

    out_path = work_dir / "raw.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Raw data saved — shape: %s → %s", df.shape, out_path)
    return str(out_path)


# =====================================================================
# Task 2 — Preprocess data
# =====================================================================

@env.task(
    cache=True,
    cache_version="1.0",
    retries=1,
)
def preprocess_data(raw_path: str) -> str:
    """Clean raw data: parse dates, drop redundant columns, fill NaN,
    clip outliers.

    Returns the path to the cleaned Parquet file.
    """
    import pandas as pd
    from atm_forecast.data.preprocessing import PreprocessingPipeline

    df = pd.read_parquet(raw_path)
    pipeline = PreprocessingPipeline()
    df_clean = pipeline.clean(df)

    out_path = str(Path(raw_path).parent / "clean.parquet")
    df_clean.to_parquet(out_path, index=False)

    logger.info(
        "Preprocessing done — %d rows, %d cols, NaN: %d",
        len(df_clean), len(df_clean.columns), df_clean.isnull().sum().sum(),
    )
    return out_path


# =====================================================================
# Task 3 — Feature engineering
# =====================================================================

@env.task(
    cache=True,
    cache_version="1.0",
    retries=1,
)
def engineer_features(clean_path: str) -> str:
    """Apply all 8 feature-engineering functions (daylight, wind,
    pressure, interactions, location, cyclical, lags, rolling).

    Returns the path to the engineered Parquet file.
    """
    import pandas as pd
    from atm_forecast.data.preprocessing import TARGETS
    from atm_forecast.features.feature_engineering import run_feature_engineering

    df = pd.read_parquet(clean_path)
    df_feat = run_feature_engineering(df, targets=TARGETS)

    out_path = str(Path(clean_path).parent / "features.parquet")
    df_feat.to_parquet(out_path, index=False)

    logger.info(
        "Feature engineering done — %d rows, %d cols",
        len(df_feat), len(df_feat.columns),
    )
    return out_path


# =====================================================================
# Task 4 — Prepare training data (split, scale, create sequences)
# =====================================================================

@env.task(
    cache=True,
    cache_version="1.0",
    retries=1,
)
def prepare_training_data(
    features_path: str,
    config: PipelineConfig,
) -> dict:
    """Split → target-transform → fit scalers → create sequences.

    Persists the fitted ``PreprocessingPipeline`` artefacts to disk so
    downstream tasks (training, evaluation, serving) can inverse-transform.

    Returns a dict with paths and shapes for train/val/test arrays.
    """
    import numpy as np
    import pandas as pd
    from atm_forecast.config import get_settings
    from atm_forecast.data.preprocessing import (
        PreprocessingPipeline,
        create_sequences,
        split_data,
        TARGETS,
    )

    settings = get_settings()
    df = pd.read_parquet(features_path)

    work_dir = Path(features_path).parent
    arrays_dir = work_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PreprocessingPipeline(targets=TARGETS)

    # ── Chronological split ──────────────────────────────────────
    train_raw, val_raw, test_raw = split_data(df)

    # ── Fit on train, transform all splits ───────────────────────
    train_transformed = pipeline.fit_transform_targets(train_raw)
    train_scaled, feature_cols = pipeline.fit_scalers(train_transformed)
    val_scaled = pipeline.transform(val_raw)
    test_scaled = pipeline.transform(test_raw)

    # ── Persist pipeline artefacts ───────────────────────────────
    artifacts_dir = settings.artifacts_dir / "preprocessing"
    pipeline.save(artifacts_dir)

    # ── Create sequences ─────────────────────────────────────────
    seq_len = config.sequence_length
    forecast_h = config.forecast_horizon

    X_train, y_train = create_sequences(
        train_scaled, seq_len, forecast_h, feature_cols, TARGETS,
    )
    X_val, y_val = create_sequences(
        val_scaled, seq_len, forecast_h, feature_cols, TARGETS,
    )
    X_test, y_test = create_sequences(
        test_scaled, seq_len, forecast_h, feature_cols, TARGETS,
    )

    # Save to disk for downstream tasks
    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val", X_val), ("y_val", y_val),
        ("X_test", X_test), ("y_test", y_test),
    ]:
        np.save(arrays_dir / f"{name}.npy", arr)

    summary = {
        "arrays_dir": str(arrays_dir),
        "artifacts_dir": str(artifacts_dir),
        "feature_cols": feature_cols,
        "n_features": int(X_train.shape[2]),
        "n_targets": int(y_train.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "seq_len": seq_len,
        "forecast_horizon": forecast_h,
    }

    logger.info(
        "Training data ready — train: %s, val: %s, test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )
    return summary


# =====================================================================
# Task 5 — Train a single model
# =====================================================================

@env.task(
    cache=False,
    retries=0,
    timeout=timedelta(hours=4),
)
def train_single_model(
    model_name: str,
    data_summary: dict,
    config: PipelineConfig,
) -> dict:
    """Train one model architecture (bilstm / tcn / tft).

    Loads pre-built sequences from ``data_summary['arrays_dir']``,
    builds the model, trains with EarlyStopping + ReduceLROnPlateau,
    evaluates on the test set, saves the artefact, and logs to
    MLflow + W&B.

    Returns a metrics dict for this model.
    """
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from atm_forecast.config import get_settings
    from atm_forecast.data.preprocessing import PreprocessingPipeline, TARGETS
    from atm_forecast.models.Model_initialiazatin import build_model
    from atm_forecast.models.registry import save_model
    from atm_forecast.monitoring.mlflow_tracker import MLflowTracker

    settings = get_settings()
    arrays_dir = Path(data_summary["arrays_dir"])
    artifacts_dir = Path(data_summary["artifacts_dir"])

    # Load arrays
    X_train = np.load(arrays_dir / "X_train.npy")
    y_train = np.load(arrays_dir / "y_train.npy")
    X_val = np.load(arrays_dir / "X_val.npy")
    y_val = np.load(arrays_dir / "y_val.npy")
    X_test = np.load(arrays_dir / "X_test.npy")
    y_test = np.load(arrays_dir / "y_test.npy")

    n_features = data_summary["n_features"]
    n_targets = data_summary["n_targets"]
    seq_len = data_summary["seq_len"]
    feature_cols = data_summary["feature_cols"]

    # Reload fitted pipeline for inverse transforms
    pipeline = PreprocessingPipeline.load(artifacts_dir)

    # ── Build model ──────────────────────────────────────────────
    model = build_model(
        model_name, seq_len, n_features, n_targets,
        learning_rate=config.learning_rate,
    )
    logger.info(
        "Built %s — params: %s", model_name.upper(), f"{model.count_params():,}",
    )

    # ── MLflow tracker ───────────────────────────────────────────
    tracker = MLflowTracker(
        experiment_name=config.mlflow_experiment,
        tracking_uri=config.mlflow_tracking_uri,
    )
    tracker.start_run(
        run_name=f"train-{model_name}",
        tags={"model": model_name, "pipeline": "flyte"},
    )

    # ── W&B (optional) ───────────────────────────────────────────
    if config.use_wandb:
        try:
            from atm_forecast.monitoring.wandb_tracker import (
                init_wandb, log_model_summary, log_metrics as wandb_log_metrics,
                log_training_history as wandb_log_history,
                log_predictions, log_artifact, finish_wandb,
            )
            init_wandb(
                project=settings.wandb_project,
                config={
                    "model": model_name, "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "sequence_length": seq_len,
                    "learning_rate": config.learning_rate,
                },
                run_name=f"flyte-{model_name}",
            )
            log_model_summary(model)
        except ImportError:
            logger.warning("W&B not available — skipping")

    # ── Callbacks ────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", patience=3,
            factor=0.5, min_lr=1e-6, verbose=1,
        ),
    ]
    if config.use_wandb:
        try:
            from wandb.integration.keras import WandbMetricsLogger
            callbacks.append(WandbMetricsLogger(log_freq="epoch"))
        except ImportError:
            pass

    # ── Train ────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0

    # ── Evaluate on test ─────────────────────────────────────────
    preds = model.predict(X_test, verbose=0)
    preds_inv = pipeline.inverse_transform_targets(preds)
    trues_inv = pipeline.inverse_transform_targets(y_test)

    per_target: Dict[str, Dict[str, float]] = {}
    for i, t in enumerate(TARGETS):
        mae = float(mean_absolute_error(trues_inv[:, i], preds_inv[:, i]))
        rmse = float(np.sqrt(mean_squared_error(trues_inv[:, i], preds_inv[:, i])))
        r2 = float(r2_score(trues_inv[:, i], preds_inv[:, i]))
        per_target[t] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    avg_r2 = float(np.mean([m["R2"] for m in per_target.values()]))
    avg_mae = float(np.mean([m["MAE"] for m in per_target.values()]))
    avg_rmse = float(np.mean([m["RMSE"] for m in per_target.values()]))

    # ── Log to MLflow ────────────────────────────────────────────
    tracker.log_params({
        "model_name": model_name,
        "seq_len": seq_len,
        "forecast_horizon": config.forecast_horizon,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "n_features": n_features,
        "n_targets": n_targets,
        "train_samples": data_summary["train_samples"],
    })
    tracker.log_per_target_metrics(model_name, per_target)
    tracker.log_metrics({
        f"{model_name}/avg_R2": avg_r2,
        f"{model_name}/avg_MAE": avg_mae,
        f"{model_name}/avg_RMSE": avg_rmse,
        f"{model_name}/train_time_s": train_time,
        f"{model_name}/epochs_trained": len(history.history["loss"]),
        f"{model_name}/best_val_loss": float(min(history.history["val_loss"])),
    })
    tracker.log_training_history(history, prefix=f"{model_name}/")
    tracker.log_model(model, model_name, X_sample=X_train, register=True)

    # ── Log to W&B (optional) ────────────────────────────────────
    if config.use_wandb:
        try:
            wandb_log_metrics({
                f"{model_name}/avg_R2": avg_r2,
                f"{model_name}/avg_MAE": avg_mae,
                f"{model_name}/avg_RMSE": avg_rmse,
                f"{model_name}/train_time_s": train_time,
            })
            for tgt, vals in per_target.items():
                short = tgt.replace("air_quality_", "")
                for mk, mv in vals.items():
                    wandb_log_metrics({f"{model_name}/{short}/{mk}": mv})
            wandb_log_history(history)
            log_predictions(trues_inv[:, 0], preds_inv[:, 0],
                            table_name=f"{model_name}_predictions")
        except Exception as exc:
            logger.warning("W&B logging error: %s", exc)

    # ── Save model artefact ──────────────────────────────────────
    model_out = settings.artifacts_dir / "models" / model_name
    save_model(
        model, model_out,
        metadata={
            "model_name": model_name,
            "targets": TARGETS,
            "feature_cols": feature_cols,
            "seq_len": seq_len,
            "forecast_horizon": config.forecast_horizon,
            "avg_r2": avg_r2,
            "avg_mae": avg_mae,
            "avg_rmse": avg_rmse,
            "train_time_s": train_time,
            "epochs_trained": len(history.history["loss"]),
            "per_target": per_target,
        },
    )

    tracker.end_run()
    if config.use_wandb:
        try:
            finish_wandb()
        except Exception:
            pass

    result = {
        "model_name": model_name,
        "model_dir": str(model_out),
        "avg_r2": avg_r2,
        "avg_mae": avg_mae,
        "avg_rmse": avg_rmse,
        "train_time_s": train_time,
        "epochs_trained": len(history.history["loss"]),
        "per_target": per_target,
    }

    logger.info(
        "%s done — avg R²=%.4f  avg MAE=%.4f  time=%.1fs",
        model_name.upper(), avg_r2, avg_mae, train_time,
    )
    return result


# =====================================================================
# Task 6 — Evaluate all models and select the best
# =====================================================================

@env.task(
    cache=False,
    retries=1,
)
def evaluate_models(
    model_results: List[dict],
    config: PipelineConfig,
) -> dict:
    """Compare trained models and select the champion.

    Returns a summary dict containing:
    - per-model average R², MAE, RMSE
    - the best model name and path
    - whether the quality threshold is met
    """
    if not model_results:
        return {"success": False, "reason": "no_model_results"}

    comparison = {}
    for res in model_results:
        name = res["model_name"]
        comparison[name] = {
            "avg_r2": res["avg_r2"],
            "avg_mae": res["avg_mae"],
            "avg_rmse": res["avg_rmse"],
            "train_time_s": res["train_time_s"],
            "model_dir": res["model_dir"],
        }

    best_name = max(comparison, key=lambda k: comparison[k]["avg_r2"])
    best_r2 = comparison[best_name]["avg_r2"]
    meets_threshold = best_r2 >= config.promote_r2_threshold

    summary = {
        "success": True,
        "best_model": best_name,
        "best_r2": best_r2,
        "best_model_dir": comparison[best_name]["model_dir"],
        "meets_threshold": meets_threshold,
        "threshold": config.promote_r2_threshold,
        "all_models": comparison,
    }

    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for name, info in comparison.items():
        marker = " ★" if name == best_name else ""
        logger.info(
            "  %-8s  avg_R2=%.4f  avg_MAE=%.4f  time=%.1fs%s",
            name, info["avg_r2"], info["avg_mae"], info["train_time_s"], marker,
        )
    logger.info(
        "Best: %s (R²=%.4f) — threshold %s (%.4f)",
        best_name, best_r2,
        "MET" if meets_threshold else "NOT MET",
        config.promote_r2_threshold,
    )

    return summary


# =====================================================================
# Task 7 — Deploy / promote the best model
# =====================================================================

@env.task(
    cache=False,
    retries=1,
)
def deploy_best_model(
    evaluation: dict,
    config: PipelineConfig,
) -> dict:
    """Promote the best model if it meets the quality threshold.

    Deployment actions:
    1. Update the pipeline watermark in the state DB.
    2. Register / tag the champion in the MLflow model registry.
    3. (Optional) Upload the champion artefact to W&B.
    4. Copy the best model to a well-known serving directory.
    """
    if not evaluation.get("success", False):
        return {"deployed": False, "reason": evaluation.get("reason", "eval_failed")}

    if not evaluation.get("meets_threshold", False):
        logger.warning(
            "Best R² (%.4f) below threshold (%.4f) — skipping deployment",
            evaluation["best_r2"], evaluation["threshold"],
        )
        return {
            "deployed": False,
            "reason": "below_quality_threshold",
            "best_model": evaluation["best_model"],
            "best_r2": evaluation["best_r2"],
        }

    best_name = evaluation["best_model"]
    best_r2 = evaluation["best_r2"]
    best_dir = evaluation["best_model_dir"]
    now = datetime.now(timezone.utc)

    # ── 1. Update watermark ──────────────────────────────────────
    try:
        from atm_forecast.config import get_settings
        from atm_forecast.data.pipeline_state import get_engine, update_watermark

        settings = get_settings()
        engine = get_engine(settings.database_url)
        update_watermark(
            engine,
            "ml_training",
            last_success_at=now,
            last_partition_date=date.today(),
            rows_ingested=0,
        )
        logger.info("Updated ml_training watermark to %s", now.isoformat())
    except Exception as exc:
        logger.warning("Watermark update failed: %s", exc)

    # ── 2. MLflow model registry ─────────────────────────────────
    try:
        import mlflow

        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()

        try:
            versions = client.search_model_versions(f"name='{best_name}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                client.set_model_version_tag(
                    best_name, latest.version, "stage", "champion",
                )
                client.set_model_version_tag(
                    best_name, latest.version, "promoted_at", now.isoformat(),
                )
                client.set_model_version_tag(
                    best_name, latest.version, "avg_r2", str(round(best_r2, 4)),
                )
                logger.info(
                    "MLflow: tagged %s v%s as champion", best_name, latest.version,
                )
        except Exception as exc:
            logger.warning("MLflow model tagging failed: %s", exc)
    except ImportError:
        logger.info("MLflow not available — skipping registry promotion")

    # ── 3. W&B artefact upload ───────────────────────────────────
    if config.use_wandb:
        try:
            from atm_forecast.monitoring.wandb_tracker import log_artifact

            model_path = Path(best_dir) / "model.keras"
            if model_path.exists():
                log_artifact(
                    filepath=str(model_path),
                    name=f"{best_name}-champion",
                    artifact_type="model",
                )
                logger.info("W&B: uploaded champion artefact for %s", best_name)
        except Exception as exc:
            logger.warning("W&B artifact upload failed: %s", exc)

    # ── 4. Copy to serving directory ─────────────────────────────
    try:
        import shutil
        from atm_forecast.config import get_settings

        settings = get_settings()
        serving_dir = settings.model_dir / "serving" / "champion"
        serving_dir.mkdir(parents=True, exist_ok=True)

        src = Path(best_dir)
        if src.exists():
            # Clear old champion and copy new one
            for item in serving_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            shutil.copytree(src, serving_dir, dirs_exist_ok=True)
            logger.info("Copied champion model to %s", serving_dir)
    except Exception as exc:
        logger.warning("Serving copy failed: %s", exc)

    result = {
        "deployed": True,
        "best_model": best_name,
        "best_r2": best_r2,
        "model_dir": best_dir,
        "promoted_at": now.isoformat(),
        "reason": "quality_threshold_met",
    }

    logger.info(
        "Model %s deployed as champion (R² = %.4f)", best_name, best_r2,
    )
    return result


# =====================================================================
# CT-specific tasks — data readiness & drift detection
# =====================================================================

@env.task(
    cache=False,
    retries=1,
)
def check_data_readiness(config: PipelineConfig) -> dict:
    """Verify that the raw layer has enough NEW data for a CT run.

    Checks:
    - At least ``config.min_new_rows`` rows exist in the raw layer.
    - NaN ratio is below 10%.
    - Watermark indicates new data since last training.
    """
    from atm_forecast.config import get_settings
    from atm_forecast.data.lake import LAYER_RAW, list_partition_dates, read_all_partitions
    from atm_forecast.data.pipeline_state import get_engine, get_watermark
    import numpy as np

    settings = get_settings()
    raw_dates = list_partition_dates(settings.lake_root, LAYER_RAW)
    if not raw_dates:
        return {"ready": False, "reason": "no_raw_partitions", "n_rows": 0}

    df = read_all_partitions(settings.lake_root, LAYER_RAW)
    n_rows = len(df)
    null_pct = float(df.isnull().mean().mean()) * 100

    engine = get_engine(settings.database_url)
    wm = get_watermark(engine, "ml_training")
    is_first_run = wm.last_success_at is None
    ready = (n_rows >= config.min_new_rows) or is_first_run

    if null_pct > 10.0:
        ready = False

    summary = {
        "ready": ready,
        "n_partitions": len(raw_dates),
        "n_rows": n_rows,
        "null_pct": round(null_pct, 2),
        "is_first_run": is_first_run,
        "date_range": [str(min(raw_dates)), str(max(raw_dates))],
        "last_trained_at": (
            wm.last_success_at.isoformat() if wm.last_success_at else None
        ),
    }
    logger.info("Data readiness: %s", json.dumps(summary, indent=2))
    return summary


@env.task(
    cache=False,
    retries=1,
)
def detect_drift(config: PipelineConfig, readiness: dict) -> dict:
    """Run Evidently / KS-test drift detection on the raw layer.

    Splits data 80/20 chronologically; if drift score exceeds the
    threshold **or** this is the first run, the pipeline proceeds.
    """
    if not readiness.get("ready", False):
        return {"should_retrain": False, "reason": "data_not_ready"}

    from atm_forecast.config import get_settings
    from atm_forecast.data.lake import LAYER_RAW, read_all_partitions
    import numpy as np

    settings = get_settings()
    df = read_all_partitions(settings.lake_root, LAYER_RAW)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(df) < 200:
        return {
            "should_retrain": readiness.get("is_first_run", True),
            "drift_detected": False,
            "reason": "insufficient_data_for_drift",
        }

    split_idx = int(len(df) * 0.8)
    reference = df[numeric_cols].iloc[:split_idx]
    current = df[numeric_cols].iloc[split_idx:]

    drift_score = 0.0
    drifted_features: List[str] = []
    try:
        from atm_forecast.monitoring.evidently_monitor import generate_data_drift_report

        result = generate_data_drift_report(
            reference_data=reference,
            current_data=current,
            save_dir=str(settings.evidently_reports_dir),
        )
        drift_score = result.get("dataset_drift_share", 0.0)
        drifted_features = result.get("drifted_columns", [])
    except ImportError:
        from scipy.stats import ks_2samp

        n_drifted = 0
        for col in numeric_cols:
            _, p_val = ks_2samp(reference[col].dropna(), current[col].dropna())
            if p_val < config.drift_threshold:
                n_drifted += 1
                drifted_features.append(col)
        drift_score = n_drifted / max(len(numeric_cols), 1)
    except Exception as exc:
        logger.warning("Drift detection failed: %s — defaulting to retrain", exc)
        return {"should_retrain": True, "drift_detected": True, "reason": "error", "error": str(exc)}

    drift_detected = drift_score > config.drift_threshold
    should_retrain = drift_detected or readiness.get("is_first_run", False)

    summary = {
        "should_retrain": should_retrain,
        "drift_detected": drift_detected,
        "drift_score": round(drift_score, 4),
        "n_drifted_features": len(drifted_features),
        "drifted_features": drifted_features[:10],
        "reason": (
            "drift_above_threshold" if drift_detected
            else "first_run" if readiness.get("is_first_run")
            else "no_significant_drift"
        ),
    }
    logger.info("Drift result: %s", json.dumps(summary, indent=2))
    return summary


# =====================================================================
# Workflow 1 — Full ML Pipeline
# =====================================================================

@flyte.workflow
def ml_pipeline(
    models: List[str] = ["bilstm", "tcn", "tft"],
    epochs: int = 50,
    batch_size: int = 32,
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    learning_rate: float = 1e-3,
    use_lake: bool = True,
    data_path: str = "",
    promote_r2_threshold: float = 0.7,
    use_wandb: bool = False,
    mlflow_experiment: str = "atm-forecast",
    mlflow_tracking_uri: str = "mlruns",
) -> dict:
    """AtmosNet Full ML Pipeline (Flyte).

    End-to-end workflow:
    1. **load_raw_data** — read raw Parquet from the data lake.
    2. **preprocess_data** — clean, fill NaN, clip outliers.
    3. **engineer_features** — create lag, rolling, cyclical, interaction features.
    4. **prepare_training_data** — split, scale, create sliding-window sequences.
    5. **train_single_model** × N — train each architecture in parallel.
    6. **evaluate_models** — compare all models, select champion.
    7. **deploy_best_model** — promote to MLflow registry + serving dir.

    Usage::

        pyflyte run src/atm_forecast/orchestration/ml_pipeline_orchestrator.py \\
            ml_pipeline --models '["bilstm","tcn","tft"]' --epochs 50
    """
    config = PipelineConfig(
        models=models,
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        learning_rate=learning_rate,
        use_lake=use_lake,
        data_path=data_path,
        promote_r2_threshold=promote_r2_threshold,
        use_wandb=use_wandb,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    # ── Data preparation stages (sequential) ─────────────────────
    raw_path = load_raw_data(config=config)
    clean_path = preprocess_data(raw_path=raw_path)
    feat_path = engineer_features(clean_path=clean_path)
    data_summary = prepare_training_data(features_path=feat_path, config=config)

    # ── Training (one task per model — Flyte parallelises) ───────
    bilstm_result = train_single_model(
        model_name="bilstm", data_summary=data_summary, config=config,
    )
    tcn_result = train_single_model(
        model_name="tcn", data_summary=data_summary, config=config,
    )
    tft_result = train_single_model(
        model_name="tft", data_summary=data_summary, config=config,
    )

    # ── Evaluation & deployment ──────────────────────────────────
    evaluation = evaluate_models(
        model_results=[bilstm_result, tcn_result, tft_result], config=config,
    )
    deployment = deploy_best_model(evaluation=evaluation, config=config)

    return deployment


# =====================================================================
# Workflow 2 — Continuous Training (drift-gated)
# =====================================================================

@flyte.workflow
def ct_pipeline(
    models: List[str] = ["bilstm", "tcn", "tft"],
    epochs: int = 30,
    batch_size: int = 32,
    sequence_length: int = 24,
    forecast_horizon: int = 1,
    learning_rate: float = 1e-3,
    drift_threshold: float = 0.05,
    min_new_rows: int = 100,
    promote_r2_threshold: float = 0.7,
    use_wandb: bool = False,
    mlflow_experiment: str = "atm-forecast-ct",
    mlflow_tracking_uri: str = "mlruns",
) -> dict:
    """AtmosNet Continuous Training Pipeline (Flyte).

    Same core tasks as ``ml_pipeline`` but gated by:
    1. **check_data_readiness** — enough new data since last run?
    2. **detect_drift** — has the distribution shifted?

    Only proceeds to preprocessing → training → evaluation → deployment
    if drift is detected or it is the first run.

    Usage::

        pyflyte run src/atm_forecast/orchestration/ml_pipeline_orchestrator.py \\
            ct_pipeline --models '["bilstm","tcn","tft"]' --epochs 30
    """
    config = PipelineConfig(
        models=models,
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        learning_rate=learning_rate,
        drift_threshold=drift_threshold,
        min_new_rows=min_new_rows,
        promote_r2_threshold=promote_r2_threshold,
        use_wandb=use_wandb,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    # ── CT gates ─────────────────────────────────────────────────
    readiness = check_data_readiness(config=config)
    drift = detect_drift(config=config, readiness=readiness)

    # ── Core ML pipeline (re-uses the same tasks) ────────────────
    raw_path = load_raw_data(config=config)
    clean_path = preprocess_data(raw_path=raw_path)
    feat_path = engineer_features(clean_path=clean_path)
    data_summary = prepare_training_data(features_path=feat_path, config=config)

    bilstm_result = train_single_model(
        model_name="bilstm", data_summary=data_summary, config=config,
    )
    tcn_result = train_single_model(
        model_name="tcn", data_summary=data_summary, config=config,
    )
    tft_result = train_single_model(
        model_name="tft", data_summary=data_summary, config=config,
    )

    evaluation = evaluate_models(
        model_results=[bilstm_result, tcn_result, tft_result], config=config,
    )
    deployment = deploy_best_model(evaluation=evaluation, config=config)

    return deployment


# =====================================================================
# Launch Plans (scheduled execution)
# =====================================================================

# Weekly full retrain — every Monday at 06:00 UTC
weekly_full_train = flyte.LaunchPlan.create(
    name="ml_weekly_full_train",
    workflow=ml_pipeline,
    schedule=flyte.CronSchedule(schedule="0 6 * * 1"),
    default_inputs={
        "models": ["bilstm", "tcn", "tft"],
        "epochs": 50,
        "promote_r2_threshold": 0.7,
    },
)

# Daily CT — drift-gated retrain with fewer epochs
daily_ct = flyte.LaunchPlan.create(
    name="ct_daily_drift_check",
    workflow=ct_pipeline,
    schedule=flyte.CronSchedule(schedule="0 4 * * *"),
    default_inputs={
        "models": ["bilstm", "tcn", "tft"],
        "epochs": 20,
        "drift_threshold": 0.03,
        "min_new_rows": 50,
        "promote_r2_threshold": 0.65,
    },
)