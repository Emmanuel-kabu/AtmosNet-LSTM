"""
MLflow Experiment Tracking for AtmosNet
=======================================

Thin wrapper around MLflow that handles:
* Experiment / run lifecycle
* Logging hyper-parameters, metrics, and artefacts
* Registering trained models in the MLflow Model Registry
* Logging engineered datasets as MLflow Artifacts

Usage
-----
>>> from atm_forecast.monitoring.mlflow_tracker import MLflowTracker
>>> tracker = MLflowTracker(experiment_name="atm-forecast")
>>> tracker.start_run(run_name="bilstm-v1")
>>> tracker.log_params({"lr": 1e-3, "epochs": 50})
>>> tracker.log_metrics({"test_r2": 0.92, "test_mae": 1.23})
>>> tracker.log_artifact("artifacts/preprocessing_meta.json")
>>> tracker.log_model(model, "bilstm")
>>> tracker.end_run()
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.tensorflow
    from mlflow.models.signature import infer_signature

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed — tracking calls will be no-ops")


class MLflowTracker:
    """Manage an MLflow experiment run.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment (auto-created if missing).
    tracking_uri : str
        MLflow tracking server URI.  Defaults to a local ``mlruns/`` folder.
    """

    def __init__(
        self,
        experiment_name: str = "atm-forecast",
        tracking_uri: str = "mlruns",
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._run = None

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(
                "MLflow tracker initialised — experiment=%s, uri=%s",
                experiment_name, tracking_uri,
            )

    # ── Run lifecycle ─────────────────────────────────────────────

    def start_run(
        self,
        run_name: str | None = None,
        tags: Dict[str, str] | None = None,
        nested: bool = False,
    ) -> Optional[Any]:
        """Start (or resume) an MLflow run."""
        if not MLFLOW_AVAILABLE:
            return None
        self._run = mlflow.start_run(run_name=run_name, nested=nested)
        if tags:
            mlflow.set_tags(tags)
        logger.info("Started MLflow run: %s", run_name or self._run.info.run_id)
        return self._run

    def end_run(self):
        if MLFLOW_AVAILABLE and self._run is not None:
            mlflow.end_run()
            logger.info("Ended MLflow run")
            self._run = None

    # ── Logging helpers ───────────────────────────────────────────

    def log_params(self, params: Dict[str, Any]):
        """Log a dictionary of hyper-parameters."""
        if not MLFLOW_AVAILABLE:
            return
        # MLflow only accepts str/int/float — flatten nested dicts
        flat = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                flat[k] = str(v)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = str(vv)
            else:
                flat[k] = v
        mlflow.log_params(flat)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int | None = None,
    ):
        """Log a dictionary of scalar metrics."""
        if not MLFLOW_AVAILABLE:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                mlflow.log_metric(k, float(v), step=step)

    def log_per_target_metrics(
        self,
        model_name: str,
        metrics: Dict[str, Dict[str, float]],
    ):
        """Log per-target MAE / RMSE / R² under a model prefix.

        Parameters
        ----------
        model_name : str  e.g. ``"bilstm"``
        metrics : dict of ``{target: {MAE: …, RMSE: …, R2: …}}``
        """
        if not MLFLOW_AVAILABLE:
            return
        flat = {}
        for target, m in metrics.items():
            short_t = target.replace("air_quality_", "aq_")
            for metric_name, val in m.items():
                flat[f"{model_name}/{short_t}/{metric_name}"] = float(val)
        mlflow.log_metrics(flat)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None):
        """Log a file or directory as an artefact."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
        logger.info("Logged artefact: %s", path)

    def log_artifacts_dir(self, dir_path: str | Path, artifact_path: str | None = None):
        """Log an entire directory."""
        if not MLFLOW_AVAILABLE:
            return
        mlflow.log_artifacts(str(dir_path), artifact_path=artifact_path)
        logger.info("Logged artefacts dir: %s", dir_path)

    # ── Dataset logging ───────────────────────────────────────────

    def log_dataset(
        self,
        df: pd.DataFrame,
        name: str = "features",
        context: str = "training",
    ):
        """Log a DataFrame as an MLflow dataset artefact (Parquet + schema).

        Parameters
        ----------
        df : pd.DataFrame
        name : str
            Artefact name (used as the file stem).
        context : str
            One of ``"training"``, ``"evaluation"``, ``"testing"``.
        """
        if not MLFLOW_AVAILABLE:
            return

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / f"{name}.parquet"
            df.to_parquet(parquet_path, index=False)

            # Log the parquet file itself
            mlflow.log_artifact(str(parquet_path), artifact_path="datasets")

            # Also log schema info
            schema = {
                "name": name,
                "context": context,
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {c: str(d) for c, d in df.dtypes.items()},
            }
            schema_path = Path(tmp) / f"{name}_schema.json"
            schema_path.write_text(json.dumps(schema, indent=2))
            mlflow.log_artifact(str(schema_path), artifact_path="datasets")

        logger.info("Logged dataset '%s' (%s) — shape %s", name, context, df.shape)

    # ── Model logging ─────────────────────────────────────────────

    def log_model(
        self,
        model,
        name: str,
        X_sample: np.ndarray | None = None,
        register: bool = False,
    ):
        """Log a Keras model as an MLflow artefact.

        Parameters
        ----------
        model : keras.Model
        name : str
            Artefact path / registered-model name.
        X_sample : np.ndarray, optional
            Sample input to infer the model signature.
        register : bool
            If True, register the model in MLflow Model Registry.
        """
        if not MLFLOW_AVAILABLE:
            return

        signature = None
        if X_sample is not None:
            try:
                preds = model.predict(X_sample[:5], verbose=0)
                signature = infer_signature(X_sample[:5], preds)
            except Exception:
                logger.warning("Could not infer model signature", exc_info=True)

        registered_name = name if register else None
        mlflow.tensorflow.log_model(
            model,
            artifact_path=name,
            signature=signature,
            registered_model_name=registered_name,
        )
        logger.info("Logged model '%s' (registered=%s)", name, register)

    # ── Training history ──────────────────────────────────────────

    def log_training_history(self, history, prefix: str = ""):
        """Log epoch-level training metrics from a Keras History object."""
        if not MLFLOW_AVAILABLE:
            return
        for epoch_idx in range(len(history.history.get("loss", []))):
            step_metrics = {}
            for key, values in history.history.items():
                metric_key = f"{prefix}{key}" if prefix else key
                step_metrics[metric_key] = float(values[epoch_idx])
            mlflow.log_metrics(step_metrics, step=epoch_idx + 1)
