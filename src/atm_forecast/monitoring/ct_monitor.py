"""Continuous Training Monitor — Evidently-powered drift & prediction quality.

This module provides a unified monitoring interface for the CT pipeline:

1. **Data Drift** — feature distribution shift (Evidently DataDriftPreset)
2. **Concept Drift** — target distribution shift (Evidently TargetDriftPreset)
3. **Prediction Quality** — per-model, per-location MAE/RMSE/R² tracking
4. **Reference Management** — save/load reference (training) datasets
5. **MLflow Dataset Logging** — register every feature snapshot as an MLflow dataset
6. **Prometheus Push** — populate drift & training gauges

Architecture
------------
The CT monitor sits between the Airflow data DAG and the Flyte CT pipeline:

    Airflow (data) → CT Monitor (drift check) → Flyte (retrain if needed)
                         ↓                          ↓
                    Evidently HTML reports       MLflow metrics
                    Prometheus gauges            W&B logs

"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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


# =====================================================================
# Reference Dataset Management
# =====================================================================

class ReferenceStore:
    """Save and load reference (training) datasets for drift comparison.

    Stores snapshots as Parquet under ``artifacts/reference/``.
    Each snapshot is timestamped so you can inspect what the model trained on.
    """

    def __init__(self, base_dir: str | Path = "artifacts/reference"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._latest_path = self.base_dir / "latest.parquet"
        self._metadata_path = self.base_dir / "latest_meta.json"

    def save(
        self,
        df: pd.DataFrame,
        metadata: dict | None = None,
    ) -> Path:
        """Save a reference snapshot (overwrites ``latest.parquet``)."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        versioned = self.base_dir / f"ref_{ts}.parquet"

        df.to_parquet(versioned, index=False)
        df.to_parquet(self._latest_path, index=False)

        meta = {
            "timestamp": ts,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            **(metadata or {}),
        }
        self._metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info("Reference snapshot saved: %s (%d rows)", versioned, len(df))
        return versioned

    def load(self) -> Optional[pd.DataFrame]:
        """Load the latest reference dataset (or None if none exists)."""
        if not self._latest_path.exists():
            logger.warning("No reference dataset found at %s", self._latest_path)
            return None
        df = pd.read_parquet(self._latest_path)
        logger.info("Loaded reference dataset: %d rows × %d cols", len(df), len(df.columns))
        return df

    def metadata(self) -> dict:
        """Load metadata for the latest reference snapshot."""
        if self._metadata_path.exists():
            return json.loads(self._metadata_path.read_text(encoding="utf-8"))
        return {}


# =====================================================================
# Drift Detection (Evidently-backed, with KS-test fallback)
# =====================================================================

class DriftDetector:
    """Multi-faceted drift detection: data drift + concept drift + quality tests.

    Uses Evidently as the primary engine with scipy KS-test as fallback.
    Reports are saved as HTML + JSON for auditability.
    """

    def __init__(
        self,
        reports_dir: str | Path = "artifacts/reports",
        drift_threshold: float = 0.05,
    ):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold

    def detect_data_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        numeric_only: bool = True,
    ) -> Dict[str, Any]:
        """Run full data drift detection on feature distributions.

        Checks feature-level distribution shifts using Evidently DataDriftPreset.
        Falls back to per-column KS-test if Evidently is unavailable.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        if numeric_only:
            num_cols = sorted(
                set(reference.select_dtypes(include=[np.number]).columns)
                & set(current.select_dtypes(include=[np.number]).columns)
            )
            reference = reference[num_cols]
            current = current[num_cols]

        result = self._try_evidently_data_drift(reference, current, ts)
        if result is None:
            result = self._fallback_ks_drift(reference, current)

        # Push to Prometheus
        self._push_drift_metrics(result)

        return result

    def detect_concept_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        target_columns: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Detect concept drift (target distribution shift).

        If predicted values are available, uses RegressionPreset for
        full performance drift. Otherwise, uses TargetDriftPreset.
        """
        target_columns = target_columns or TARGETS
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        results = {}
        for target in target_columns:
            if target not in reference.columns or target not in current.columns:
                continue
            try:
                from atm_forecast.monitoring.evidently_monitor import generate_target_drift_report

                output_path = self.reports_dir / f"concept_drift_{target}_{ts}.html"
                from evidently import ColumnMapping
                report = generate_target_drift_report(
                    reference=reference,
                    current=current,
                    target_column=target,
                    output_path=str(output_path),
                )
                results[target] = {
                    "drift_detected": self._extract_target_drift(report),
                    "report_path": str(output_path),
                }
            except (ImportError, Exception) as exc:
                # Fallback: KS-test on target column
                from scipy.stats import ks_2samp
                stat, p_val = ks_2samp(
                    reference[target].dropna(), current[target].dropna()
                )
                results[target] = {
                    "drift_detected": p_val < self.drift_threshold,
                    "ks_statistic": round(float(stat), 6),
                    "p_value": round(float(p_val), 6),
                }
                if isinstance(exc, ImportError):
                    logger.debug("Evidently unavailable for concept drift, using KS-test")
                else:
                    logger.warning("Evidently concept drift failed: %s", exc)

        n_drifted = sum(1 for r in results.values() if r.get("drift_detected"))
        return {
            "concept_drift_detected": n_drifted > 0,
            "n_targets_drifted": n_drifted,
            "total_targets": len(results),
            "per_target": results,
        }

    def run_data_quality_tests(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run Evidently data quality + stability test suite."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            from atm_forecast.monitoring.evidently_monitor import run_data_quality_test
            output_path = self.reports_dir / f"data_quality_{ts}.html"
            return run_data_quality_test(reference, current, output_path=str(output_path))
        except ImportError:
            logger.warning("Evidently not available for data quality tests")
            return {"status": "skipped", "reason": "evidently_not_installed"}

    def full_drift_check(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        drift_threshold: float | None = None,
    ) -> Dict[str, Any]:
        """Run all drift checks and return a unified verdict.

        Returns
        -------
        dict with keys:
            should_retrain : bool
            data_drift : dict
            concept_drift : dict
            data_quality : dict
        """
        threshold = drift_threshold or self.drift_threshold
        t0 = time.time()

        data_drift = self.detect_data_drift(reference, current)
        concept_drift = self.detect_concept_drift(reference, current)
        quality = self.run_data_quality_tests(reference, current)

        # Retrain if EITHER data drift OR concept drift is detected
        data_drifted = data_drift.get("drift_detected", False)
        concept_drifted = concept_drift.get("concept_drift_detected", False)
        should_retrain = data_drifted or concept_drifted

        elapsed = time.time() - t0
        summary = {
            "should_retrain": should_retrain,
            "data_drift": data_drift,
            "concept_drift": concept_drift,
            "data_quality": quality,
            "drift_check_time_s": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save JSON summary
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        summary_path = self.reports_dir / f"drift_summary_{ts}.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Full drift check complete in %.1fs — retrain=%s", elapsed, should_retrain)

        return summary

    # ── Internal helpers ─────────────────────────────────────────

    def _try_evidently_data_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame, ts: str,
    ) -> Optional[Dict[str, Any]]:
        """Attempt Evidently data drift report."""
        try:
            from atm_forecast.monitoring.evidently_monitor import generate_data_drift_report

            output_path = self.reports_dir / f"data_drift_{ts}.html"
            report = generate_data_drift_report(
                reference=reference,
                current=current,
                output_path=str(output_path),
            )

            # Extract metrics from Evidently's report structure
            metrics = report.get("metrics", [{}])
            drift_result = metrics[0].get("result", {}) if metrics else {}

            drift_score = drift_result.get("share_of_drifted_columns",
                          drift_result.get("dataset_drift_share", 0.0))
            n_drifted = drift_result.get("number_of_drifted_columns", 0)
            n_total = drift_result.get("number_of_columns", len(reference.columns))

            # Per-column drift details
            drifted_features = []
            col_drifts = drift_result.get("drift_by_columns", {})
            for col_name, col_info in col_drifts.items():
                if col_info.get("drift_detected", False):
                    drifted_features.append(col_name)

            return {
                "drift_detected": drift_score > self.drift_threshold,
                "drift_score": round(float(drift_score), 4),
                "n_drifted_features": n_drifted,
                "n_total_features": n_total,
                "drifted_features": drifted_features[:20],
                "report_path": str(output_path),
                "method": "evidently",
            }

        except ImportError:
            logger.info("Evidently not installed, falling back to KS-test")
            return None
        except Exception as exc:
            logger.warning("Evidently data drift report failed: %s", exc)
            return None

    def _fallback_ks_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame,
    ) -> Dict[str, Any]:
        """KS-test fallback when Evidently is unavailable."""
        from atm_forecast.monitoring.drift import check_data_drift

        per_col = check_data_drift(reference, current, threshold=self.drift_threshold)
        n_drifted = sum(1 for r in per_col.values() if r["drift_detected"])
        n_total = len(per_col)
        drift_score = n_drifted / max(n_total, 1)
        drifted_features = [c for c, r in per_col.items() if r["drift_detected"]]

        return {
            "drift_detected": drift_score > self.drift_threshold,
            "drift_score": round(drift_score, 4),
            "n_drifted_features": n_drifted,
            "n_total_features": n_total,
            "drifted_features": drifted_features[:20],
            "per_column": per_col,
            "method": "ks_test",
        }

    @staticmethod
    def _extract_target_drift(report: dict) -> bool:
        """Parse Evidently target drift report to get drift boolean."""
        try:
            metrics = report.get("metrics", [])
            if metrics:
                return metrics[0].get("result", {}).get("drift_detected", False)
        except Exception:
            pass
        return False

    @staticmethod
    def _push_drift_metrics(result: dict) -> None:
        """Push drift results to Prometheus gauges."""
        try:
            from atm_forecast.monitoring.metrics import (
                DRIFT_DETECTED, DRIFT_SCORE, MONITORING_REPORT_COUNT,
            )
            drift_type = result.get("method", "unknown")
            DRIFT_SCORE.labels(drift_type=drift_type, column="overall").set(
                result.get("drift_score", 0)
            )
            if result.get("drift_detected"):
                for feat in result.get("drifted_features", []):
                    DRIFT_DETECTED.labels(drift_type=drift_type, column=feat).inc()
            MONITORING_REPORT_COUNT.labels(report_type="data_drift").inc()
        except Exception as exc:
            logger.debug("Prometheus push failed: %s", exc)


# =====================================================================
# Prediction Quality Monitor
# =====================================================================

class PredictionQualityMonitor:
    """Track prediction quality per model, per location, per target.

    Stores prediction logs as Parquet for offline analysis and generates
    Evidently regression performance reports when enough data accumulates.

    Metrics tracked:
    - MAE, RMSE, R² per (model, location, target)
    - Residual distribution (for detecting systematic bias)
    - Inference latency
    """

    def __init__(
        self,
        storage_dir: str | Path = "artifacts/predictions",
        min_samples_for_report: int = 50,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.min_samples = min_samples_for_report
        self._buffer: List[Dict] = []

    def log_prediction(
        self,
        model_name: str,
        location: str,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        targets: List[str] | None = None,
        inference_time_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> Dict[str, Any]:
        """Log a single prediction (could be multi-target).

        Parameters
        ----------
        model_name : str  — e.g. "bilstm", "tcn", "tft"
        location : str    — e.g. "London", "Accra"
        y_true : array    — actual values (shape: [n_targets] or scalar)
        y_pred : array    — predicted values
        targets : list    — target column names
        inference_time_ms : float — time taken for this prediction
        metadata : dict   — any additional context

        Returns
        -------
        dict — per-target metrics for this prediction
        """
        targets = targets or TARGETS
        y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
        y_pred = np.atleast_1d(np.asarray(y_pred, dtype=float))

        now = datetime.now(timezone.utc)
        per_target_metrics = {}

        for i, target in enumerate(targets[:len(y_true)]):
            true_val = float(y_true[i])
            pred_val = float(y_pred[i])
            error = abs(true_val - pred_val)

            record = {
                "timestamp": now.isoformat(),
                "model": model_name,
                "location": location,
                "target": target,
                "y_true": true_val,
                "y_pred": pred_val,
                "absolute_error": error,
                "squared_error": error ** 2,
                "inference_time_ms": inference_time_ms,
                **(metadata or {}),
            }
            self._buffer.append(record)

            per_target_metrics[target] = {
                "mae": error,
                "se": error ** 2,
            }

        # Flush buffer periodically
        if len(self._buffer) >= 1000:
            self._flush()

        return per_target_metrics

    def get_quality_summary(
        self,
        group_by: List[str] | None = None,
    ) -> pd.DataFrame:
        """Aggregate prediction quality metrics.

        Parameters
        ----------
        group_by : list of str
            Columns to group by, e.g. ["model"], ["model", "location"],
            ["model", "target"], ["model", "location", "target"].

        Returns
        -------
        pd.DataFrame with columns: group_keys + MAE, RMSE, count
        """
        self._flush()
        group_by = group_by or ["model"]

        all_data = self._load_all_logs()
        if all_data.empty:
            return pd.DataFrame()

        agg = all_data.groupby(group_by).agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
            count=("absolute_error", "count"),
            mean_y_true=("y_true", "mean"),
            mean_y_pred=("y_pred", "mean"),
            mean_inference_ms=("inference_time_ms", "mean"),
        ).round(4).reset_index()

        return agg

    def generate_performance_report(
        self,
        model_name: str | None = None,
        location: str | None = None,
    ) -> Dict[str, Any]:
        """Generate an Evidently regression performance report.

        Filters to a specific model/location, splits 50/50 for
        reference vs current, and runs RegressionPreset.
        """
        self._flush()
        df = self._load_all_logs()
        if df.empty:
            return {"status": "no_data"}

        if model_name:
            df = df[df["model"] == model_name]
        if location:
            df = df[df["location"] == location]

        if len(df) < self.min_samples:
            return {"status": "insufficient_data", "n_samples": len(df)}

        # Split chronologically for reference vs current
        split = len(df) // 2
        ref = df.iloc[:split]
        cur = df.iloc[split:]

        try:
            from atm_forecast.monitoring.evidently_monitor import generate_model_performance_report

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            label = f"{model_name or 'all'}_{location or 'all'}"
            output_path = self.storage_dir / f"perf_report_{label}_{ts}.html"

            report = generate_model_performance_report(
                reference=ref,
                current=cur,
                target_column="y_true",
                prediction_column="y_pred",
                output_path=str(output_path),
            )
            return {"status": "ok", "report_path": str(output_path), "report": report}
        except ImportError:
            logger.warning("Evidently not available for performance report")
            return {"status": "no_evidently"}

    def _flush(self) -> None:
        """Write buffered records to Parquet."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = self.storage_dir / f"predictions_{ts}.parquet"
        df.to_parquet(out_path, index=False)
        logger.debug("Flushed %d prediction records to %s", len(self._buffer), out_path)
        self._buffer.clear()

    def _load_all_logs(self) -> pd.DataFrame:
        """Load all prediction log Parquet files."""
        files = sorted(self.storage_dir.glob("predictions_*.parquet"))
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)


# =====================================================================
# MLflow Dataset Logger
# =====================================================================

class MLflowDatasetLogger:
    """Log feature snapshots as MLflow datasets for lineage tracking.

    Each feature engineering run produces a dataset that gets registered
    in MLflow with metadata (row count, columns, drift stats, timestamp).
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri

    def log_feature_dataset(
        self,
        df: pd.DataFrame,
        name: str = "atm_features",
        run_id: str | None = None,
        tags: dict | None = None,
        save_path: str | Path | None = None,
    ) -> dict:
        """Register a feature DataFrame as an MLflow dataset.

        Parameters
        ----------
        df : pd.DataFrame — the feature-engineered data
        name : str — dataset name in the registry
        run_id : str — MLflow run ID to attach to (or creates a new run)
        tags : dict — extra tags (e.g. drift scores, partition dates)
        save_path : Path — where to save the Parquet snapshot

        Returns
        -------
        dict with dataset_name, version, n_rows, n_cols, run_id
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)

            # Save snapshot
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(save_path, index=False)

            # Create MLflow dataset
            dataset = mlflow.data.from_pandas(
                df,
                name=name,
                targets=",".join([c for c in TARGETS if c in df.columns]),
            )

            # Log to MLflow run
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_input(dataset, context="training")
                    if tags:
                        mlflow.set_tags({f"dataset/{k}": str(v) for k, v in tags.items()})
            else:
                with mlflow.start_run(run_name=f"dataset_{name}"):
                    mlflow.log_input(dataset, context="feature_store")
                    mlflow.log_params({
                        "dataset_name": name,
                        "n_rows": len(df),
                        "n_cols": len(df.columns),
                    })
                    if tags:
                        mlflow.set_tags({f"dataset/{k}": str(v) for k, v in tags.items()})
                    if save_path:
                        mlflow.log_artifact(str(save_path), artifact_path="datasets")
                    current_run_id = mlflow.active_run().info.run_id

            logger.info(
                "MLflow dataset logged: %s (%d rows × %d cols)",
                name, len(df), len(df.columns),
            )
            return {
                "dataset_name": name,
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "columns": list(df.columns),
            }

        except ImportError:
            logger.warning("MLflow not available for dataset logging")
            return {"error": "mlflow_not_installed"}
        except Exception as exc:
            logger.warning("MLflow dataset logging failed: %s", exc)
            return {"error": str(exc)}

    def log_reference_dataset(
        self,
        df: pd.DataFrame,
        model_name: str,
        experiment_name: str = "atm-forecast-ct",
    ) -> dict:
        """Log the training reference dataset linked to a specific model."""
        return self.log_feature_dataset(
            df,
            name=f"reference_{model_name}",
            tags={
                "type": "reference",
                "model": model_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            save_path=Path("artifacts") / "reference" / f"ref_{model_name}.parquet",
        )

    def log_production_dataset(
        self,
        df: pd.DataFrame,
        batch_id: str | None = None,
    ) -> dict:
        """Log incoming production data for lineage tracking."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.log_feature_dataset(
            df,
            name=f"production_{ts}",
            tags={
                "type": "production",
                "batch_id": batch_id or ts,
                "n_rows": len(df),
            },
            save_path=Path("artifacts") / "datasets" / f"prod_{ts}.parquet",
        )


# =====================================================================
# CT Orchestration Helper
# =====================================================================

class CTMonitor:
    """Unified continuous-training monitor.

    Combines drift detection, prediction quality, and dataset logging
    into a single interface used by the CT pipeline and Airflow DAG.

    All monitoring data flows to **MLOps tools** (not the end-user frontend):

    - **W&B** — drift score line charts, drift tables, prediction quality
      tables, Evidently HTML reports as artifacts
    - **MLflow** — drift metrics per run, Evidently reports as artifacts,
      feature datasets with lineage tags
    - **Evidently** — standalone HTML reports in ``artifacts/reports/``
      (viewable directly or via Evidently's own UI)
    - **Prometheus** — drift/training gauges for Grafana dashboards
    """

    def __init__(
        self,
        reports_dir: str | Path = "artifacts/reports",
        reference_dir: str | Path = "artifacts/reference",
        predictions_dir: str | Path = "artifacts/predictions",
        mlflow_tracking_uri: str = "http://localhost:5000",
        drift_threshold: float = 0.05,
        wandb_project: str = "atm-forecast",
        wandb_entity: str | None = None,
    ):
        self.reports_dir = Path(reports_dir)
        self.drift_detector = DriftDetector(
            reports_dir=reports_dir,
            drift_threshold=drift_threshold,
        )
        self.reference_store = ReferenceStore(base_dir=reference_dir)
        self.prediction_monitor = PredictionQualityMonitor(
            storage_dir=predictions_dir,
        )
        self.dataset_logger = MLflowDatasetLogger(
            tracking_uri=mlflow_tracking_uri,
        )
        self.drift_threshold = drift_threshold
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

    def check_and_decide(
        self,
        current_data: pd.DataFrame,
        is_first_run: bool = False,
    ) -> Dict[str, Any]:
        """Main entry point: check drift and decide whether to retrain.

        Drift results are automatically logged to **W&B** (as summary
        metrics + a drift table) and **MLflow** (as run metrics + report
        artifacts).  Evidently HTML reports land in ``artifacts/reports/``.

        Parameters
        ----------
        current_data : pd.DataFrame
            Latest feature-engineered data.
        is_first_run : bool
            If True, always retrain (no reference available).

        Returns
        -------
        dict with: should_retrain, data_drift, concept_drift, quality
        """
        if is_first_run:
            logger.info("First run — saving reference and triggering training")
            self.reference_store.save(current_data, {"reason": "first_run"})
            result = {
                "should_retrain": True,
                "reason": "first_run",
                "data_drift": {},
                "concept_drift": {},
            }
            self._log_drift_to_wandb(result)
            self._log_drift_to_mlflow(result)
            return result

        reference = self.reference_store.load()
        if reference is None:
            logger.info("No reference found — saving current as reference")
            self.reference_store.save(current_data, {"reason": "no_reference"})
            result = {
                "should_retrain": True,
                "reason": "no_reference",
                "data_drift": {},
                "concept_drift": {},
            }
            self._log_drift_to_wandb(result)
            self._log_drift_to_mlflow(result)
            return result

        # Log production data to MLflow
        self.dataset_logger.log_production_dataset(current_data)

        # Run full drift analysis (Evidently reports saved to disk)
        result = self.drift_detector.full_drift_check(reference, current_data)

        # Push to MLOps dashboards
        self._log_drift_to_wandb(result)
        self._log_drift_to_mlflow(result)

        return result

    def post_training_hook(
        self,
        training_data: pd.DataFrame,
        model_name: str,
        metrics: dict,
    ) -> None:
        """Called after successful retraining to update reference + log datasets.

        Pushes to W&B, MLflow, and Prometheus.

        Parameters
        ----------
        training_data : pd.DataFrame
            The data the model was trained on.
        model_name : str
            Name of the champion model.
        metrics : dict
            Training metrics (R², MAE, etc.)
        """
        # Update reference snapshot
        self.reference_store.save(training_data, {
            "model": model_name,
            "metrics": metrics,
        })

        # Log training dataset to MLflow
        self.dataset_logger.log_reference_dataset(training_data, model_name)

        # Push training metrics to Prometheus
        try:
            from atm_forecast.monitoring.metrics import (
                TRAINING_METRIC, TRAINING_RUNS,
            )
            TRAINING_RUNS.labels(status="success").inc()
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    TRAINING_METRIC.labels(metric_name=metric_name).set(value)
        except Exception as exc:
            logger.debug("Prometheus training metrics push failed: %s", exc)

        # Log retraining event to W&B
        self._log_retrain_to_wandb(model_name, metrics)

        # Log retraining event to MLflow
        self._log_retrain_to_mlflow(model_name, metrics)

        logger.info("Post-training hook complete for model=%s", model_name)

    def log_prediction_quality_to_wandb(self) -> None:
        """Push aggregated prediction quality tables to W&B.

        Logs tables grouped by (model), (model, location), and
        (model, target) so they appear as interactive W&B Tables.
        """
        try:
            import wandb
            if wandb.run is None:
                return

            for group_by, table_name in [
                (["model"], "pred_quality/by_model"),
                (["model", "location"], "pred_quality/by_model_location"),
                (["model", "target"], "pred_quality/by_model_target"),
            ]:
                df = self.prediction_monitor.get_quality_summary(group_by=group_by)
                if df.empty:
                    continue
                table = wandb.Table(dataframe=df)
                wandb.log({table_name: table})

            logger.info("Prediction quality tables logged to W&B")
        except ImportError:
            logger.debug("wandb not installed — skipping prediction quality push")
        except Exception as exc:
            logger.debug("W&B prediction quality push failed: %s", exc)

    # ── Private: W&B integration ─────────────────────────────────

    def _log_drift_to_wandb(self, result: dict) -> None:
        """Log drift detection results to W&B.

        Creates:
        - Scalar metrics: drift_score, n_drifted_features, should_retrain
        - A W&B Table of per-target concept drift results
        - Evidently HTML reports uploaded as W&B Artifacts
        """
        try:
            import wandb
            if wandb.run is None:
                # Start a dedicated CT monitoring run
                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=f"ct-drift-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
                    job_type="ct-monitoring",
                    tags=["ct-pipeline", "drift-detection"],
                    reinit="finish_previous",
                )

            data_drift = result.get("data_drift", {})
            concept_drift = result.get("concept_drift", {})

            # Scalar metrics (appear as line charts across runs)
            wandb.log({
                "ct/should_retrain": int(result.get("should_retrain", False)),
                "ct/data_drift_score": data_drift.get("drift_score", 0.0),
                "ct/n_drifted_features": data_drift.get("n_drifted_features", 0),
                "ct/n_total_features": data_drift.get("n_total_features", 0),
                "ct/concept_drift_targets": concept_drift.get("n_targets_drifted", 0),
                "ct/drift_check_time_s": result.get("drift_check_time_s", 0),
            })

            # Drifted features as a W&B Table
            drifted = data_drift.get("drifted_features", [])
            if drifted:
                table = wandb.Table(
                    columns=["feature", "drifted"],
                    data=[[f, True] for f in drifted],
                )
                wandb.log({"ct/drifted_features_table": table})

            # Concept drift per-target as a table
            per_target = concept_drift.get("per_target", {})
            if per_target:
                rows = []
                for target, info in per_target.items():
                    rows.append([
                        target,
                        info.get("drift_detected", False),
                        info.get("p_value", None),
                        info.get("ks_statistic", None),
                    ])
                table = wandb.Table(
                    columns=["target", "drift_detected", "p_value", "ks_statistic"],
                    data=rows,
                )
                wandb.log({"ct/concept_drift_table": table})

            # Upload Evidently HTML reports as artifacts
            self._upload_reports_to_wandb()

            wandb.summary["ct/last_drift_check"] = datetime.now(timezone.utc).isoformat()
            logger.info("Drift results logged to W&B")

        except ImportError:
            logger.debug("wandb not installed — skipping drift push")
        except Exception as exc:
            logger.debug("W&B drift push failed: %s", exc)

    def _upload_reports_to_wandb(self) -> None:
        """Upload the latest Evidently HTML reports as a W&B Artifact."""
        try:
            import wandb
            if wandb.run is None:
                return

            html_files = list(self.reports_dir.glob("*.html"))
            if not html_files:
                return

            artifact = wandb.Artifact(
                name="evidently-reports",
                type="monitoring-report",
                description="Evidently data drift, concept drift, and quality reports",
            )
            for f in html_files[-10:]:  # last 10 reports
                artifact.add_file(str(f))
            wandb.log_artifact(artifact)
            logger.info("Uploaded %d Evidently reports to W&B", min(len(html_files), 10))
        except Exception as exc:
            logger.debug("W&B artifact upload failed: %s", exc)

    def _log_retrain_to_wandb(self, model_name: str, metrics: dict) -> None:
        """Log a retraining event to W&B."""
        try:
            import wandb
            if wandb.run is None:
                return

            retrain_metrics = {
                "ct/retrain_model": model_name,
                "ct/retrain_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    retrain_metrics[f"ct/retrain_{k}"] = v
            wandb.log(retrain_metrics)
            wandb.summary["ct/last_retrain"] = datetime.now(timezone.utc).isoformat()
            wandb.summary["ct/last_retrain_model"] = model_name
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("W&B retrain push failed: %s", exc)

    # ── Private: MLflow integration ──────────────────────────────

    def _log_drift_to_mlflow(self, result: dict) -> None:
        """Log drift metrics and Evidently reports to MLflow.

        Creates a dedicated MLflow run under the ``atm-forecast-ct``
        experiment with drift metrics + HTML report artifacts.
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("atm-forecast-ct")

            data_drift = result.get("data_drift", {})
            concept_drift = result.get("concept_drift", {})

            with mlflow.start_run(
                run_name=f"drift-check-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
                tags={
                    "pipeline": "ct",
                    "stage": "drift_detection",
                    "should_retrain": str(result.get("should_retrain", False)),
                },
            ):
                # Log scalar drift metrics
                mlflow.log_metrics({
                    "data_drift_score": data_drift.get("drift_score", 0.0),
                    "n_drifted_features": data_drift.get("n_drifted_features", 0),
                    "n_total_features": data_drift.get("n_total_features", 0),
                    "concept_drift_targets": concept_drift.get("n_targets_drifted", 0),
                    "total_targets": concept_drift.get("total_targets", 0),
                    "drift_check_time_s": result.get("drift_check_time_s", 0),
                })

                mlflow.log_params({
                    "should_retrain": str(result.get("should_retrain", False)),
                    "reason": result.get("reason", "drift_analysis"),
                    "drift_method": data_drift.get("method", "unknown"),
                    "drift_threshold": self.drift_threshold,
                })

                # Log per-target concept drift
                for target, info in concept_drift.get("per_target", {}).items():
                    short = target.replace("air_quality_", "aq_")
                    if "p_value" in info:
                        mlflow.log_metric(f"concept_drift/{short}/p_value", info["p_value"])
                    if "ks_statistic" in info:
                        mlflow.log_metric(f"concept_drift/{short}/ks_stat", info["ks_statistic"])

                # Upload Evidently HTML reports as artifacts
                html_files = list(self.reports_dir.glob("*.html"))
                for f in html_files[-10:]:
                    mlflow.log_artifact(str(f), artifact_path="evidently_reports")

                # Upload JSON drift summary
                json_files = list(self.reports_dir.glob("drift_summary_*.json"))
                for f in json_files[-5:]:
                    mlflow.log_artifact(str(f), artifact_path="drift_summaries")

            logger.info("Drift results logged to MLflow (atm-forecast-ct)")

        except ImportError:
            logger.debug("mlflow not installed — skipping drift push")
        except Exception as exc:
            logger.debug("MLflow drift push failed: %s", exc)

    def _log_retrain_to_mlflow(self, model_name: str, metrics: dict) -> None:
        """Log a retraining completion event to MLflow."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("atm-forecast-ct")

            with mlflow.start_run(
                run_name=f"retrain-{model_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
                tags={
                    "pipeline": "ct",
                    "stage": "retraining",
                    "champion_model": model_name,
                },
            ):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"retrain/{k}", float(v))
                mlflow.log_params({
                    "model_name": model_name,
                    "retrain_timestamp": datetime.now(timezone.utc).isoformat(),
                })
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("MLflow retrain push failed: %s", exc)
