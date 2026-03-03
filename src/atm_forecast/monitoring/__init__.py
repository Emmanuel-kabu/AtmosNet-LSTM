"""Model and data monitoring — W&B, Evidently, Prometheus, CT Monitor."""

from atm_forecast.monitoring.drift import check_data_drift
from atm_forecast.monitoring.evidently_monitor import (
    generate_data_drift_report,
    generate_full_monitoring_report,
    generate_model_performance_report,
    generate_target_drift_report,
    run_data_quality_test,
)
from atm_forecast.monitoring.ct_monitor import (
    CTMonitor,
    DriftDetector,
    MLflowDatasetLogger,
    PredictionQualityMonitor,
    ReferenceStore,
)
from atm_forecast.monitoring.metrics import setup_metrics
from atm_forecast.monitoring.wandb_tracker import (
    finish_wandb,
    init_wandb,
    log_artifact,
    log_metrics,
    log_model_summary,
    log_predictions,
    log_training_history,
)

__all__ = [
    # CT Monitor (orchestration-level)
    "CTMonitor",
    "DriftDetector",
    "MLflowDatasetLogger",
    "PredictionQualityMonitor",
    "ReferenceStore",
    # KS-test drift (lightweight)
    "check_data_drift",
    # Evidently reports
    "generate_data_drift_report",
    "generate_full_monitoring_report",
    "generate_model_performance_report",
    "generate_target_drift_report",
    "run_data_quality_test",
    # Prometheus
    "setup_metrics",
    # Weights & Biases
    "finish_wandb",
    "init_wandb",
    "log_artifact",
    "log_metrics",
    "log_model_summary",
    "log_predictions",
    "log_training_history",
]
