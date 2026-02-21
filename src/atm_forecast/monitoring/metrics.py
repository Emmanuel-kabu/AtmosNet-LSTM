"""Prometheus metrics for the serving and monitoring layers."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, Summary

# ═══════════════════════════════════════════════════════════════════════════
#  Application Info
# ═══════════════════════════════════════════════════════════════════════════
APP_INFO = Info("atm_forecast", "Atmospheric Forecasting Service")

# ═══════════════════════════════════════════════════════════════════════════
#  API / HTTP Metrics
# ═══════════════════════════════════════════════════════════════════════════
HTTP_REQUEST_COUNT = Counter(
    "atm_forecast_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_LATENCY = Histogram(
    "atm_forecast_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "atm_forecast_http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["method", "endpoint"],
)

# ═══════════════════════════════════════════════════════════════════════════
#  Prediction / Inference Metrics
# ═══════════════════════════════════════════════════════════════════════════
PREDICTION_COUNT = Counter(
    "atm_forecast_predictions_total",
    "Total number of prediction requests",
    ["status"],
)

INFERENCE_LATENCY = Histogram(
    "atm_forecast_inference_latency_seconds",
    "Time spent inside model.predict() (excludes pre/post-processing)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREPROCESSING_LATENCY = Histogram(
    "atm_forecast_preprocessing_latency_seconds",
    "Time spent scaling / reshaping input before inference",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05],
)

PREDICTION_VALUE = Summary(
    "atm_forecast_prediction_value",
    "Distribution of predicted values (for monitoring output drift)",
)

# ═══════════════════════════════════════════════════════════════════════════
#  Model Lifecycle Metrics
# ═══════════════════════════════════════════════════════════════════════════
MODEL_LOAD_TIME = Histogram(
    "atm_forecast_model_load_seconds",
    "Time to load the model on startup",
)

MODEL_LOADED = Gauge(
    "atm_forecast_model_loaded",
    "Whether a model is currently loaded (1) or not (0)",
)

# ═══════════════════════════════════════════════════════════════════════════
#  Drift & Monitoring Metrics
# ═══════════════════════════════════════════════════════════════════════════
DRIFT_DETECTED = Counter(
    "atm_forecast_drift_detected_total",
    "Number of drift detections",
    ["drift_type", "column"],
)

DRIFT_SCORE = Gauge(
    "atm_forecast_drift_score",
    "Latest drift statistic per column",
    ["drift_type", "column"],
)

MONITORING_REPORT_COUNT = Counter(
    "atm_forecast_monitoring_reports_total",
    "Number of Evidently monitoring reports generated",
    ["report_type"],
)

# ═══════════════════════════════════════════════════════════════════════════
#  Training Metrics (pushed from training runs)
# ═══════════════════════════════════════════════════════════════════════════
TRAINING_LOSS = Gauge(
    "atm_forecast_training_loss",
    "Latest training loss value",
    ["split"],  # "train" or "val"
)

TRAINING_METRIC = Gauge(
    "atm_forecast_training_metric",
    "Latest training evaluation metric",
    ["metric_name"],
)

TRAINING_RUNS = Counter(
    "atm_forecast_training_runs_total",
    "Total number of training runs completed",
    ["status"],
)


def setup_metrics(version: str = "0.1.0") -> None:
    """Initialise application-level info metrics."""
    APP_INFO.info({"version": version})
