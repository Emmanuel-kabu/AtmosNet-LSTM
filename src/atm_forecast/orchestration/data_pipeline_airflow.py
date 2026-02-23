"""
Airflow DAG — AtmosNet Data Pipeline
======================================

Orchestrates the three-stage data lake pipeline that **feeds** the
continuous-training ML pipeline (Flyte):

    ┌──────────┐    ┌──────────┐    ┌───────────────┐
    │  ingest  │──▶ │  clean   │──▶ │  features     │
    │  (raw)   │    │          │    │  engineering   │
    └──────────┘    └──────────┘    └───────────────┘

Each stage is:
- **Watermark-aware**: only processes new data since the last run.
- **Manifest-tracked**: writes a JSON manifest to ``data/manifests/``.
- **Idempotent**: re-runs are safe no-ops if no new data exists.

Schedule: daily at 02:00 UTC (configurable via ``AIRFLOW_DATA_SCHEDULE``).

Data sources supported:
- Kaggle dataset (default: ``nelgiriyewithana/global-weather-repository``)
- Local CSV file
- REST API (template)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ── DAG-level configuration (override via env vars) ──────────────────────

# Extraction source: "kaggle", "csv", or "api"
DATA_SOURCE = os.getenv("ATM_DATA_SOURCE", "kaggle")
KAGGLE_DATASET = os.getenv(
    "ATM_KAGGLE_DATASET", "nelgiriyewithana/global-weather-repository"
)
CSV_FILEPATH = os.getenv("ATM_CSV_FILEPATH", "")
DATE_COLUMN = os.getenv("ATM_DATE_COLUMN", "last_updated")
DATABASE_URL = os.getenv("ATM_DATABASE_URL", "sqlite:///./atm_forecast.db")
SCHEDULE = os.getenv("ATM_DATA_SCHEDULE", "0 2 * * *")  # daily 02:00 UTC

default_args = {
    "owner": "atm-forecast",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# =====================================================================
# Task callables — each wraps an existing pipeline module
# =====================================================================

def _task_ingest(**context):
    """Stage 1: Extract raw data → ``data/lake/raw/`` (watermark-aware)."""
    from atm_forecast.data.data_extraction import ExtractorFactory

    kwargs = {"date_column": DATE_COLUMN, "db_url": DATABASE_URL}

    if DATA_SOURCE == "kaggle":
        kwargs["dataset_slug"] = KAGGLE_DATASET
    elif DATA_SOURCE == "csv":
        if not CSV_FILEPATH:
            raise ValueError("ATM_CSV_FILEPATH must be set for csv source")
        kwargs["filepath"] = CSV_FILEPATH

    extractor = ExtractorFactory.create(DATA_SOURCE, **kwargs)
    written_dates = extractor.extract()

    # Push the partition dates downstream so clean/features know what to do
    date_strs = [d.isoformat() for d in written_dates]
    context["ti"].xcom_push(key="written_dates", value=date_strs)
    logger.info("Ingested %d partitions: %s", len(written_dates), date_strs)
    return date_strs


def _task_clean(**context):
    """Stage 2: Clean raw partitions → ``data/lake/clean/``.

    If ingest produced specific dates, clean only those.
    Otherwise, clean **all** pending raw partitions (catch-up mode).
    """
    from datetime import date as date_type
    from atm_forecast.data.pipeline import transform_clean

    date_strs = context["ti"].xcom_pull(
        task_ids="ingest_raw", key="written_dates"
    )

    dates = None
    if date_strs:
        dates = [date_type.fromisoformat(d) for d in date_strs]

    processed = transform_clean(dates=dates)
    processed_strs = [d.isoformat() for d in processed]
    context["ti"].xcom_push(key="cleaned_dates", value=processed_strs)
    logger.info("Cleaned %d partitions", len(processed))
    return processed_strs


def _task_features(**context):
    """Stage 3: Feature engineering → ``data/lake/features/``.

    Computes lags, rolling stats, cyclical time encodings, interactions
    on the newly cleaned partitions.
    """
    from datetime import date as date_type
    from atm_forecast.data.pipeline import transform_features

    date_strs = context["ti"].xcom_pull(
        task_ids="clean_data", key="cleaned_dates"
    )

    dates = None
    if date_strs:
        dates = [date_type.fromisoformat(d) for d in date_strs]

    processed = transform_features(dates=dates)
    processed_strs = [d.isoformat() for d in processed]
    context["ti"].xcom_push(key="feature_dates", value=processed_strs)
    logger.info("Engineered features for %d partitions", len(processed))
    return processed_strs


def _task_validate(**context):
    """Stage 4 (optional): Quick data-quality check on the features layer.

    Logs warnings but does not fail the DAG — downstream CT will
    decide whether to retrain based on drift detection.
    """
    from atm_forecast.config import get_settings
    from atm_forecast.data.lake import read_all_partitions, LAYER_FEATURES
    import numpy as np

    settings = get_settings()
    lake_root = settings.lake_root

    try:
        df = read_all_partitions(lake_root, LAYER_FEATURES)
    except FileNotFoundError:
        logger.warning("No feature partitions found — skipping validation")
        return {"status": "skipped", "reason": "no_data"}

    stats = {
        "total_rows": len(df),
        "null_pct": round(float(df.isnull().mean().mean()) * 100, 2),
        "n_columns": len(df.columns),
        "inf_count": int(np.isinf(df.select_dtypes(include="number")).sum().sum()),
    }

    if stats["null_pct"] > 5.0:
        logger.warning("High null percentage: %.2f%%", stats["null_pct"])
    if stats["inf_count"] > 0:
        logger.warning("Found %d infinite values", stats["inf_count"])

    context["ti"].xcom_push(key="validation_stats", value=stats)
    logger.info("Validation stats: %s", stats)
    return stats


# =====================================================================
# DAG definition
# =====================================================================

with DAG(
    dag_id="atm_forecast_data_pipeline",
    default_args=default_args,
    description=(
        "Incremental data pipeline: ingest → clean → features. "
        "Feeds the Flyte continuous-training pipeline."
    ),
    schedule=SCHEDULE,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["atm-forecast", "data-pipeline", "etl"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest_raw",
        python_callable=_task_ingest,
        doc_md="Extract new data from source and write to raw Parquet partitions.",
    )

    clean = PythonOperator(
        task_id="clean_data",
        python_callable=_task_clean,
        doc_md="Apply cleaning rules (types, dedup, interpolate NaN) to raw partitions.",
    )

    features = PythonOperator(
        task_id="engineer_features",
        python_callable=_task_features,
        doc_md="Compute lags, rolling stats, cyclical encodings on cleaned data.",
    )

    validate = PythonOperator(
        task_id="validate_features",
        python_callable=_task_validate,
        doc_md="Run data-quality checks on the features layer.",
    )

    # Pipeline: ingest → clean → features → validate
    ingest >> clean >> features >> validate