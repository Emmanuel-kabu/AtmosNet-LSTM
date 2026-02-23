"""Monitoring & observability endpoints.

Exposes on-demand Evidently drift reports and monitoring health.
Updated to use the Parquet-based data lake and multi-target system.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from atm_forecast.api.dependencies import (
    DataService,
    get_data_service,
    get_model_service,
)
from atm_forecast.config import get_settings
from atm_forecast.data.preprocessing import TARGETS
from atm_forecast.monitoring.metrics import MONITORING_REPORT_COUNT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.get("/drift", summary="Run data drift report")
def data_drift_report(
    model_service=Depends(get_model_service),
    data_svc: DataService = Depends(get_data_service),
) -> dict[str, Any]:
    """Generate an Evidently data drift report comparing reference vs recent data."""
    try:
        from atm_forecast.monitoring.evidently_monitor import generate_data_drift_report
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="evidently is not installed. Install with: pip install evidently",
        )

    settings = get_settings()

    # Load data from the Parquet-based data lake
    try:
        df = data_svc.raw_dataframe()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not load data from lake: {exc}",
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No numeric columns found in data lake",
        )

    # Split into reference / current (80/20 chronological)
    if "date" in df.columns:
        df = df.sort_values("date")
    split_idx = int(len(df) * 0.8)
    reference = df[numeric_cols].iloc[:split_idx]
    current = df[numeric_cols].iloc[split_idx:]

    result = generate_data_drift_report(
        reference_data=reference,
        current_data=current,
        save_dir=str(settings.evidently_reports_dir),
    )

    MONITORING_REPORT_COUNT.labels(report_type="data_drift").inc()
    return result


@router.get("/performance", summary="Run model performance report")
def model_performance_report(
    model_service=Depends(get_model_service),
    data_svc: DataService = Depends(get_data_service),
) -> dict[str, Any]:
    """Generate an Evidently regression performance report."""
    try:
        from atm_forecast.monitoring.evidently_monitor import (
            generate_model_performance_report,
        )
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="evidently is not installed. Install with: pip install evidently",
        )

    settings = get_settings()

    try:
        df = data_svc.raw_dataframe()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not load data from lake: {exc}",
        )

    # Use first available target column
    target_col = None
    for tgt in TARGETS:
        if tgt in df.columns:
            target_col = tgt
            break
    if target_col is None:
        target_col = df.select_dtypes(include=[np.number]).columns[-1]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "date" in df.columns:
        df = df.sort_values("date")
    split_idx = int(len(df) * 0.8)
    reference = df[numeric_cols].iloc[:split_idx].copy()
    current = df[numeric_cols].iloc[split_idx:].copy()

    # Add dummy predictions column (persistence baseline) for demonstration
    reference["prediction"] = reference[target_col].shift(1).bfill()
    current["prediction"] = current[target_col].shift(1).bfill()

    result = generate_model_performance_report(
        reference_data=reference,
        current_data=current,
        target_column=target_col,
        prediction_column="prediction",
        save_dir=str(settings.evidently_reports_dir),
    )

    MONITORING_REPORT_COUNT.labels(report_type="model_performance").inc()
    return result


@router.get("/status", summary="Monitoring stack health")
def monitoring_status() -> dict[str, Any]:
    """Return current monitoring configuration status."""
    settings = get_settings()

    stack = {
        "prometheus": {"enabled": settings.enable_prometheus},
        "wandb": {
            "enabled": settings.wandb_enabled,
            "project": settings.wandb_project,
        },
        "evidently": {
            "reports_dir": str(settings.evidently_reports_dir),
        },
        "drift_threshold": settings.drift_threshold,
    }

    try:
        import wandb  # noqa: F401

        stack["wandb"]["installed"] = True
    except ImportError:
        stack["wandb"]["installed"] = False

    try:
        import evidently  # noqa: F401

        stack["evidently"]["installed"] = True
    except ImportError:
        stack["evidently"]["installed"] = False

    return {"monitoring": stack}
