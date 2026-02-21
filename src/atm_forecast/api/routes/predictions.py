"""Prediction / inference endpoints."""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from atm_forecast.api.dependencies import get_model_service
from atm_forecast.api.schemas import ForecastRequest, ForecastResponse
from atm_forecast.monitoring.metrics import (
    INFERENCE_LATENCY,
    PREDICTION_COUNT,
    PREDICTION_VALUE,
    PREPROCESSING_LATENCY,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=ForecastResponse)
def predict(
    request: ForecastRequest,
    model_service=Depends(get_model_service),
) -> ForecastResponse:
    """Generate a temperature forecast from input time-series data."""
    try:
        input_array = np.array(request.features)

        # Validate shape
        if input_array.ndim != 2:
            PREDICTION_COUNT.labels(status="invalid").inc()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="features must be a 2-D array (timesteps x features).",
            )

        # ── Preprocessing (scaling + reshape) ──────────────────────────
        preprocess_start = time.perf_counter()
        if model_service.scaler is not None:
            input_array = model_service.scaler.transform(input_array)
        if input_array.ndim == 2:
            input_array = input_array[np.newaxis, ...]
        PREPROCESSING_LATENCY.observe(time.perf_counter() - preprocess_start)

        # ── Inference ─────────────────────────────────────────────
        inference_start = time.perf_counter()
        raw_preds = model_service.model.predict(input_array, verbose=0)
        INFERENCE_LATENCY.observe(time.perf_counter() - inference_start)

        predictions = raw_preds.flatten()

        # Track output distribution for drift detection
        for val in predictions:
            PREDICTION_VALUE.observe(float(val))

        PREDICTION_COUNT.labels(status="success").inc()

        return ForecastResponse(
            predictions=predictions.tolist(),
            model_version=model_service.model_version,
        )
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_COUNT.labels(status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        ) from exc
