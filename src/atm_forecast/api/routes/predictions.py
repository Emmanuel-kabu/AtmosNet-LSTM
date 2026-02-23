"""Prediction / inference endpoints.

Provides:
- ``POST /api/v1/predict``   — legacy flat prediction (backward-compatible).
- ``POST /api/v1/forecast``  — full multi-target forecast aligned with frontend.
- ``GET  /api/v1/models``    — list available trained models.
- ``GET  /api/v1/models/{name}`` — single model metadata.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status

from atm_forecast.api.dependencies import (
    MODEL_NAMES,
    TARGET_DISPLAY,
    TARGET_UNITS,
    get_engineered_data,
    get_model_registry,
    get_model_service,
    get_pipeline_service,
    run_forecast,
    ModelRegistry,
    PipelineService,
)
from atm_forecast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    ModelInfo,
    ModelListResponse,
    MultiForecastRequest,
    MultiForecastResponse,
    TargetForecast,
)
from atm_forecast.data.preprocessing import TARGETS
from atm_forecast.monitoring.metrics import (
    INFERENCE_LATENCY,
    PREDICTION_COUNT,
    PREDICTION_VALUE,
    PREPROCESSING_LATENCY,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


# =================================================================
# POST /api/v1/predict — legacy endpoint (backward-compatible)
# =================================================================

@router.post("/predict", response_model=ForecastResponse)
def predict(
    request: ForecastRequest,
    model_service=Depends(get_model_service),
    pipeline_svc: PipelineService = Depends(get_pipeline_service),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ForecastResponse:
    """Generate a multi-target forecast from a raw feature array.

    Accepts the legacy flat-array format but now returns labelled
    multi-target predictions in original-scale units.
    """
    try:
        input_array = np.array(request.features, dtype=np.float32)

        if input_array.ndim != 2:
            PREDICTION_COUNT.labels(status="invalid").inc()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="features must be a 2-D array (timesteps x features).",
            )

        # ── Select model ──────────────────────────────────────────
        model_name = request.model_name or model_service.model_name
        try:
            model, metadata = registry.get(model_name)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
            )

        # ── Preprocessing ────────────────────────────────────────
        preprocess_start = time.perf_counter()
        if input_array.ndim == 2:
            input_array = input_array[np.newaxis, ...]
        PREPROCESSING_LATENCY.observe(time.perf_counter() - preprocess_start)

        # ── Inference ────────────────────────────────────────────
        inference_start = time.perf_counter()
        raw_preds = model.predict(input_array, verbose=0)
        INFERENCE_LATENCY.observe(time.perf_counter() - inference_start)

        # Inverse-transform to original scale
        preds_original = pipeline_svc.inverse_transform_targets(raw_preds)
        predictions = preds_original.flatten()

        for val in predictions:
            PREDICTION_VALUE.observe(float(val))

        PREDICTION_COUNT.labels(status="success").inc()

        return ForecastResponse(
            predictions=predictions.tolist(),
            target_names=TARGETS,
            model_version=metadata.get("saved_at"),
            model_name=model_name,
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


# =================================================================
# POST /api/v1/forecast — full multi-target forecast
# =================================================================

@router.post("/forecast", response_model=MultiForecastResponse)
def forecast(
    request: MultiForecastRequest,
    pipeline_svc: PipelineService = Depends(get_pipeline_service),
    registry: ModelRegistry = Depends(get_model_registry),
) -> MultiForecastResponse:
    """Run the complete inference pipeline and return structured
    multi-target forecasts with display names and units.

    This endpoint mirrors the data flow used by the Streamlit frontend:
    engineered data → scale → create_sequences → predict → inverse_transform.
    """
    try:
        # ── Resolve model ─────────────────────────────────────────
        try:
            model, metadata = registry.get(request.model_name)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc),
            )

        seq_len = metadata.get("seq_len", 24)

        # ── Engineered data ───────────────────────────────────────
        preprocess_start = time.perf_counter()
        df_eng = get_engineered_data()
        PREPROCESSING_LATENCY.observe(time.perf_counter() - preprocess_start)

        # ── Run forecast ──────────────────────────────────────────
        inference_start = time.perf_counter()
        forecast_df = run_forecast(
            model, pipeline_svc, df_eng, seq_len, request.horizon,
        )
        INFERENCE_LATENCY.observe(time.perf_counter() - inference_start)

        if forecast_df.empty:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Not enough data to produce forecasts. "
                       "Ensure the data lake has sufficient history.",
            )

        # ── Filter targets ────────────────────────────────────────
        selected = request.targets or TARGETS
        invalid = [t for t in selected if t not in TARGETS]
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown targets: {invalid}. Valid: {TARGETS}",
            )

        # ── Build response ────────────────────────────────────────
        target_forecasts = []
        for tgt in selected:
            vals = forecast_df[tgt].tolist()
            target_forecasts.append(
                TargetForecast(
                    target=tgt,
                    display_name=TARGET_DISPLAY.get(tgt, tgt),
                    unit=TARGET_UNITS.get(tgt, ""),
                    values=vals,
                    mean=round(float(np.mean(vals)), 4),
                    min=round(float(np.min(vals)), 4),
                    max=round(float(np.max(vals)), 4),
                )
            )

        # Track predictions for drift monitoring
        for tf_ in target_forecasts:
            for v in tf_.values:
                PREDICTION_VALUE.observe(float(v))
        PREDICTION_COUNT.labels(status="success").inc()

        return MultiForecastResponse(
            model_name=request.model_name,
            model_display_name=MODEL_NAMES.get(request.model_name, request.model_name),
            horizon=request.horizon,
            forecasts=target_forecasts,
            model_version=metadata.get("saved_at"),
            avg_r2=metadata.get("avg_r2"),
        )

    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_COUNT.labels(status="error").inc()
        logger.exception("Forecast failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast error: {exc}",
        ) from exc


# =================================================================
# GET /api/v1/models — list available models
# =================================================================

@router.get("/models", response_model=ModelListResponse)
def list_models(
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelListResponse:
    """Return metadata for all available trained models."""
    all_meta = registry.list_metadata()
    models = []
    for name, meta in all_meta.items():
        models.append(
            ModelInfo(
                name=name,
                display_name=MODEL_NAMES.get(name, name.upper()),
                total_params=meta.get("total_params"),
                avg_r2=meta.get("avg_r2"),
                avg_mae=meta.get("avg_mae"),
                avg_rmse=meta.get("avg_rmse"),
                seq_len=meta.get("seq_len"),
                epochs_trained=meta.get("epochs_trained"),
                saved_at=meta.get("saved_at"),
                per_target=meta.get("per_target"),
            )
        )
    return ModelListResponse(
        models=models,
        best_model=registry.best_model_name(),
    )


# =================================================================
# GET /api/v1/models/{name} — single model metadata
# =================================================================

@router.get("/models/{name}", response_model=ModelInfo)
def get_model_info(
    name: str,
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelInfo:
    """Return metadata for a specific model architecture."""
    all_meta = registry.list_metadata()
    if name not in all_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found. Available: {list(all_meta.keys())}",
        )
    meta = all_meta[name]
    return ModelInfo(
        name=name,
        display_name=MODEL_NAMES.get(name, name.upper()),
        total_params=meta.get("total_params"),
        avg_r2=meta.get("avg_r2"),
        avg_mae=meta.get("avg_mae"),
        avg_rmse=meta.get("avg_rmse"),
        seq_len=meta.get("seq_len"),
        epochs_trained=meta.get("epochs_trained"),
        saved_at=meta.get("saved_at"),
        per_target=meta.get("per_target"),
    )
