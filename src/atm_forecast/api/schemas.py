"""Pydantic request / response schemas for the API.

Provides both the **legacy** single-array predict schema and the new
**multi-target forecast** schemas that align with the Streamlit frontend.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# =====================================================================
# Legacy schemas (backward-compatible)
# =====================================================================

class ForecastRequest(BaseModel):
    """Legacy input payload for prediction (raw feature array)."""

    features: list[list[float]] = Field(
        ...,
        description=(
            "2-D array of shape (sequence_length, num_features). "
            "Each inner list is one time-step."
        ),
        min_length=1,
    )
    model_name: str | None = Field(
        None,
        description="Model architecture to use: bilstm, tcn, or tft. "
                    "Defaults to the first available model.",
    )

    model_config = {"json_schema_extra": {"examples": [{"features": [[15.2, 60.0, 1013.0]]}]}}


class ForecastResponse(BaseModel):
    """Legacy output payload from prediction."""

    predictions: list[float] = Field(..., description="Forecasted values (one per target).")
    target_names: list[str] | None = Field(
        None, description="Target names corresponding to predictions.",
    )
    model_version: str | None = Field(None, description="Model version / run ID.")
    model_name: str | None = Field(None, description="Model architecture used.")


# =====================================================================
# Multi-target forecast schemas (used by Streamlit frontend)
# =====================================================================

class MultiForecastRequest(BaseModel):
    """Full multi-target forecast request with model/target/horizon selection."""

    model_name: str = Field(
        "bilstm",
        description="Model architecture: bilstm, tcn, or tft.",
    )
    targets: list[str] | None = Field(
        None,
        description="Target columns to include in response. "
                    "None = all 7 targets.",
    )
    horizon: int = Field(
        7, ge=1, le=30,
        description="Number of forecast steps (days) to return.",
    )
    country: str | None = Field(
        None, description="Optional country filter.",
    )
    location: str | None = Field(
        None, description="Optional location_name filter.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_name": "bilstm",
                    "targets": ["temperature_celsius", "air_quality_Ozone"],
                    "horizon": 7,
                }
            ]
        }
    }


class TargetForecast(BaseModel):
    """Forecast result for a single target variable."""

    target: str = Field(..., description="Raw target column name.")
    display_name: str = Field(..., description="Human-readable target name.")
    unit: str = Field(..., description="Measurement unit.")
    values: list[float] = Field(..., description="Forecasted values per step.")
    mean: float = Field(..., description="Mean of forecasted values.")
    min: float = Field(..., description="Min of forecasted values.")
    max: float = Field(..., description="Max of forecasted values.")


class MultiForecastResponse(BaseModel):
    """Structured multi-target forecast response."""

    model_name: str
    model_display_name: str
    horizon: int
    forecasts: list[TargetForecast]
    model_version: str | None = None
    avg_r2: float | None = None


# =====================================================================
# Model & location listing schemas
# =====================================================================

class ModelInfo(BaseModel):
    """Metadata for one trained model."""

    name: str = Field(..., description="Model key (e.g. bilstm).")
    display_name: str
    total_params: int | None = None
    avg_r2: float | None = None
    avg_mae: float | None = None
    avg_rmse: float | None = None
    seq_len: int | None = None
    epochs_trained: int | None = None
    saved_at: str | None = None
    per_target: Dict[str, Dict[str, float]] | None = None


class ModelListResponse(BaseModel):
    """Response for GET /models."""

    models: list[ModelInfo]
    best_model: str | None = Field(
        None, description="Name of the model with highest avg R².",
    )


class LocationInfo(BaseModel):
    """One monitoring location."""

    location_name: str
    country: str
    latitude: float
    longitude: float


class LocationListResponse(BaseModel):
    """Response for GET /locations."""

    total: int
    countries: list[str]
    locations: list[LocationInfo]


class DataSummaryResponse(BaseModel):
    """High-level summary of the raw data lake."""

    total_rows: int
    date_range: list[str] = Field(
        ..., description="[min_date, max_date] as ISO strings.",
    )
    n_countries: int
    n_locations: int
    columns: list[str]
    target_stats: Dict[str, Dict[str, float]] | None = None


# =====================================================================
# Error
# =====================================================================

class ErrorResponse(BaseModel):
    """Standard error body."""

    detail: str
