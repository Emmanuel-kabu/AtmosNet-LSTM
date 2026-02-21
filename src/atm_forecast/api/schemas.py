"""Pydantic request / response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Input payload for prediction."""

    features: list[list[float]] = Field(
        ...,
        description=(
            "2-D array of shape (sequence_length, num_features). "
            "Each inner list is one time-step."
        ),
        min_length=1,
    )

    model_config = {"json_schema_extra": {"examples": [{"features": [[15.2, 60.0, 1013.0]]}]}}


class ForecastResponse(BaseModel):
    """Output payload from prediction."""

    predictions: list[float] = Field(..., description="Forecasted values.")
    model_version: str | None = Field(None, description="Model version / run ID.")


class ErrorResponse(BaseModel):
    """Standard error body."""

    detail: str
