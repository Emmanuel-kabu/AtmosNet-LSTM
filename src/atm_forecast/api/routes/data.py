"""Data & location endpoints.

Provides:
- ``GET /api/v1/locations``     — all monitoring locations with lat/lon.
- ``GET /api/v1/data/summary``  — high-level data-lake statistics.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from atm_forecast.api.dependencies import DataService, get_data_service
from atm_forecast.api.schemas import (
    DataSummaryResponse,
    LocationInfo,
    LocationListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["data"])


# =================================================================
# GET /api/v1/locations
# =================================================================

@router.get("/locations", response_model=LocationListResponse)
def list_locations(
    country: Optional[str] = Query(
        None, description="Filter locations by country name.",
    ),
    data_svc: DataService = Depends(get_data_service),
) -> LocationListResponse:
    """Return all unique monitoring locations with geographic metadata.

    Optionally filter by country.
    """
    try:
        loc_df = data_svc.locations()
        if country:
            loc_df = loc_df[loc_df["country"] == country]

        locations = [
            LocationInfo(
                location_name=row["location_name"],
                country=row["country"],
                latitude=round(float(row["latitude"]), 4),
                longitude=round(float(row["longitude"]), 4),
            )
            for _, row in loc_df.iterrows()
        ]

        countries = sorted(loc_df["country"].unique().tolist())

        return LocationListResponse(
            total=len(locations),
            countries=countries,
            locations=locations,
        )
    except Exception as exc:
        logger.exception("Failed to load locations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load locations: {exc}",
        ) from exc


# =================================================================
# GET /api/v1/data/summary
# =================================================================

@router.get("/data/summary", response_model=DataSummaryResponse)
def data_summary(
    data_svc: DataService = Depends(get_data_service),
) -> DataSummaryResponse:
    """Return high-level statistics about the raw data lake.

    Includes row count, date range, country/location counts,
    column list, and per-target descriptive stats.
    """
    try:
        info = data_svc.summary()
        return DataSummaryResponse(**info)
    except Exception as exc:
        logger.exception("Failed to generate data summary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data summary error: {exc}",
        ) from exc
