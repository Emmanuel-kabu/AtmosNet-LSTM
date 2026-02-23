"""API route definitions."""

from atm_forecast.api.routes.data import router as data_router
from atm_forecast.api.routes.health import router as health_router
from atm_forecast.api.routes.monitoring import router as monitoring_router
from atm_forecast.api.routes.predictions import router as predictions_router

__all__ = ["data_router", "health_router", "monitoring_router", "predictions_router"]
