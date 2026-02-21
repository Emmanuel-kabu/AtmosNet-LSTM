"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from atm_forecast.api.middleware import PrometheusMiddleware
from atm_forecast.api.routes.health import router as health_router
from atm_forecast.api.routes.monitoring import router as monitoring_router
from atm_forecast.api.routes.predictions import router as predictions_router
from atm_forecast.config import get_settings
from atm_forecast.monitoring.metrics import setup_metrics
from atm_forecast.utils.logging import setup_logging


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    setup_logging(level=settings.log_level, json_format=settings.environment == "production")
    setup_metrics(version="0.1.0")

    app = FastAPI(
        title="Atmospheric Forecasting API",
        version="0.1.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )

    # ── Middleware ────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PrometheusMiddleware)

    # ── Routers ──────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(predictions_router)
    app.include_router(monitoring_router)

    # ── Prometheus /metrics endpoint ─────────────────────────────────
    @app.get("/metrics", tags=["monitoring"], include_in_schema=False)
    def prometheus_metrics() -> PlainTextResponse:
        """Expose Prometheus metrics for scraping."""
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "atm_forecast service is running", "environment": settings.environment}

    return app


# Default application instance used by ``uvicorn atm_forecast.api.main:app``
app = create_app()
