"""Custom middleware for the FastAPI application."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from atm_forecast.monitoring.metrics import (
    HTTP_REQUEST_COUNT,
    HTTP_REQUEST_LATENCY,
    HTTP_REQUESTS_IN_PROGRESS,
)

logger = logging.getLogger(__name__)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record HTTP request metrics for Prometheus.

    Tracks:
      - ``atm_forecast_http_requests_total`` (counter)
      - ``atm_forecast_http_request_duration_seconds`` (histogram)
      - ``atm_forecast_http_requests_in_progress`` (gauge)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        method = request.method
        path = request.url.path

        # Skip metrics endpoint itself to avoid recursion
        if path == "/metrics":
            return await call_next(request)

        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            HTTP_REQUEST_COUNT.labels(
                method=method, endpoint=path, status_code="500"
            ).inc()
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).dec()
            raise

        elapsed = time.perf_counter() - start

        HTTP_REQUEST_COUNT.labels(
            method=method, endpoint=path, status_code=str(response.status_code)
        ).inc()
        HTTP_REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).dec()

        logger.info(
            "%s %s -> %d (%.1f ms)",
            method,
            path,
            response.status_code,
            elapsed * 1000,
        )
        return response
