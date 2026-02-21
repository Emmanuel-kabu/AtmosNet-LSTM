"""Integration tests for the FastAPI application."""

from __future__ import annotations

from fastapi.testclient import TestClient

from atm_forecast.api.main import create_app


class TestAPIHealth:
    def setup_method(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_root(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready(self):
        resp = self.client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_predict_without_model_fails(self):
        """Prediction should fail gracefully when no model is loaded."""
        resp = self.client.post(
            "/api/v1/predict",
            json={"features": [[15.0, 60.0, 1013.0]]},
        )
        # 500 or 422 — both acceptable when model isn't trained yet
        assert resp.status_code in (500, 422)
