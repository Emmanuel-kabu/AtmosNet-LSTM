"""Tests for configuration settings."""

from __future__ import annotations

from atm_forecast.config.settings import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.app_name == "atm-forecast"
        assert s.environment == "development"
        assert s.batch_size == 32
        assert s.epochs == 50

    def test_override_via_kwargs(self):
        s = Settings(epochs=10, batch_size=64)
        assert s.epochs == 10
        assert s.batch_size == 64

    def test_validation_split_bounds(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(validation_split=1.5)
