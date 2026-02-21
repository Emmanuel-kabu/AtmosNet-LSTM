"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_weather_df() -> pd.DataFrame:
    """Generate a small synthetic weather DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "temperature": 20 + 10 * np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 1, n),
            "humidity": rng.uniform(30, 90, n),
            "pressure": 1013 + rng.normal(0, 5, n),
        },
        index=dates,
    )


@pytest.fixture()
def tmp_csv(sample_weather_df: pd.DataFrame, tmp_path) -> str:
    """Write sample data to a temporary CSV and return the path."""
    path = tmp_path / "weather.csv"
    df = sample_weather_df.copy()
    df.index.name = "date"
    df.to_csv(path)
    return str(path)
