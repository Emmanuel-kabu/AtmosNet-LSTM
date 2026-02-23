"""Tests for feature engineering."""

from __future__ import annotations

import pandas as pd

from atm_forecast.features.feature_engineering import (
    add_cyclical_time_features,
    add_lag_features,
    add_rolling_features,
)


class TestLagFeatures:
    def test_creates_columns(self, sample_weather_df):
        result = add_lag_features(sample_weather_df, columns=["temperature"], lags=[1, 2])
        assert "temperature_lag_1" in result.columns
        assert "temperature_lag_2" in result.columns

    def test_does_not_modify_original(self, sample_weather_df):
        original_cols = list(sample_weather_df.columns)
        add_lag_features(sample_weather_df, columns=["temperature"], lags=[1])
        assert list(sample_weather_df.columns) == original_cols


class TestRollingFeatures:
    def test_creates_mean_and_std(self, sample_weather_df):
        result = add_rolling_features(sample_weather_df, columns=["temperature"], windows=[6])
        assert "temperature_rolling_mean_6" in result.columns
        assert "temperature_rolling_std_6" in result.columns


class TestCyclicalFeatures:
    def test_adds_sin_cos_columns(self, sample_weather_df):
        result = add_cyclical_time_features(sample_weather_df)
        for col in ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
            assert col in result.columns

    def test_values_in_range(self, sample_weather_df):
        result = add_cyclical_time_features(sample_weather_df)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()
