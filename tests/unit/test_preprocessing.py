"""Tests for data preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pytest

from atm_forecast.data.preprocessing import create_sequences, prepare_pipeline, split_data


class TestSplitData:
    def test_split_sizes(self, sample_weather_df):
        train, val, test = split_data(sample_weather_df, test_ratio=0.1, val_ratio=0.2)
        assert len(train) + len(val) + len(test) == len(sample_weather_df)

    def test_chronological_order(self, sample_weather_df):
        train, val, test = split_data(sample_weather_df)
        assert train.index[-1] <= val.index[0]
        assert val.index[-1] <= test.index[0]


class TestPipeline:
    def test_minmax_scaler(self):
        pipe = prepare_pipeline("minmax")
        data = np.random.rand(100, 3)
        scaled = pipe.fit_transform(data)
        assert scaled.min() >= -1e-10
        assert scaled.max() <= 1.0 + 1e-10

    def test_standard_scaler(self):
        pipe = prepare_pipeline("standard")
        data = np.random.rand(100, 3)
        scaled = pipe.fit_transform(data)
        assert abs(scaled.mean()) < 0.5  # approximately zero

    def test_invalid_scaler_raises(self):
        with pytest.raises(ValueError, match="Unknown scaler"):
            prepare_pipeline("invalid")


class TestCreateSequences:
    def test_output_shapes(self):
        data = np.random.rand(100, 3)
        X, y = create_sequences(data, sequence_length=10, forecast_horizon=1)
        assert X.shape == (90, 10, 3)
        assert y.shape == (90,)

    def test_horizon_reduces_samples(self):
        data = np.random.rand(100, 3)
        X1, _ = create_sequences(data, sequence_length=10, forecast_horizon=1)
        X5, _ = create_sequences(data, sequence_length=10, forecast_horizon=5)
        assert len(X5) < len(X1)
