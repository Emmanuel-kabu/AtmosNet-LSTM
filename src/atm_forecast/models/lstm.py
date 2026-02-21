"""LSTM model architecture for time-series forecasting."""

from __future__ import annotations
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def build_lstm_model(
    input_shape: tuple[int, int],
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    forecast_horizon: int = 1,
    num_lstm_layers: int = 2,
) -> keras.Model:
    """Build and compile a stacked-LSTM regression model.

    Parameters
    ----------
    input_shape : tuple[int, int]
        (sequence_length, num_features).
    lstm_units : int
        Number of units per LSTM layer.
    dropout_rate : float
        Dropout probability after each LSTM layer.
    learning_rate : float
        Optimiser learning rate.
    forecast_horizon : int
        Number of output time-steps.
    num_lstm_layers : int
        Number of stacked LSTM layers.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    inputs = keras.Input(shape=input_shape, name="timeseries_input")
    x = inputs

    for i in range(num_lstm_layers):
        return_sequences = i < num_lstm_layers - 1
        x = layers.LSTM(
            units=lstm_units,
            return_sequences=return_sequences,
            name=f"lstm_{i}",
        )(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    x = layers.Dense(32, activation="relu", name="dense_hidden")(x)
    outputs = layers.Dense(forecast_horizon, name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_forecast")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info(
        "Built LSTM model — input_shape=%s, lstm_units=%d, layers=%d, params=%s",
        input_shape,
        lstm_units,
        num_lstm_layers,
        f"{model.count_params():,}",
    )
    return model
