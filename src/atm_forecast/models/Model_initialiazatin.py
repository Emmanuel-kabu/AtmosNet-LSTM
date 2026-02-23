"""
Model Architectures for Multi-Target Atmospheric Forecasting
=============================================================

Three model architectures:

1. **Bi-LSTM** (baseline) — bidirectional stacked LSTM
2. **TCN** — Temporal Convolutional Network with causal dilated convolutions
3. **TFT** — simplified Temporal Fusion Transformer (GRN + multi-head attention)

Each ``build_*`` function returns a *compiled* ``keras.Model`` ready for
``.fit()``.  All models accept input shape ``(seq_len, n_features)`` and
output ``(n_targets,)``.
"""

from __future__ import annotations

import logging
from typing import Literal

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Bi-LSTM (baseline)
# =====================================================================

def build_bilstm(
    seq_len: int,
    n_features: int,
    n_targets: int,
    lstm_units: tuple[int, int] = (128, 64),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Two-layer Bidirectional LSTM → Dense → output.

    Parameters
    ----------
    seq_len : int
        Look-back window length.
    n_features : int
        Number of input features per time-step.
    n_targets : int
        Number of output targets.
    lstm_units : tuple[int, int]
        Units for the first and second LSTM layers.
    dropout_rate : float
        Dropout probability applied after each LSTM and dense layer.
    learning_rate : float
        Adam optimiser learning rate.
    """
    inputs = layers.Input(shape=(seq_len, n_features), name="bilstm_input")

    x = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True, name="lstm_1"),
        name="bilstm_1",
    )(inputs)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units[1], return_sequences=False, name="lstm_2"),
        name="bilstm_2",
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_targets, name="output")(x)

    model = keras.Model(inputs, outputs, name="BiLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info(
        "Built Bi-LSTM — params: %s, input=(%d,%d), targets=%d",
        f"{model.count_params():,}", seq_len, n_features, n_targets,
    )
    return model


# =====================================================================
# 2. TCN — Temporal Convolutional Network
# =====================================================================

class CausalConv1DBlock(layers.Layer):
    """Residual block with two causal dilated Conv1D layers."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv1D(
            filters, kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu",
        )
        self.conv2 = layers.Conv1D(
            filters, kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation="relu",
        )
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.downsample = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(self.filters, 1, padding="same")
        super().build(input_shape)

    def call(self, x, training=None):
        residual = x
        out = self.dropout1(self.conv1(x), training=training)
        out = self.dropout2(self.conv2(out), training=training)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return layers.add([out, residual])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.conv1.kernel_size[0],
            "dilation_rate": self.conv1.dilation_rate[0],
            "dropout_rate": self.dropout1.rate,
        })
        return cfg


def build_tcn(
    seq_len: int,
    n_features: int,
    n_targets: int,
    num_filters: int = 64,
    kernel_size: int = 3,
    num_blocks: int = 4,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Temporal Convolutional Network with exponentially increasing dilation.

    Parameters
    ----------
    num_blocks : int
        Number of residual blocks (dilations: 1, 2, 4, 8, …).
    """
    inputs = layers.Input(shape=(seq_len, n_features), name="tcn_input")
    x = inputs

    for i in range(num_blocks):
        x = CausalConv1DBlock(
            num_filters, kernel_size, dilation_rate=2 ** i, dropout_rate=dropout,
            name=f"tcn_block_{i}",
        )(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_targets, name="output")(x)

    model = keras.Model(inputs, outputs, name="TCN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info(
        "Built TCN — params: %s, blocks=%d, filters=%d",
        f"{model.count_params():,}", num_blocks, num_filters,
    )
    return model


# =====================================================================
# 3. TFT — simplified Temporal Fusion Transformer
# =====================================================================

class GatedLinearUnit(layers.Layer):
    """GLU: element-wise gating via sigmoid."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units)
        self.gate = layers.Dense(units, activation="sigmoid")

    def call(self, x):
        return self.dense(x) * self.gate(x)

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.dense.units
        return cfg


class GatedResidualNetwork(layers.Layer):
    """GRN — core building block of TFT."""

    def __init__(self, units: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense1 = layers.Dense(units, activation="elu")
        self.dense2 = layers.Dense(units)
        self.glu = GatedLinearUnit(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization()
        self.project = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.project = layers.Dense(self.units)
        super().build(input_shape)

    def call(self, x, training=None):
        residual = self.project(x) if self.project is not None else x
        h = self.dense2(self.dense1(x))
        h = self.dropout(h, training=training)
        h = self.glu(h)
        return self.layernorm(h + residual)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "dropout_rate": self.dropout.rate})
        return cfg


def build_tft(
    seq_len: int,
    n_features: int,
    n_targets: int,
    d_model: int = 64,
    n_heads: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Simplified Temporal Fusion Transformer.

    Architecture: Dense projection → GRN → Bi-LSTM encoder →
    GLU gate → Multi-Head Self-Attention → GRN → Dense output.
    """
    inputs = layers.Input(shape=(seq_len, n_features), name="tft_input")

    # Feature projection + GRN
    x = layers.Dense(d_model)(inputs)
    x = layers.Dropout(dropout)(x)
    x = GatedResidualNetwork(d_model, dropout, name="grn_1")(x)

    # LSTM encoder
    lstm_out = layers.Bidirectional(
        layers.LSTM(d_model // 2, return_sequences=True)
    )(x)
    lstm_out = layers.Dropout(dropout)(lstm_out)

    # GLU gate + skip connection
    x = GatedLinearUnit(d_model, name="glu_gate")(lstm_out)
    x = layers.LayerNormalization()(x + layers.Dense(d_model)(inputs))

    # Multi-head self-attention
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,
        dropout=dropout,
        name="mha",
    )(x, x)
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.LayerNormalization()(x + attn_out)

    # Post-attention GRN
    x = GatedResidualNetwork(d_model, dropout, name="grn_2")(x)

    # Take last time-step → output
    x = x[:, -1, :]
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_targets, name="output")(x)

    model = keras.Model(inputs, outputs, name="TFT")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info(
        "Built TFT — params: %s, d_model=%d, heads=%d",
        f"{model.count_params():,}", d_model, n_heads,
    )
    return model


# =====================================================================
# Factory helper
# =====================================================================

MODEL_REGISTRY = {
    "bilstm": build_bilstm,
    "tcn": build_tcn,
    "tft": build_tft,
}


def build_model(
    name: Literal["bilstm", "tcn", "tft"],
    seq_len: int,
    n_features: int,
    n_targets: int,
    **kwargs,
) -> keras.Model:
    """Build a model by name.

    Parameters
    ----------
    name : str
        One of ``'bilstm'``, ``'tcn'``, ``'tft'``.
    seq_len, n_features, n_targets : int
        Standard input/output dimensions.
    **kwargs
        Extra keyword args forwarded to the builder function.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](seq_len, n_features, n_targets, **kwargs)


# Legacy alias
build_lstm_model = build_bilstm
