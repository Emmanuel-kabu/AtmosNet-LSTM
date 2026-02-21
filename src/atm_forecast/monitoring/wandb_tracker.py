"""Weights & Biases experiment tracking integration."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_wandb_run = None


def init_wandb(
    project: str,
    config: dict[str, Any] | None = None,
    run_name: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    group: str | None = None,
) -> Any:
    """Initialise a Weights & Biases run.

    Parameters
    ----------
    project : str
        W&B project name.
    config : dict, optional
        Hyperparameters / settings to log.
    run_name : str, optional
        Human-readable run name.
    tags : list[str], optional
        Tags for filtering runs.
    notes : str, optional
        Free-text description for the run.
    group : str, optional
        Group name (e.g. "experiment-v2").

    Returns
    -------
    wandb.Run
        The active W&B run object.
    """
    global _wandb_run  # noqa: PLW0603

    try:
        import wandb
    except ImportError:
        logger.warning(
            "wandb is not installed — skipping W&B tracking.  "
            "Install with: pip install wandb"
        )
        return None

    _wandb_run = wandb.init(
        project=project,
        config=config,
        name=run_name,
        tags=tags or [],
        notes=notes,
        group=group,
        reinit=True,
    )
    logger.info("W&B run initialised: %s (%s)", _wandb_run.name, _wandb_run.id)
    return _wandb_run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log scalar metrics to the active W&B run.

    Parameters
    ----------
    metrics : dict[str, float]
        Key-value pairs of metric names and values.
    step : int, optional
        Global step / epoch number.
    """
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_model_summary(model) -> None:
    """Log model architecture summary and parameter count to W&B.

    Parameters
    ----------
    model : keras.Model
        The model to log.
    """
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    wandb.log({
        "model/total_params": model.count_params(),
        "model/name": model.name,
    })

    # Watch gradients and weights
    try:
        wandb.watch(model, log="all", log_freq=50)
    except Exception:
        logger.debug("wandb.watch not supported for this model type", exc_info=True)


def log_training_history(history) -> None:
    """Log full Keras training history epoch-by-epoch to W&B.

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned by ``model.fit()``.
    """
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    for epoch in range(len(history.history.get("loss", []))):
        epoch_metrics = {k: v[epoch] for k, v in history.history.items()}
        wandb.log({"epoch": epoch, **epoch_metrics}, step=epoch)


def log_predictions(
    y_true,
    y_pred,
    table_name: str = "predictions",
    max_rows: int = 500,
) -> None:
    """Log prediction vs actual values as a W&B table.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Model predictions.
    table_name : str
        Name of the W&B table.
    max_rows : int
        Max rows to log (to avoid oversized tables).
    """
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    import numpy as np

    y_true = np.asarray(y_true).flatten()[:max_rows]
    y_pred = np.asarray(y_pred).flatten()[:max_rows]

    table = wandb.Table(columns=["index", "actual", "predicted", "error"])
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        table.add_data(i, float(yt), float(yp), float(yt - yp))

    wandb.log({table_name: table})


def log_artifact(
    filepath: str,
    name: str,
    artifact_type: str = "model",
) -> None:
    """Log a file as a W&B artifact.

    Parameters
    ----------
    filepath : str
        Path to the file or directory.
    name : str
        Artifact name.
    artifact_type : str
        Artifact type (e.g. "model", "dataset").
    """
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(filepath)
    wandb.log_artifact(artifact)
    logger.info("Logged W&B artifact: %s (%s)", name, artifact_type)


def finish_wandb() -> None:
    """Finish the active W&B run."""
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        wandb.finish()
        logger.info("W&B run finished")
