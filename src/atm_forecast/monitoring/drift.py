"""Data drift detection utilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def check_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.05,
    columns: list[str] | None = None,
) -> dict[str, dict]:
    """Run Kolmogorov-Smirnov tests to detect distribution drift.

    Parameters
    ----------
    reference : pd.DataFrame
        Baseline (training) data.
    current : pd.DataFrame
        Incoming (production) data.
    threshold : float
        p-value threshold below which drift is flagged.
    columns : list[str] | None
        Columns to check. Defaults to all shared numeric columns.

    Returns
    -------
    dict[str, dict]
        Per-column drift report with statistic, p-value, and drift flag.
    """
    if columns is None:
        shared = set(reference.select_dtypes(include=[np.number]).columns) & set(
            current.select_dtypes(include=[np.number]).columns
        )
        columns = sorted(shared)

    results: dict[str, dict] = {}
    for col in columns:
        stat, p_value = stats.ks_2samp(reference[col].dropna(), current[col].dropna())
        drifted = p_value < threshold
        results[col] = {
            "ks_statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "drift_detected": drifted,
        }
        if drifted:
            logger.warning("Drift detected in column '%s' (p=%.4f)", col, p_value)

    n_drifted = sum(1 for r in results.values() if r["drift_detected"])
    logger.info(
        "Drift check complete: %d/%d columns drifted (threshold=%.3f)",
        n_drifted,
        len(columns),
        threshold,
    )
    return results
