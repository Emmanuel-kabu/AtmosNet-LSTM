"""Evidently AI integration for data drift, concept drift, and model performance."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str | Path | None = None,
    column_mapping: Any | None = None,
) -> dict:
    """Generate an Evidently data drift report.

    Parameters
    ----------
    reference : pd.DataFrame
        Training / baseline data.
    current : pd.DataFrame
        Production / incoming data.
    output_path : str | Path, optional
        If provided, save the HTML report to this path.
    column_mapping : ColumnMapping, optional
        Evidently column mapping.

    Returns
    -------
    dict
        Drift report as JSON-serialisable dictionary.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        logger.error(
            "evidently is not installed. Install with: pip install evidently"
        )
        return {"error": "evidently not installed"}

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    result = report.as_dict()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))
        logger.info("Data drift report saved to %s", output_path)

    n_drifted = result.get("metrics", [{}])[0].get("result", {}).get(
        "number_of_drifted_columns", 0
    )
    total = result.get("metrics", [{}])[0].get("result", {}).get(
        "number_of_columns", 0
    )
    logger.info("Evidently data drift: %d/%d columns drifted", n_drifted, total)

    return result


def generate_model_performance_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str = "target",
    prediction_column: str = "prediction",
    output_path: str | Path | None = None,
) -> dict:
    """Generate an Evidently model performance (regression) report.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference data with target & prediction columns.
    current : pd.DataFrame
        Current data with target & prediction columns.
    target_column : str
        Name of the actual target column.
    prediction_column : str
        Name of the prediction column.
    output_path : str | Path, optional
        If provided, save the HTML report to this path.

    Returns
    -------
    dict
        Model performance report as dictionary.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import RegressionPreset
        from evidently import ColumnMapping
    except ImportError:
        logger.error("evidently is not installed.")
        return {"error": "evidently not installed"}

    column_mapping = ColumnMapping(
        target=target_column,
        prediction=prediction_column,
    )

    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))
        logger.info("Model performance report saved to %s", output_path)

    return result


def generate_target_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str = "target",
    output_path: str | Path | None = None,
) -> dict:
    """Generate an Evidently target drift (concept drift) report.

    Concept drift = the relationship between features and target has shifted,
    which shows up as target distribution drift.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference data with target column.
    current : pd.DataFrame
        Current data with target column.
    target_column : str
        Name of the target column.
    output_path : str | Path, optional
        If provided, save the HTML report.

    Returns
    -------
    dict
        Target drift report as dictionary.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import TargetDriftPreset
        from evidently import ColumnMapping
    except ImportError:
        logger.error("evidently is not installed.")
        return {"error": "evidently not installed"}

    column_mapping = ColumnMapping(target=target_column)

    report = Report(metrics=[TargetDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))
        logger.info("Target drift (concept drift) report saved to %s", output_path)

    return result


def run_data_quality_test(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str | Path | None = None,
) -> dict:
    """Run Evidently data quality tests.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference data.
    current : pd.DataFrame
        Current data to validate.
    output_path : str | Path, optional
        Save HTML test suite results.

    Returns
    -------
    dict
        Test suite results.
    """
    try:
        from evidently.test_suite import TestSuite
        from evidently.test_preset import DataStabilityTestPreset, DataQualityTestPreset
    except ImportError:
        logger.error("evidently is not installed.")
        return {"error": "evidently not installed"}

    suite = TestSuite(tests=[
        DataQualityTestPreset(),
        DataStabilityTestPreset(),
    ])
    suite.run(reference_data=reference, current_data=current)

    result = suite.as_dict()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suite.save_html(str(output_path))
        logger.info("Data quality test results saved to %s", output_path)

    # Summarise pass/fail
    tests = result.get("tests", [])
    passed = sum(1 for t in tests if t.get("status") == "SUCCESS")
    failed = sum(1 for t in tests if t.get("status") == "FAIL")
    logger.info("Data quality tests: %d passed, %d failed", passed, failed)

    return result


def generate_full_monitoring_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str = "target",
    prediction_column: str = "prediction",
    output_dir: str | Path = "artifacts/monitoring",
) -> dict[str, dict]:
    """Generate all Evidently reports in one call.

    Returns
    -------
    dict[str, dict]
        Keys: "data_drift", "model_performance", "target_drift", "data_quality".
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results: dict[str, dict] = {}

    results["data_drift"] = generate_data_drift_report(
        reference_data,
        current_data,
        output_path=output_dir / f"data_drift_{timestamp}.html",
    )

    results["target_drift"] = generate_target_drift_report(
        reference_data,
        current_data,
        target_column=target_column,
        output_path=output_dir / f"target_drift_{timestamp}.html",
    )

    if prediction_column in current_data.columns:
        results["model_performance"] = generate_model_performance_report(
            reference_data,
            current_data,
            target_column=target_column,
            prediction_column=prediction_column,
            output_path=output_dir / f"model_performance_{timestamp}.html",
        )

    results["data_quality"] = run_data_quality_test(
        reference_data,
        current_data,
        output_path=output_dir / f"data_quality_{timestamp}.html",
    )

    # Persist summary JSON
    summary_path = output_dir / f"monitoring_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Full monitoring report saved to %s", output_dir)

    return results
