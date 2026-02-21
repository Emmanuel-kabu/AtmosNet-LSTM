"""Centralized settings using Pydantic for validation and .env loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Project root is three levels up from this file:
#   src/atm_forecast/config/settings.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env files."""

    # ── General ──────────────────────────────────────────────────────────
    app_name: str = "atm-forecast"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data"
    lake_root: Path = PROJECT_ROOT / "data" / "lake"
    manifests_dir: Path = PROJECT_ROOT / "data" / "manifests"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    model_dir: Path = PROJECT_ROOT / "models"

    # ── Model hyper-parameters ───────────────────────────────────────────
    sequence_length: int = Field(default=24, ge=1, description="Lookback window size")
    forecast_horizon: int = Field(default=1, ge=1, description="Steps to predict ahead")
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=50, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    lstm_units: int = Field(default=64, ge=1)
    dropout_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0)
    test_split: float = Field(default=0.1, ge=0.0, le=1.0)

    # ── Experiment tracking ──────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "atm-forecast"
    wandb_project: str = "atm-forecast"
    wandb_entity: str = ""  # W&B team/org (leave empty for personal)
    wandb_enabled: bool = False
    wandb_api_key: str = ""  # Set via ATM_WANDB_API_KEY env var

    # ── API / Serving ────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = Field(default=1, ge=1)
    cors_origins: list[str] = ["*"]

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./atm_forecast.db"

    # ── Monitoring ───────────────────────────────────────────────────────
    enable_prometheus: bool = True
    drift_threshold: float = Field(default=0.05, gt=0)
    evidently_reports_dir: Path = PROJECT_ROOT / "artifacts" / "reports"
    monitoring_schedule_minutes: int = Field(default=60, ge=1)
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "ATM_",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @field_validator("data_dir", "lake_root", "manifests_dir", "artifacts_dir", "model_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
