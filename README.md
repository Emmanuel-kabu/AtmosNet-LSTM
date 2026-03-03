# AtmosNet — Atmospheric Forecasting Platform

Production-grade time-series forecasting for atmospheric variables (temperature, air quality PM2.5/PM10, Carbon Monoxide, Ozone, NO₂, SO₂) across global locations. Powered by deep-learning architectures (**BiLSTM**, **TCN**, **TFT**), orchestrated end-to-end with **Flyte v2**, and monitored with **Evidently**, **MLflow**, and **Weights & Biases**.

## Highlights

- **Three model architectures** — BiLSTM, Temporal Convolutional Network (TCN), Temporal Fusion Transformer (TFT)
- **7 forecast targets** — temperature + 6 air-quality pollutants
- **Automated feature engineering** — lag, rolling stats, cyclical time encoding, wind/pressure interactions, daylight hours
- **Flyte v2 orchestration** — full ML pipeline and drift-gated continuous training pipeline with TUI
- **Continuous Training** — Evidently data drift + concept drift detection, automatic retraining triggers
- **Experiment tracking** — W&B (training curves, prediction tables, drift tables) + MLflow (model registry, dataset lineage, Evidently report artifacts)
- **Data pipeline** — Airflow DAG (ingest → clean → features → validate → drift check & CT trigger)
- **Serving** — FastAPI with health checks, input validation, Prometheus metrics
- **End-user dashboard** — Streamlit frontend with multi-location forecasts, target comparison, geographic views
- **Docker-ready** — MLflow + Postgres + Airflow + Prometheus + Grafana via docker-compose

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER (Airflow)                            │
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────┐│
│  │  Ingest  │──▶│  Clean   │──▶│ Features  │──▶│ Validate │──▶│Drift ││
│  │  (raw)   │   │          │   │           │   │          │   │Check ││
│  └──────────┘   └──────────┘   └───────────┘   └──────────┘   └──┬───┘│
│     daily 02:00 UTC                                               │    │
└───────────────────────────────────────────────────────────────────┼────┘
                                                                    │
                                          drift detected? ──────────┤
                                                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      ML PIPELINE (Flyte v2)                             │
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────────────┐│
│  │ Load Raw │──▶│Preprocess│──▶│ Engineer  │──▶│ Prepare Training     ││
│  │          │   │          │   │ Features  │   │ Data (split/scale)   ││
│  └──────────┘   └──────────┘   └───────────┘   └──────────┬───────────┘│
│                                                            │           │
│                  ┌───────────────────┬─────────────────────┤           │
│                  ▼                   ▼                     ▼           │
│           ┌────────────┐     ┌────────────┐     ┌────────────┐        │
│           │  BiLSTM    │     │    TCN     │     │    TFT     │        │
│           └─────┬──────┘     └─────┬──────┘     └─────┬──────┘        │
│                 └──────────────────┼──────────────────┘               │
│                                    ▼                                   │
│                           ┌────────────────┐                           │
│                           │   Evaluate &   │                           │
│                           │   Deploy Best  │                           │
│                           └────────────────┘                           │
└──────────────────────────────────────────────────────────────────────────┘
          │                        │                       │
          ▼                        ▼                       ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │   MLflow    │         │    W&B      │         │  Evidently  │
   │  Registry   │         │  Dashboard  │         │   Reports   │
   │:5000        │         │             │         │(HTML + JSON)│
   └─────────────┘         └─────────────┘         └─────────────┘
          │
          ▼
   ┌─────────────┐         ┌─────────────┐
   │  FastAPI    │────────▶│  Streamlit  │
   │  :8000      │         │  :8501      │
   └─────────────┘         └─────────────┘
```

## Data Ingestion Pipeline

The project includes a production-grade, OOP data-extraction system that lands raw weather data into a **Parquet-partitioned data lake**.

### Data Lake Layout

```
data/lake/
├── raw/                     # Immutable landing zone (extraction output)
│   ├── date=2024-05-16/
│   │   └── part-0.parquet
│   └── ...
├── clean/                   # Validated & deduplicated (pipeline stage 2)
└── features/                # Engineered features (pipeline stage 3)
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BaseExtractor` | `data/extraction/data-extraction.py` | Abstract template — watermark check, partitioning, manifest |
| `KaggleExtractor` | same file | Downloads datasets via `kagglehub` |
| `CSVExtractor` | same file | Ingests from local CSV files |
| `APIExtractor` | same file | Pulls from REST APIs (template) |
| `ExtractorFactory` | same file | Registry + factory for source selection |
| Lake I/O | `src/atm_forecast/data/lake.py` | Parquet read/write with `pyarrow` |
| Pipeline State | `src/atm_forecast/data/pipeline_state.py` | Watermark table (tracks last ingestion) |
| Manifests | `src/atm_forecast/data/manifest.py` | JSON run audit trail |

### Running the Ingestion

```bash
# Default: download the Global Weather Repository from Kaggle
python data/extraction/data-extraction.py kaggle

# Specify a different Kaggle dataset
python data/extraction/data-extraction.py kaggle --dataset sumanthvrao/daily-climate-time-series-data

# Ingest from a local CSV
python data/extraction/data-extraction.py csv path/to/file.csv
```

Every run is **watermark-aware** — only rows newer than the last ingestion are fetched, making re-runs safe and idempotent.

## ML Pipeline (Flyte v2)

The full ML pipeline is orchestrated with [Flyte v2](https://docs.flyte.org/) and renders an interactive TUI during local execution.

### Stages

1. **load_raw_data** — read raw Parquet from the data lake
2. **preprocess_data** — clean, fill NaN, clip outliers
3. **engineer_features** — lag, rolling, cyclical, interaction, daylight features
4. **prepare_training_data** — train/val/test split, fit scalers, create sliding-window sequences
5. **train_single_model** × 3 — train BiLSTM, TCN, TFT in parallel
6. **evaluate_models** — compare all models, select champion by R²
7. **deploy_best_model** — promote champion to MLflow registry + serving directory

```bash
# Full training pipeline (local with TUI)
make flyte-ml

# Customise models and epochs
make flyte-ml MODELS='["bilstm","tcn"]' EPOCHS=100

# On a Flyte cluster
make flyte-ml-remote
```

## Continuous Training Pipeline

The CT pipeline adds **drift gates** before training so retraining only happens when the data distribution has actually shifted.

### How It Works

1. **check_data_readiness** — verifies enough new data exists since last training (watermark-aware, reads from features layer)
2. **detect_drift** — runs Evidently data drift + concept drift detection against the reference (training) dataset snapshot
3. If drift detected → full ML pipeline executes → watermark updated → new reference snapshot saved
4. If no drift → pipeline exits early (no resources wasted)

### Drift Detection

| Type | Method | Tool |
|------|--------|------|
| Data Drift | Feature distribution shift | Evidently `DataDriftPreset` (fallback: KS-test) |
| Concept Drift | Target distribution shift | Evidently `TargetDriftPreset` per target |
| Data Quality | NaN %, inf count, stability | Evidently `DataQualityTestPreset` + `DataStabilityTestPreset` |

### Reference Dataset Management

After each successful training, the CT pipeline saves a **reference snapshot** (`artifacts/reference/latest.parquet`) of the feature-engineered data. Future drift checks compare incoming data against this snapshot instead of using an arbitrary 80/20 split.

```bash
# Run CT pipeline (drift-gated)
make flyte-ct

# On a Flyte cluster
make flyte-ct-remote
```

## Monitoring & Experiment Tracking

All CT monitoring data flows to **MLOps tools** rather than the end-user frontend:

### Weights & Biases

- Training curves (loss, MAE, R² per epoch)
- Prediction tables (actual vs predicted)
- **Drift metrics** — `ct/data_drift_score`, `ct/n_drifted_features`, `ct/should_retrain` as line charts across runs
- **Drift tables** — per-feature drift status, per-target concept drift (p-value, KS statistic)
- **Evidently HTML reports** uploaded as W&B Artifacts
- Retraining events with champion model and final metrics

### MLflow

- Model registry with aliases (not stages)
- Per-target metrics (MAE, RMSE, R² grouped by `model/target`)
- **Dataset lineage** — feature snapshots registered via `mlflow.log_input()` with drift metadata tags
- **Drift runs** — dedicated `atm-forecast-ct` experiment with drift scores, per-target concept drift metrics
- **Evidently reports** uploaded as run artifacts under `evidently_reports/`
- Retraining completion runs with `retrain/` prefixed metrics

### Evidently

- Standalone HTML reports saved to `artifacts/reports/` (data drift, concept drift, data quality, regression performance)
- JSON drift summaries for programmatic consumption
- Reports viewable directly or via Evidently's own UI

### Prometheus + Grafana

- API metrics: request count, latency histograms, in-progress gauge
- Inference metrics: prediction count, model load time, preprocessing latency
- Drift gauges: `atm_forecast_drift_score`, `atm_forecast_drift_detected_total`
- Training gauges: `atm_forecast_training_metric`, `atm_forecast_training_runs_total`

## Airflow Data Pipeline

The Airflow DAG orchestrates the data layer and **automatically triggers** the Flyte CT pipeline when drift is detected.

### DAG: `atm_forecast_data_pipeline`

| Task | Purpose |
|------|---------|
| `ingest_raw` | Extract new data from Kaggle/CSV/API → `data/lake/raw/` |
| `clean_data` | Validate, dedup, interpolate NaN → `data/lake/clean/` |
| `engineer_features` | Compute lags, rolling stats, cyclical encodings → `data/lake/features/` |
| `validate_features` | Data quality checks (null %, inf count) |
| `drift_check_and_trigger_ct` | Evidently drift check → trigger Flyte CT pipeline if needed |

**Schedule:** daily at 02:00 UTC

```bash
# Start Airflow (Docker)
make airflow-init    # first time only
make airflow-start   # UI at http://localhost:8080 (admin/admin)

# Manually trigger
make airflow-trigger

# Stop
make airflow-stop
```

## Serving

### FastAPI (`:8000`)

```bash
# Development
make serve

# Production
uvicorn atm_forecast.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe |
| POST | `/api/v1/predict` | Generate forecast |
| GET | `/docs` | Swagger UI |

### Streamlit Dashboard (`:8501`)

End-user-facing forecast dashboard with 5 pages:

| Page | Description |
|------|-------------|
| **Forecast** | Single or multi-location forecast with line charts |
| **Compare Targets** | Side-by-side target comparison across locations |
| **Relationship Analysis** | Feature correlation and scatter plots |
| **Geographic View** | Map-based forecast visualisation with per-target views and radar charts |
| **Model Explorer** | Model architecture details, training history, per-target R² |

```bash
streamlit run src/atm_forecast/frontend_deployment/streamlit_frontend.py
```

## Project Structure

```
├── docker/                  # Dockerfile, docker-compose, Prometheus, Grafana
├── data/
│   ├── extraction/          # OOP data-extraction CLI (Kaggle, CSV, API)
│   ├── lake/                # Parquet-partitioned data lake
│   │   ├── raw/             #   Immutable landing zone
│   │   ├── clean/           #   Validated & deduplicated
│   │   └── features/        #   Engineered features
│   └── manifests/           # JSON audit trail per extraction run
├── src/atm_forecast/        # Main Python package
│   ├── api/                 #   FastAPI app, routes, schemas, middleware
│   ├── config/              #   Pydantic settings (env var support)
│   ├── data/                #   Lake I/O, pipeline state, manifests
│   ├── features/            #   Feature engineering (8 transforms)
│   ├── frontend_deployment/ #   Streamlit dashboard
│   ├── models/              #   BiLSTM, TCN, TFT architectures
│   ├── monitoring/          #   Drift detection, CT monitor, W&B, MLflow,
│   │   │                    #   Evidently, Prometheus metrics
│   │   ├── ct_monitor.py    #   Unified CT monitor (drift + quality + MLflow/W&B push)
│   │   ├── drift.py         #   KS-test drift detection
│   │   ├── evidently_monitor.py  # Evidently report generators
│   │   ├── metrics.py       #   Prometheus metric definitions
│   │   ├── mlflow_tracker.py#   MLflow experiment tracking wrapper
│   │   └── wandb_tracker.py #   W&B experiment tracking wrapper
│   ├── orchestration/       #   Flyte ML/CT pipelines, Airflow DAG
│   ├── training/            #   Training pipeline & evaluation
│   └── utils/               #   Logging & shared utilities
├── tests/                   # Unit & integration tests
├── artifacts/               # Training artefacts, reports, reference datasets
│   ├── models/              #   Saved model weights (bilstm, tcn, tft)
│   ├── preprocessing/       #   Fitted scalers & metadata
│   ├── reference/           #   Training data snapshots for drift comparison
│   ├── reports/             #   Evidently HTML + JSON reports
│   ├── predictions/         #   Prediction quality logs (Parquet)
│   └── datasets/            #   MLflow dataset snapshots
├── notebooks/               # Jupyter exploration
├── pyproject.toml           # Build config (ruff, pytest, mypy)
├── Makefile                 # All commands below
└── requirements.txt         # Pinned runtime dependencies
```

## Quickstart

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for MLflow, Airflow, Prometheus, Grafana)

### Install

```bash
# Runtime
pip install -r requirements.txt
pip install -e .

# With dev tools
pip install -r requirements-dev.txt
pip install -e .
```

### Configuration

```bash
cp .env.example .env
# Edit .env — keys: ATM_WANDB_API_KEY, ATM_MLFLOW_TRACKING_URI, ATM_DATABASE_URL, etc.
```

### Full Workflow

```bash
# 1. Start infrastructure
make docker-up                    # MLflow (:5000) + Postgres

# 2. Ingest data
python data/extraction/data-extraction.py kaggle

# 3. Train all models
make flyte-ml                     # BiLSTM + TCN + TFT, tracked in W&B + MLflow

# 4. Serve
make serve                        # FastAPI at :8000
streamlit run src/atm_forecast/frontend_deployment/streamlit_frontend.py  # :8501

# 5. Start continuous training (optional)
make airflow-init ; make airflow-start   # Airflow at :8080
make flyte-ct                            # Or let Airflow trigger automatically
```

## Make Targets

| Command | Description |
|---------|-------------|
| `make install` | Install runtime dependencies |
| `make install-dev` | Install all dependencies (including dev tools) |
| `make lint` | Run ruff linter |
| `make format` | Auto-format code |
| `make test` | Run test suite |
| `make test-cov` | Run tests with coverage |
| `make serve` | Start FastAPI dev server |
| `make docker-up` | Start MLflow + Postgres via docker-compose |
| `make docker-down` | Stop docker-compose services |
| `make flyte-ml` | Run full ML training pipeline (local with TUI) |
| `make flyte-ml-remote` | Run ML pipeline on Flyte cluster |
| `make flyte-ct` | Run continuous training pipeline (drift-gated) |
| `make flyte-ct-remote` | Run CT pipeline on Flyte cluster |
| `make airflow-init` | Initialise Airflow DB + admin user |
| `make airflow-start` | Start Airflow webserver + scheduler |
| `make airflow-stop` | Stop Airflow services |
| `make airflow-trigger` | Manually trigger the data pipeline DAG |
| `make clean` | Remove build artefacts |

## License

MIT
