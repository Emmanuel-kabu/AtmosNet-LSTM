# Atmospheric Forecasting

Production-ready time-series forecasting for atmospheric variables (temperature, air quality, etc.) using LSTM models with TensorFlow.

## Features

- **LSTM-based forecasting** with configurable architecture
- **Automated feature engineering** — lag, rolling stats, cyclical time encoding
- **FastAPI serving layer** with health checks, input validation, and Prometheus metrics
- **Data drift detection** via Kolmogorov-Smirnov tests
- **Docker-ready** with multi-stage builds and docker-compose orchestration
- **CI/CD pipelines** for linting, testing, and container deployment
- **Pydantic-based configuration** with environment variable support

## Data Ingestion Pipeline

The project includes a production-grade, OOP data-extraction system that lands raw weather data into a **Parquet-partitioned data lake**.

### Architecture

```
┌────────────────────┐
│  External Sources  │   Kaggle · CSV files · REST APIs
└────────┬───────────┘
         │  _fetch_dataframe()
         ▼
┌────────────────────┐
│   BaseExtractor    │   Watermark check → filter → validate
│   (Template)       │   date parsing → partition by day
└────────┬───────────┘
         │  write_partition()
         ▼
┌────────────────────┐
│  data/lake/raw/    │   Snappy-compressed Parquet, partitioned by date
│  date=YYYY-MM-DD/  │   e.g. data/lake/raw/date=2025-06-01/part-0.parquet
│    part-0.parquet   │
└────────┬───────────┘
         │
    ┌────┴────┐
    ▼         ▼
Watermark   Manifest
(SQLite/    (JSON audit
 Postgres)   trail)
```

**Key components:**

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

# Override the date column or database URL
python data/extraction/data-extraction.py kaggle --date-col last_updated --db-url postgresql://user:pass@host/db
```

### Incremental Extraction

Every run is **watermark-aware**. The pipeline tracks the latest ingested timestamp in a `pipeline_state` table (SQLite for dev, Postgres for production). On subsequent runs, only rows newer than the watermark are fetched and written — making re-runs safe and idempotent.

### Data Lake Layout

```
data/lake/
├── raw/                     # Immutable landing zone (extraction output)
│   ├── date=2024-05-16/
│   │   └── part-0.parquet
│   ├── date=2024-05-17/
│   │   └── part-0.parquet
│   └── ...
├── clean/                   # Validated & deduplicated (pipeline stage 2)
└── features/                # Engineered features (pipeline stage 3)
```

### Manifests

Each extraction run produces a JSON manifest in `data/manifests/` recording the pipeline name, run ID, timestamps, and per-partition row counts — providing a full audit trail.

## Project Structure

```
├── .github/workflows/       # CI/CD pipelines
├── docker/                  # Dockerfile, docker-compose, Prometheus config
├── data/
│   ├── extraction/          # OOP data-extraction CLI (Kaggle, CSV, API)
│   ├── lake/                # Parquet-partitioned data lake (gitignored)
│   │   ├── raw/             #   Immutable landing zone
│   │   ├── clean/           #   Validated & deduplicated
│   │   └── features/        #   Engineered features ready for training
│   └── manifests/           # JSON audit trail per extraction run
├── src/atm_forecast/        # Main Python package
│   ├── api/                 #   FastAPI app, routes, schemas, middleware
│   ├── config/              #   Pydantic settings
│   ├── data/                #   Lake I/O, pipeline state, manifests
│   ├── features/            #   Time-series feature engineering
│   ├── models/              #   LSTM architecture & model registry
│   ├── monitoring/          #   Drift detection & Prometheus metrics
│   ├── training/            #   Training pipeline & evaluation
│   └── utils/               #   Logging & shared utilities
├── tests/                   # Unit & integration tests
│   ├── unit/
│   └── integration/
├── models/                  # Saved model artefacts (gitignored)
├── artifacts/               # Training artefacts (gitignored)
├── notebooks/               # Jupyter exploration
├── pyproject.toml           # Build config, tool settings (ruff, pytest, mypy)
├── Makefile                 # Common commands
└── requirements.txt         # Pinned runtime dependencies
```

## Quickstart

### Prerequisites

- Python 3.10+
- (Optional) Docker & Docker Compose

### Install

```bash
# Runtime only
pip install -r requirements.txt
pip install -e .

# With dev tools (linter, tests, type checker)
pip install -r requirements-dev.txt
pip install -e .
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your settings (all prefixed with ATM_)
```

### Train a Model

```bash
python -m atm_forecast.training --data data/weather.csv --target temperature

# Or with overrides
python -m atm_forecast.training \
    --data data/weather.csv \
    --target temperature \
    --epochs 100 \
    --batch-size 64
```

### Serve the API

```bash
# Development (hot-reload)
uvicorn atm_forecast.api.main:app --reload

# Production
uvicorn atm_forecast.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

API endpoints:
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe |
| POST | `/api/v1/predict` | Generate forecast |
| GET | `/docs` | Swagger UI (dev only) |

### Docker

```bash
# Build and run
docker build -f docker/Dockerfile -t atm-forecast .
docker run -p 8000:8000 atm-forecast

# Full stack (API + MLflow + Prometheus + Grafana)
docker compose -f docker/docker-compose.yml --profile tracking --profile monitoring up -d
```

## Development

```bash
# Run tests
make test

# Lint & format
make lint
make format

# Type check
make typecheck

# Tests with coverage
make test-cov
```

## Make Targets

| Command | Description |
|---------|-------------|
| `make install` | Install runtime dependencies |
| `make install-dev` | Install all dependencies |
| `make lint` | Run ruff linter |
| `make format` | Auto-format code |
| `make test` | Run test suite |
| `make test-cov` | Run tests with coverage |
| `make serve` | Start dev API server |
| `make train DATA=path.csv` | Run training pipeline |
| `make docker-build` | Build Docker image |
| `make docker-up` | Start docker-compose services |
| `make clean` | Remove build artefacts |

## License

MIT
