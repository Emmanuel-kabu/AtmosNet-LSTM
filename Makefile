.PHONY: help install install-dev lint format test test-cov serve train docker-build docker-up docker-down clean \
       flyte-start flyte-stop flyte-ml flyte-ml-remote flyte-ct flyte-ct-remote flyte-register flyte-build \
       airflow-init airflow-start airflow-stop airflow-trigger

# ── Variables ────────────────────────────────────────────────────────────
PYTHON    ?= python
PIP       ?= pip
PYTEST    ?= pytest
RUFF      ?= ruff
UVICORN   ?= uvicorn

# Flyte
FLYTE_IMAGE  ?= ghcr.io/emmanuelkabu/atm-forecast-flyte:latest
ML_ORCH      := src/atm_forecast/orchestration/ml_pipeline_orchestrator.py
DATA_ORCH    := src/atm_forecast/orchestration/data_pipeline_airflow.py
AIRFLOW_HOME ?= $(CURDIR)/airflow_home
MODELS       ?= '["bilstm","tcn","tft"]'
EPOCHS       ?= 50
EPOCHS_CT    ?= 30

# ── Default ──────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ─────────────────────────────────────────────────────────
install: ## Install runtime dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install all dependencies (including dev)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

# ── Code Quality ─────────────────────────────────────────────────────────
lint: ## Run linter
	$(RUFF) check src/ tests/

format: ## Auto-format code
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

typecheck: ## Run type checker
	mypy src/atm_forecast/ --ignore-missing-imports

# ── Testing ──────────────────────────────────────────────────────────────
test: ## Run tests
	$(PYTEST) tests/ -v --tb=short

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ -v --tb=short --cov=atm_forecast --cov-report=term-missing --cov-report=html

# ── Development ──────────────────────────────────────────────────────────
serve: ## Start API server (development)
	$(UVICORN) atm_forecast.api.main:app --reload --host 0.0.0.0 --port 8000

train: ## Run training pipeline (pass DATA=path/to/data.csv)
	$(PYTHON) -m atm_forecast.training --data $(DATA) --target temperature

# ── Docker ───────────────────────────────────────────────────────────────
docker-build: ## Build production Docker image
	docker build -f docker/Dockerfile -t atm-forecast:latest .

docker-up: ## Start all services via docker-compose
	docker compose -f docker/docker-compose-ml.yml up -d

docker-down: ## Stop all services
	docker compose -f docker/docker-compose-ml.yml down

docker-dev: ## Start dev environment with hot-reload
	docker compose -f docker/docker-compose-ml.yml --profile dev up -d api-dev

# ── Cleanup ──────────────────────────────────────────────────────────────
clean: ## Remove build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

# ── Flyte ────────────────────────────────────────────────────────────────
flyte-start: ## Start Flyte sandbox (local cluster)
	DOCKER_API_VERSION=1.44 flytectl demo start --image cr.flyte.org/flyteorg/flyte-sandbox-bundled:latest

flyte-stop: ## Stop Flyte sandbox
	DOCKER_API_VERSION=1.44 flytectl demo teardown

flyte-build: ## Build the Flyte task Docker image
	docker build -f docker/Dockerfile.flyte -t $(FLYTE_IMAGE) .

flyte-ml: ## Run ML pipeline locally (pass MODELS and EPOCHS)
	$(PYTHON) $(ML_ORCH) ml_pipeline --models $(MODELS) --epochs $(EPOCHS) --use-wandb

flyte-ml-remote: ## Run ML pipeline on Flyte cluster
	$(PYTHON) $(ML_ORCH) ml_pipeline --remote --models $(MODELS) --epochs $(EPOCHS) --use-wandb

flyte-ct: ## Run continuous-training pipeline locally (drift-gated)
	$(PYTHON) $(ML_ORCH) ct_pipeline --models $(MODELS) --epochs $(EPOCHS_CT) --use-wandb

flyte-ct-remote: ## Run continuous-training pipeline on Flyte cluster
	$(PYTHON) $(ML_ORCH) ct_pipeline --remote --models $(MODELS) --epochs $(EPOCHS_CT) --use-wandb

# ── Airflow Data Pipeline ────────────────────────────────────────────────
airflow-init: ## Initialise Airflow DB + admin user (Docker)
	docker compose -f docker/docker-compose-ml.yml --profile airflow run --rm airflow-init

airflow-start: ## Start Airflow webserver + scheduler (Docker)
	docker compose -f docker/docker-compose-ml.yml --profile airflow up -d airflow-webserver airflow-scheduler
	@echo "Airflow running — UI: http://localhost:8080 (admin/admin)"

airflow-stop: ## Stop Airflow services
	docker compose -f docker/docker-compose-ml.yml --profile airflow stop airflow-webserver airflow-scheduler
	@echo "Airflow stopped"

airflow-trigger: ## Manually trigger the data pipeline DAG
	docker compose -f docker/docker-compose-ml.yml --profile airflow exec airflow-webserver \
		airflow dags trigger atm_forecast_data_pipeline
	@echo "DAG triggered — check http://localhost:8080 for status"
