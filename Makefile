.PHONY: help install install-dev lint format test test-cov serve train docker-build docker-up docker-down clean

# ── Variables ────────────────────────────────────────────────────────────
PYTHON    ?= python
PIP       ?= pip
PYTEST    ?= pytest
RUFF      ?= ruff
UVICORN   ?= uvicorn

# ── Default ──────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ─────────────────────────────────────────────────────────
install: ## Install runtime dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install all dependencies (including dev)
	$(PIP) install --upgrade pip
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
	docker compose -f docker/docker-compose.yml up -d

docker-down: ## Stop all services
	docker compose -f docker/docker-compose.yml down

docker-dev: ## Start dev environment with hot-reload
	docker compose -f docker/docker-compose.yml --profile dev up -d api-dev

# ── Cleanup ──────────────────────────────────────────────────────────────
clean: ## Remove build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
