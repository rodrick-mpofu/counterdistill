.PHONY: \
	help install dev \
	train tune explain aggregate evaluate \
	dashboard mlflow \
	lint format format-check typecheck test check \
	docker docker-down \
	clean

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL ?= random_forest
RUN_ID ?=
MODEL_NAME ?= RandomForestClassifier
N_CLUSTERS ?= 6


# --------------------------------------------------
# Help
# --------------------------------------------------

help:
	@echo "CounterDistill commands:"
	@echo ""
	@echo "Environment"
	@echo "  make install       Install production dependencies"
	@echo "  make dev           Install development dependencies"
	@echo ""
	@echo "Modeling"
	@echo "  make train         Train a model"
	@echo "  make tune          Run hyperparameter optimization"
	@echo ""
	@echo "Explainability"
	@echo "  make explain       Generate DiCE + SHAP explanations"
	@echo "  make aggregate     Cluster counterfactuals and extract rules"
	@echo "  make evaluate      Generate final evaluation report"
	@echo ""
	@echo "Applications"
	@echo "  make dashboard     Launch Streamlit dashboard"
	@echo "  make mlflow        Launch local MLflow server"
	@echo ""
	@echo "Quality"
	@echo "  make lint          Run Ruff linting"
	@echo "  make format        Format code with Ruff"
	@echo "  make format-check  Check formatting without modifying files"
	@echo "  make typecheck     Run mypy"
	@echo "  make test          Run tests"
	@echo "  make check         Run all release quality checks"
	@echo ""
	@echo "Docker"
	@echo "  make docker        Build and start Docker services"
	@echo "  make docker-down   Stop Docker services"
	@echo ""
	@echo "Utilities"
	@echo "  make clean         Remove Python/tool caches"


# --------------------------------------------------
# Environment
# --------------------------------------------------

install:
	uv sync --no-dev

dev:
	uv sync --dev


# --------------------------------------------------
# Modeling
# --------------------------------------------------

train:
	uv run python -m src.modeling.train model=$(MODEL)

tune:
	uv run python -m src.modeling.tune model=$(MODEL)


# --------------------------------------------------
# Explainability / Distillation
# --------------------------------------------------

explain:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "RUN_ID is required."; \
		echo "Example:"; \
		echo "  make explain RUN_ID=<mlflow-run-id>"; \
		exit 1; \
	fi
	uv run python -m src.explainability.explain \
		model=$(MODEL) \
		+mlflow.run_id=$(RUN_ID)

aggregate:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "RUN_ID is required."; \
		echo "Example:"; \
		echo "  make aggregate RUN_ID=<run-id>"; \
		exit 1; \
	fi
	uv run python -m src.aggregation.aggregate \
		--run-id $(RUN_ID) \
		--model-name $(MODEL_NAME) \
		--n-clusters $(N_CLUSTERS)

evaluate:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "RUN_ID is required."; \
		echo "Example:"; \
		echo "  make evaluate RUN_ID=<run-id>"; \
		exit 1; \
	fi
	uv run python -m src.evaluation.report \
		--run-id $(RUN_ID) \
		--model-name $(MODEL_NAME)


# --------------------------------------------------
# Applications
# --------------------------------------------------

dashboard:
	uv run streamlit run app/app.py

mlflow:
	uv run mlflow server \
		--backend-store-uri sqlite:///database/mlflow.db \
		--host 127.0.0.1 \
		--port 5000 \
		--workers 1


# --------------------------------------------------
# Quality
# --------------------------------------------------

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

format-check:
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest

check:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/
	uv run pytest


# --------------------------------------------------
# Docker
# --------------------------------------------------

docker:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
