.PHONY: help install dev train tune dashboard docker lint format test clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install production dependencies"
	@echo "  make dev        - Install development dependencies"
	@echo "  make train      - Train models"
	@echo "  make tune       - Run hyperparameter optimization"
	@echo "  make dashboard  - Launch Streamlit dashboard"
	@echo "  make docker     - Build and run Docker containers"
	@echo "  make lint       - Run linting checks"
	@echo "  make format     - Format code"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean cache files"

install:
	uv sync --no-dev

dev:
	uv sync --dev

train:
	python src/modeling/train.py

tune:
	python src/modeling/optimize.py

dashboard:
	streamlit run app/streamlit_app.py

docker:
	docker-compose -f docker/docker-compose.yml up --build

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
