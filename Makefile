# EEG-RAG Makefile
# Simple commands for building, running, and testing

.PHONY: help install install-dev docker-build docker-up docker-down test lint clean setup-data

help:
	@echo "EEG-RAG Development Commands"
	@echo "============================"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install        - Install EEG-RAG (pip, editable)"
	@echo "  make ui             - Start web UI (http://localhost:8501)"
	@echo "  make docker-up      - Start all services via Docker"
	@echo ""
	@echo "Development:"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Run linters"
	@echo "  make clean          - Remove build artifacts"
	@echo ""
	@echo "Web UI:"
	@echo "  make ui             - Start Streamlit web interface"
	@echo "  make ui-dev         - Start with auto-reload"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-up      - Start full stack (app + neo4j + redis)"
	@echo "  make docker-up-lite - Start minimal (just app)"
	@echo "  make docker-down    - Stop all containers"
	@echo "  make docker-shell   - Open shell in container"
	@echo ""
	@echo "Data:"
	@echo "  make setup-data              - Create data directories"
	@echo "  make download-benchmark-data - Download Roy et al. 2019 data"
	@echo "  make demo                    - Run demo"

install:
	@echo "Installing EEG-RAG..."
	pip install -e .
	@echo "Installation complete!"

install-dev:
	@echo "Installing EEG-RAG with dev dependencies..."
	pip install -e ".[dev,full]"
	@echo "Development installation complete!"

install-full:
	pip install -e ".[full]"

docker-build:
	@echo "Building Docker image..."
	docker build -t eeg-rag:latest -f docker/Dockerfile .

docker-up:
	@echo "Starting EEG-RAG services..."
	docker compose -f docker/docker-compose.yml up -d
	@echo "Services started!"
	@echo "  EEG-RAG API: http://localhost:8000"
	@echo "  Neo4j: http://localhost:7474"

docker-up-lite:
	@echo "Starting EEG-RAG (lite mode)..."
	docker compose -f docker/docker-compose.lite.yml up -d
	@echo "EEG-RAG started at http://localhost:8000"

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-shell:
	docker exec -it eeg-rag-app /bin/bash

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f

test:
	@echo "Running tests..."
	python -m pytest tests/ -v --tb=short

test-fast:
	python -m pytest tests/ -v --tb=short -m "not slow and not integration"

test-coverage:
	python -m pytest tests/ --cov=src/eeg_rag --cov-report=html --cov-report=term

lint:
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

typecheck:
	mypy src/eeg_rag --ignore-missing-imports

# Web UI
ui:
	@echo "Starting EEG-RAG Web UI..."
	streamlit run src/eeg_rag/web_ui/app.py --server.port 8501

ui-dev:
	@echo "Starting EEG-RAG Web UI (dev mode with auto-reload)..."
	streamlit run src/eeg_rag/web_ui/app.py --server.port 8501 --server.runOnSave true

setup-data:
	mkdir -p data/raw data/processed data/embeddings/cache data/systematic_review logs
	@echo "Data directories created!"

download-benchmark-data:
	@echo "Downloading Roy et al. 2019 benchmark data..."
	mkdir -p data/systematic_review
	curl -o data/systematic_review/roy_et_al_2019_data_items.csv \
		https://raw.githubusercontent.com/hubertjb/dl-eeg-review/master/data/data_items.csv
	@echo "Benchmark data downloaded!"

demo:
	python examples/demo_all_components.py

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-docker:
	docker compose -f docker/docker-compose.yml down -v --rmi local
