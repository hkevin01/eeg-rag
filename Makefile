# EEG-RAG Development Makefile
# Provides common development tasks and quality checks

.PHONY: help install clean test lint format type-check security health cli docs docker
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip
PYTEST := python -m pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
SAFETY := safety

# Source directories
SRC_DIR := src/eeg_rag
TEST_DIR := tests
DOCS_DIR := docs

help: ## Show this help message
	@echo "EEG-RAG Development Commands"
	@echo "============================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package and dependencies
	@echo "🔧 Installing EEG-RAG and dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo "✅ Installation complete"

clean: ## Clean build artifacts and cache files
	@echo "🧹 Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "✅ Cleanup complete"

test: ## Run all tests
	@echo "🧪 Running test suite..."
	$(PYTEST) $(TEST_DIR) -v --tb=short --durations=10
	@echo "✅ Tests complete"

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	$(PYTEST) $(TEST_DIR) -v --ignore=$(TEST_DIR)/integration/ --tb=short

test-integration: ## Run integration tests only  
	@echo "🧪 Running integration tests..."
	$(PYTEST) $(TEST_DIR)/test_integration_pipeline.py -v --tb=short

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/"

test-benchmark: ## Run performance benchmark tests
	@echo "⚡ Running performance benchmarks..."
	$(PYTEST) $(TEST_DIR)/test_system_resilience.py::TestPerformanceBenchmarks -v -s

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR) --count --statistics --show-source
	@echo "✅ Linting complete"

format: ## Format code with black and isort
	@echo "✨ Formatting code..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(ISORT) $(SRC_DIR) $(TEST_DIR)
	@echo "✅ Formatting complete"

format-check: ## Check code formatting without changes
	@echo "🔍 Checking code formatting..."
	$(BLACK) --check --diff $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only --diff $(SRC_DIR) $(TEST_DIR)

type-check: ## Run type checking with mypy
	@echo "🔍 Running type checks..."
	$(MYPY) $(SRC_DIR) --ignore-missing-imports --strict-optional
	@echo "✅ Type checking complete"

security: ## Run security checks
	@echo "🔒 Running security checks..."
	$(SAFETY) check
	@echo "✅ Security checks complete"

quality: format-check lint type-check ## Run all quality checks
	@echo "✅ All quality checks complete"

health: ## Check system health and requirements
	@echo "🏥 Checking system health..."
	@$(PYTHON) -c "from eeg_rag.utils.common_utils import check_system_health; h = check_system_health(); print(f'System Status: {h.status.value}'); [print(f'  ⚠️  {w}') for w in h.warnings]"
	@echo "📋 Checking dependencies..."
	@$(PIP) check
	@echo "✅ Health check complete"

cli: ## Start interactive CLI
	@echo "🧠 Starting EEG-RAG CLI..."
	$(PYTHON) -m eeg_rag.cli

cli-demo: ## Run CLI demo with sample query
	@echo "🧠 Running CLI demo..."
	$(PYTHON) -m eeg_rag.cli --query "What is the typical alpha frequency range?"

stats: ## Show system statistics
	@echo "📊 System Statistics:"
	$(PYTHON) -m eeg_rag.cli --stats

build: ## Build package
	@echo "📦 Building package..."
	$(PYTHON) -m build
	@echo "✅ Package built in dist/"

docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@mkdir -p $(DOCS_DIR)/api
	@echo "API documentation would go here"
	@echo "✅ Documentation generated"

docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t eeg-rag:latest -f docker/Dockerfile .
	@echo "✅ Docker image built"

docker-run: ## Run in Docker container
	@echo "🐳 Running in Docker container..."
	docker run -it --rm eeg-rag:latest

setup-dev: install ## Setup development environment
	@echo "🚀 Setting up development environment..."
	@mkdir -p data/test logs memory-bank/test
	@echo "OPENAI_API_KEY=your_key_here" > .env.example
	@echo "✅ Development environment setup complete"
	@echo "👉 Next steps:"
	@echo "   1. Copy .env.example to .env and add your API keys"
	@echo "   2. Run 'make test' to verify installation"
	@echo "   3. Run 'make cli' to start the interactive interface"

pre-commit: quality test-unit ## Run pre-commit checks
	@echo "✅ Pre-commit checks passed"

ci: clean install quality test ## Run full CI pipeline locally
	@echo "✅ CI pipeline complete"

# Development workflow targets
dev-setup: setup-dev ## Alias for setup-dev
dev-test: test-unit ## Quick development tests
dev-check: format-check lint ## Quick development checks

# Release workflow targets
release-check: clean install quality test security ## Pre-release validation
release-build: release-check build ## Build release package

# Monitoring and debugging
monitor: ## Monitor system resources during testing
	@echo "📊 Monitoring system resources..."
	@$(PYTHON) -c "import time; from eeg_rag.utils.common_utils import check_system_health; \
	[print(f'{time.strftime(\"%H:%M:%S\")} - CPU: {check_system_health().cpu_percent:.1f}% Memory: {check_system_health().memory_percent:.1f}%') or time.sleep(5) for _ in range(12)]"

debug: ## Run tests with debugging enabled
	@echo "🐛 Running tests with debugging..."
	$(PYTEST) $(TEST_DIR) -v -s --tb=long --pdb-trace

profile: ## Profile performance of critical components
	@echo "⚡ Profiling performance..."
	$(PYTHON) -m cProfile -o profile.stats -m pytest $(TEST_DIR)/test_system_resilience.py::TestPerformanceBenchmarks
	@echo "📊 Profile saved to profile.stats"

performance-test: ## Run performance monitoring tests
	@echo "⚡ Running performance monitoring tests..."
	$(PYTEST) $(TEST_DIR)/test_performance_monitoring.py -v
	@echo "✅ Performance tests complete"

benchmark: ## Run system benchmarks
	@echo "📊 Running system benchmarks..."
	@$(PYTHON) -c "from eeg_rag.monitoring import PerformanceMonitor; from eeg_rag.memory import MemoryManager; from pathlib import Path; import tempfile; import time; monitor = PerformanceMonitor(); temp_dir = tempfile.mkdtemp(); mm = MemoryManager(Path(temp_dir) / 'bench.db'); result = monitor.benchmark_operation(lambda: mm.add_query('test query'), iterations=50, benchmark_name='memory_operations'); print(f'Benchmark: {result.avg_duration_ms:.2f}ms avg, {result.throughput_ops_per_sec:.2f} ops/sec, {result.success_rate:.1%} success')"
	@echo "✅ Benchmark complete"

# Help for specific categories
help-test: ## Show testing commands
	@echo "Testing Commands:"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  test-coverage - Run with coverage report"
	@echo "  test-benchmark - Run performance benchmarks"

help-quality: ## Show quality check commands  
	@echo "Quality Commands:"
	@echo "  format        - Format code with black/isort"
	@echo "  format-check  - Check formatting without changes"
	@echo "  lint          - Run flake8 linting"
	@echo "  type-check    - Run mypy type checking"
	@echo "  security      - Run security checks"
	@echo "  quality       - Run all quality checks"

help-dev: ## Show development commands
	@echo "Development Commands:"
	@echo "  setup-dev     - Setup development environment"
	@echo "  cli           - Start interactive CLI"
	@echo "  health        - Check system health"
	@echo "  stats         - Show system statistics"
	@echo "  monitor       - Monitor system resources"