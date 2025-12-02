# ============================================================================
# Shadow Deployment & Drift Detection Platform - Makefile
# ============================================================================
#
# This Makefile provides convenient commands for development, testing,
# and deployment of the MLOps platform.
#
# Usage:
#     make help          Show all available commands
#     make install       Install dependencies
#     make test          Run all tests
#     make run           Start the API server
#
# ============================================================================

.PHONY: help install install-dev test test-cov lint format clean docker-build \
        docker-run docker-stop run run-prod drift-check feast-apply docs

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m  # No Color

# Project configuration
PROJECT_NAME := shadow-mlops
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := $(PROJECT_NAME):latest
PORT := 8000

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# Help
# ============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║     Shadow Deployment & Drift Detection Platform             ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf ""} \
		/^[a-zA-Z_-]+:.*##/ { printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ============================================================================
# Installation
# ============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install pytest pytest-cov pytest-asyncio black ruff mypy pre-commit
	pre-commit install || true
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

sync-deps: ## Sync dependencies with pyproject.toml
	@echo "$(BLUE)Syncing dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Dependencies synced$(NC)"

# ============================================================================
# Development
# ============================================================================

run: ## Run the development server
	@echo "$(BLUE)Starting development server on port $(PORT)...$(NC)"
	$(PYTHON) -m uvicorn src.api:app --reload --host 0.0.0.0 --port $(PORT)

run-prod: ## Run the production server
	@echo "$(BLUE)Starting production server on port $(PORT)...$(NC)"
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port $(PORT) --workers 4

debug: ## Run with debug logging
	@echo "$(BLUE)Starting server with debug logging...$(NC)"
	LOG_LEVEL=DEBUG $(PYTHON) -m uvicorn src.api:app --reload --host 0.0.0.0 --port $(PORT)

# ============================================================================
# Testing
# ============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v
	@echo "$(GREEN)✓ Tests completed$(NC)"

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=src --cov=monitoring --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-fast: ## Run tests without slow tests
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-unit: ## Run only unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/test_models.py tests/test_drift.py -v

test-api: ## Run API tests
	@echo "$(BLUE)Running API tests...$(NC)"
	$(PYTHON) -m pytest tests/test_api.py -v

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linters (ruff)
	@echo "$(BLUE)Running linters...$(NC)"
	$(PYTHON) -m ruff check src/ monitoring/ tests/
	@echo "$(GREEN)✓ Linting completed$(NC)"

lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(NC)"
	$(PYTHON) -m ruff check src/ monitoring/ tests/ --fix
	@echo "$(GREEN)✓ Linting issues fixed$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	$(PYTHON) -m black src/ monitoring/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code format...$(NC)"
	$(PYTHON) -m black src/ monitoring/ tests/ --check
	@echo "$(GREEN)✓ Format check passed$(NC)"

typecheck: ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	$(PYTHON) -m mypy src/ monitoring/ --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking completed$(NC)"

quality: lint format-check typecheck ## Run all quality checks
	@echo "$(GREEN)✓ All quality checks passed$(NC)"

# ============================================================================
# Drift Detection
# ============================================================================

drift-check: ## Run drift detection with sample data
	@echo "$(BLUE)Running drift detection...$(NC)"
	$(PYTHON) -m monitoring.detect_drift --generate-sample --verbose
	@echo "$(GREEN)✓ Drift detection completed$(NC)"

drift-report: ## Generate drift report and save to file
	@echo "$(BLUE)Generating drift report...$(NC)"
	$(PYTHON) -m monitoring.detect_drift --generate-sample --output logs/drift_report.json
	@echo "$(GREEN)✓ Report saved to logs/drift_report.json$(NC)"

# ============================================================================
# Feast Feature Store
# ============================================================================

feast-apply: ## Apply Feast feature definitions
	@echo "$(BLUE)Applying Feast feature definitions...$(NC)"
	cd feature_repo && feast apply
	@echo "$(GREEN)✓ Feature store updated$(NC)"

feast-materialize: ## Materialize features to online store
	@echo "$(BLUE)Materializing features...$(NC)"
	cd feature_repo && feast materialize-incremental $$(date +%Y-%m-%dT%H:%M:%S)
	@echo "$(GREEN)✓ Features materialized$(NC)"

feast-ui: ## Launch Feast UI
	@echo "$(BLUE)Starting Feast UI...$(NC)"
	cd feature_repo && feast ui

# ============================================================================
# Docker
# ============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE)$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -d --name $(PROJECT_NAME) -p $(PORT):8000 $(DOCKER_IMAGE)
	@echo "$(GREEN)✓ Container running on port $(PORT)$(NC)"

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker stop $(PROJECT_NAME) || true
	docker rm $(PROJECT_NAME) || true
	@echo "$(GREEN)✓ Container stopped$(NC)"

docker-logs: ## View Docker container logs
	docker logs -f $(PROJECT_NAME)

compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"

compose-down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

compose-logs: ## View docker-compose logs
	docker-compose logs -f

# ============================================================================
# Documentation
# ============================================================================

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	$(PYTHON) -m pdoc src monitoring -o docs/api
	@echo "$(GREEN)✓ Documentation generated in docs/api/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	$(PYTHON) -m pdoc src monitoring --http localhost:8080

# ============================================================================
# Utilities
# ============================================================================

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info build/ dist/
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf logs/*.json logs/*.log
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-docker: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker-compose down --volumes --rmi local || true
	docker rmi $(DOCKER_IMAGE) || true
	@echo "$(GREEN)✓ Docker resources cleaned$(NC)"

tree: ## Show project structure
	@echo "$(BLUE)Project Structure:$(NC)"
	@tree -I '__pycache__|*.egg-info|.git|.pytest_cache|htmlcov|.mypy_cache' -a --dirsfirst

health: ## Check API health
	@echo "$(BLUE)Checking API health...$(NC)"
	@curl -s http://localhost:$(PORT)/health | python -m json.tool || echo "$(RED)API not responding$(NC)"

benchmark: ## Run simple benchmark
	@echo "$(BLUE)Running benchmark...$(NC)"
	@echo "Sending 100 requests to /predict endpoint..."
	@ab -n 100 -c 10 -T application/json -p tests/fixtures/sample_request.json http://localhost:$(PORT)/predict/ 2>/dev/null | grep -E "(Requests per second|Time per request|Failed requests)"

# ============================================================================
# CI/CD Helpers
# ============================================================================

ci-test: ## Run CI test suite
	@echo "$(BLUE)Running CI tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short --junitxml=test-results.xml

ci-lint: ## Run CI linting
	@echo "$(BLUE)Running CI linting...$(NC)"
	$(PYTHON) -m ruff check src/ monitoring/ tests/ --output-format=github

ci-full: ci-lint ci-test ## Run full CI pipeline locally

# ============================================================================
# Git Helpers
# ============================================================================

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

git-clean: ## Show files that would be cleaned by git
	git clean -nxd

release: ## Create a release (usage: make release VERSION=1.0.0)
	@if [ -z "$(VERSION)" ]; then echo "$(RED)VERSION required (make release VERSION=x.y.z)$(NC)"; exit 1; fi
	@echo "$(BLUE)Creating release $(VERSION)...$(NC)"
	git tag -a v$(VERSION) -m "Release $(VERSION)"
	git push origin v$(VERSION)
	@echo "$(GREEN)✓ Release $(VERSION) created$(NC)"
