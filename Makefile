.DEFAULT_GOAL := help

PYTHON := $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,python3)
UV := $(shell which uv 2>/dev/null)

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install development dependencies in current environment
ifdef UV
	uv pip install -e ".[dev]"
else
	$(PYTHON) -m pip install -e ".[dev]"
endif

.PHONY: test
test: ## Run unit and integration tests with pytest
	$(PYTHON) -m pytest

.PHONY: test-fast
test-fast: ## Run fast unit tests (skip slow tests)
	$(PYTHON) -m pytest -m "not slow"

.PHONY: typecheck
typecheck: ## Check static typing with mypy in strict mode
	$(PYTHON) -m mypy src

.PHONY: lint
lint: ## Run linter and formatting checks with ruff
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

.PHONY: format
format: ## Auto-format and auto-fix code with ruff
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

.PHONY: doctor
doctor: ## Run system diagnostic and health checks
	$(PYTHON) -m filesystem_rag_mcp.cli --doctor

.PHONY: stats
stats: ## Print full-text and vector index statistics
	$(PYTHON) -m filesystem_rag_mcp.cli stats

.PHONY: search
search: ## Run a hybrid search query via the CLI (usage: make search QUERY="...")
ifndef QUERY
	$(error Usage: make search QUERY="<your query>" [TOP_K=5] [GLOB="docs/**/*.md"])
endif
	$(PYTHON) -m filesystem_rag_mcp.cli search "$(QUERY)" \
		$(if $(TOP_K),--top-k $(TOP_K)) \
		$(if $(GLOB),--glob "$(GLOB)") \
		$(if $(ALPHA),--alpha $(ALPHA))

.PHONY: index
index: ## Build/refresh the on-disk index (usage: make index [THOROUGH=1] [ROOT_DIR=/path])
	$(PYTHON) -m filesystem_rag_mcp.cli index \
		$(if $(THOROUGH),--thorough) \
		$(if $(ROOT_DIR),--root-dir "$(ROOT_DIR)") \
		$(if $(DATA_DIR),--data-dir "$(DATA_DIR)")

.PHONY: check ci
check: lint typecheck test ## Run full CI pipeline locally (lint, typecheck, test)
	@echo "\033[32mAll checks passed!\033[0m"

ci: check ## Alias for check

.PHONY: clean
clean: ## Remove caches, build artifacts, and test databases
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
