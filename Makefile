.PHONY: clean clean-test clean-docs clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

ifeq (, $(shell which snakeviz))
	PROFILE = pytest --profile-svg
	PROFILE_RESULT = prof/combined.svg
	PROFILE_VIEWER = $(BROWSER)
else
    PROFILE = pytest --profile
    PROFILE_RESULT = prof/combined.prof
	PROFILE_VIEWER = snakeviz
endif

install: clean ## Install all package and development dependencies for testing to the active Python's site-packages
	uv sync --extra testing --extra linting --extra dev

format: ## format code ruff formatter
	uv run ruff format histoplus tests

lint: ## Check style with ruff linter
	uv run ruff check --fix histoplus tests

typing: ## Check static typing with mypy
	uv run mypy histoplus

pre-commit-checks: ## Run pre-commit checks on all files
	uv run pre-commit run --hook-stage manual --all-files

lint-all: pre-commit-checks lint typing ## Run all linting checks.

test: ## Run tests quickly with the default Python
	uv run pytest
