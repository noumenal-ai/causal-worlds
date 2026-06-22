.PHONY: install fmt lint type test check validate hooks

install:  ## sync env + dev tools
	uv sync

fmt:  ## auto-format
	uv run ruff format .

lint:  ## lint (with autofix)
	uv run ruff check --fix .

type:  ## strict type-check
	uv run mypy

test:  ## tests + coverage floor
	uv run pytest

check:  ## the read-only gate (what CI runs, minus tests)
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy

validate: check test  ## the full gate — run before every commit

hooks:  ## install pre-commit hooks
	uv run pre-commit install
