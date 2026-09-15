.PHONY: lint format typecheck test test-unit test-integration test-e2e check install

install:
	uv sync --all-extras

lint:
	uv run ruff check longline/ tests/

format:
	uv run ruff format longline/ tests/

typecheck:
	uv run mypy longline/

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-e2e:
	uv run pytest tests/e2e/ -v -m e2e

check: lint typecheck test-unit
