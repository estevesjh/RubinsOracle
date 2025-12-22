.PHONY: lint format test coverage docs clean install help

help:
	@echo "Available commands:"
	@echo "  make install   - Install package and pre-commit hooks"
	@echo "  make format    - Format code with ruff"
	@echo "  make lint      - Run all linters (ruff, mypy, interrogate)"
	@echo "  make test      - Run tests"
	@echo "  make coverage  - Run tests with coverage report"
	@echo "  make docs      - Build documentation"
	@echo "  make clean     - Remove generated files"

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	mypy rubin_oracle
	interrogate -v rubin_oracle

format:
	ruff format .
	ruff check --fix .

test:
	pytest -v

coverage:
	pytest --cov --cov-report=html --cov-report=term

docs:
	cd docs && make html

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	rm -rf docs/_build
	rm -rf *.egg-info
	rm -rf dist build
