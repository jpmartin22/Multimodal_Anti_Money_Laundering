.PHONY: install dev data train predict test lint format clean docker_build docker_run docs

# Note: 'uv' is a faster alternative to pip. Install with: pip install uv
# Then replace 'pip install' with 'uv pip install' in the commands below.

install:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -r requirements_dev.txt
	pre-commit install

data:
	python -m multimodal_anti_money_laundering.data.make_dataset

train:
	python -m multimodal_anti_money_laundering.train_model

predict:
	python -m multimodal_anti_money_laundering.predict_model

test:
	pytest tests/

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

docker_build:
	docker build -t multimodal_anti_money_laundering -f dockerfiles/Dockerfile .

docker_run:
	docker run --rm multimodal_anti_money_laundering

docs:
	mkdocs serve
