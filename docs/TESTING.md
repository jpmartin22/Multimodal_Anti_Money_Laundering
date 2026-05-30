# Running tests locally

1. Create and activate your virtual environment (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

2. Run the test suite with coverage:

```bash
pytest --cov=src --cov-report=term
```

3. Run linters and type checks:

```bash
ruff check .
mypy .
```

4. Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

CI behavior

- The repository contains a GitHub Actions workflow at `.github/workflows/ci.yml` that runs linting, mypy, and pytest across Python 3.9–3.11 and enforces a coverage threshold (default 80%).
