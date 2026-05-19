# Docker Setup — Multimodal AML Detection

Phase 2 containerization deliverable.

---

## Contents

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage image (python:3.11-slim builder → slim runtime) |
| `../docker-compose.yaml` | Full stack: training, API, Prometheus, Grafana |
| `../.env.example` | All supported environment variables with descriptions |
| `../scripts/test_container.sh` | Container smoke test (import check + API health) |

---

## Quick Start

```bash
# 1. Copy env template
cp .env.example .env          # edit values as needed

# 2. Build image
make docker_build

# 3. Start full stack (API + Prometheus + Grafana)
make docker_up

# 4. Verify containers are healthy
docker compose ps

# 5. Tear down
make docker_down
```

---

## Environment Variables

All variables have safe defaults baked into the image. Override via `.env` or
`--env KEY=VALUE` at runtime.

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `file:///app/mlruns` | MLflow server URI; set to `http://mlflow:5000` for a remote server |
| `AML_THRESHOLD` | `0.5` | Risk score threshold above which a transaction is flagged |
| `PORT` | `8000` | Port the API listens on |
| `LOG_LEVEL` | `INFO` | Python log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering (required for real-time logs) |
| `PYTHONDONTWRITEBYTECODE` | `1` | Skip `.pyc` file generation |
| `GF_SECURITY_ADMIN_USER` | `admin` | Grafana admin username |
| `GF_SECURITY_ADMIN_PASSWORD` | `aml_admin` | Grafana admin password — **change in production** |

---

## Build

### Standard build

```bash
docker build -f dockerfiles/Dockerfile -t multimodal_anti_money_laundering:latest .
```

### Build with custom tag

```bash
docker build -f dockerfiles/Dockerfile \
  -t multimodal_anti_money_laundering:v1.2.0 \
  --label "git.commit=$(git rev-parse --short HEAD)" \
  .
```

---

## Run

### API server (inference)

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  --env-file .env \
  multimodal_anti_money_laundering:latest
```

The API is then available at `http://localhost:8000`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check → `{"status": "ok"}` |
| `/predict` | POST | Score a transaction |
| `/metrics` | GET | Prometheus metrics scrape endpoint |
| `/docs` | GET | Interactive Swagger UI |

### Training job

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  --env-file .env \
  multimodal_anti_money_laundering:latest \
  python -m multimodal_anti_money_laundering.train_model
```

### Interactive shell (debugging)

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  --env-file .env \
  multimodal_anti_money_laundering:latest \
  /bin/bash
```

---

## Docker Compose Services

| Service | Port | Purpose |
|---------|------|---------|
| `train` | — | One-shot training job; exits when training completes |
| `api` | 8000 | FastAPI inference server with Prometheus `/metrics` |
| `prometheus` | 9090 | Metrics scraper; waits for `api` to be healthy |
| `grafana` | 3000 | Dashboard; default login `admin / aml_admin` |

```bash
# Start only the API + monitoring stack
docker compose up -d api prometheus grafana

# Run a one-off training job
docker compose run --rm train

# Follow API logs
docker compose logs -f api

# Check health status of all services
docker compose ps
```

---

## Volume Mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./data` | `/app/data` | Raw and processed datasets (read by training) |
| `./models` | `/app/models` | Saved model artifacts (written by training, read by API) |
| `./mlruns` | `/app/mlruns` | MLflow run metadata (when using local filesystem tracking) |

Data and models are deliberately excluded from the image (see `.dockerignore`)
and provided at runtime via mounts, keeping the image small and build times fast.

---

## Container Testing

Run the full smoke test suite (build → import check → CLI → API health → predict):

```bash
make docker_test
# or directly:
bash scripts/test_container.sh
# or with a specific tag:
bash scripts/test_container.sh multimodal_anti_money_laundering:v1.2.0
```

The script exits `0` if all checks pass, `1` if any check fails. CI runs this
on every PR that modifies `dockerfiles/Dockerfile` or `requirements.txt`.

---

## Environment Consistency

The image is built on CI (GitHub Actions) using the same `Dockerfile` and
`requirements.txt` as local development, ensuring identical package versions
inside and outside the container. To verify:

```bash
# Check installed package versions inside the container
docker run --rm multimodal_anti_money_laundering:latest pip list

# Compare with local environment
pip list

# Run the test suite inside the container to confirm identical results
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  multimodal_anti_money_laundering:latest \
  pytest tests/ -q
```
