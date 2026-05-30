# PHASE 3: Continuous Machine Learning (CML) & Deployment

## Overview
Phase 3 implements continuous integration/continuous deployment (CI/CD) pipelines and productionizes Multimodal Anti Money Laundering on cloud infrastructure. This phase covers automated testing, containerized workflows, CML integration, and multi-platform deployment options including GCP, Cloud Run, and serverless functions.

---

## 1. Continuous Integration & Testing

- [x] **Unit Tests**: pytest test scripts for data processing and model components (`tests/test_model.py`, `tests/test_serving.py`)
- [x] **Integration Tests**: Full pipeline integration tests (`tests/test_integration.py`)
- [x] **Test Coverage**: >80% code coverage with pytest-cov (reported via Codecov)
- [x] **GitHub Actions - Tests**: CI workflow runs on every push/PR (`.github/workflows/ci.yml`)
  - [x] Trigger on: push to master and PRs
  - [x] Test across Python 3.10 and 3.11
  - [x] Report coverage metrics via Codecov
- [x] **GitHub Actions - Code Quality**:
  - [x] ruff linter
  - [x] ruff format check
- [x] **GitHub Actions - Docker Build**: Docker build workflow (`.github/workflows/docker-build.yml`)
  - [x] Build on push to master
  - [x] Smoke-test built image
- [x] **Pre-commit Hooks**: `.pre-commit-config.yaml` configured for ruff formatting and linting
- [x] **Test Documentation**: CI steps documented in `.github/workflows/ci.yml` and CONTRIBUTING.md

---

## 2. Continuous Docker Building & CML

- [x] **Automated Docker Builds**: Docker build pipeline (`.github/workflows/docker-build.yml`)
  - [x] Triggered on commits to master
  - [x] Manual workflow dispatch supported
- [x] **Docker Push**: Push to GitHub Container Registry (ghcr.io)
- [x] **CML Workflow**: `.github/workflows/cml.yml` — generates metrics and posts to PR
  - [x] Generates performance metrics
  - [x] Creates visualizations/plots
  - [x] Comments results on PR (pending REPO_TOKEN secret — Preshita)
- [x] **CML Metrics Output**: `reports/cml_report.md`, `reports/figures/cml_auc_pr.png`
- [x] **Model Comparison**: CML report compares current vs. baseline AUC-PR

---

## 3. Deployment on GCP

- [x] **FastAPI Service**: `src/multimodal_anti_money_laundering/serving/api.py`
  - [x] `/predict` inference endpoint with full request validation
  - [x] `/health` health check endpoint
  - [x] API documented in `docs/api.md`
- [x] **Cloud Run Deployment**: `.github/workflows/deploy-cloudrun.yml`
  - [x] Dockerfile optimized for Cloud Run (`dockerfiles/Dockerfile`)
  - [x] Deploy workflow configured (pending GCP secrets — Rajani)
- [x] **HuggingFace Spaces (Option C)**: `deploy/huggingface/app.py` Gradio app
- [x] **Load Testing**: `tests/locustfile.py` — 50 concurrent users, p50/p95 latency
- [x] **Monitoring Setup**: Prometheus + Grafana via `docker-compose.yaml`
  - [x] Prometheus metrics exported from API (`/metrics` endpoint)
  - [x] Grafana dashboard (`grafana/dashboards/aml_dashboard.json`)

---

## 4. Documentation & Repository Updates

- [x] **Comprehensive README**: Updated with architecture diagram, CI/CD overview, Phase 3 results
- [x] **Deployment Guide**: `deploy/DEPLOYMENT.md` — GCP setup, Cloud Run, env vars, rollback
- [x] **API Documentation**: `docs/api.md` — all endpoints with request/response schemas and curl examples
- [x] **CML Guide**: `docs/cml_docker_guide.md`
- [x] **Model Card**: `MODEL_CARD.md` — description, training data, metrics, limitations, intended use
- [x] **Screenshots**: `docs/screenshots/` — Swagger UI, terminal output
- [x] **Resource Cleanup**: `CLEANUP.md` — GCP teardown instructions
- [x] **Contributing Guide**: `CONTRIBUTING.md` updated with CI/CD and testing requirements
- [x] **Changelog**: `CHANGELOG.md` maintained

---

## Member A (Jaya) — Phase 3 Specific Deliverables

- [x] Late-fusion MLP trained — GraphSAGE + BiLSTM + DistilBERT (AUC-PR = **0.9975**)
- [x] Platt calibration fitted on val logits
- [x] Ablation study — 4 variants, results in `reports/ablation_results.json` + `.png`
- [x] SHAP explainability — force plot + summary plot in `reports/` and `outputs/`
- [x] REPORT.md updated with fusion results, ablation table, SHAP analysis
- [x] `fusion_mlp.pt` + `fusion_calibrator.joblib` tracked via DVC

---

> **Checklist:** Use this as a guide for documenting your Phase 3 deliverables.
