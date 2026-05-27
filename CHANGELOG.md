# Changelog

All notable changes to the Multimodal AML Detection project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.3.0] — 2026-05-27 · Phase 3: CI/CD & Deployment

### Added

**Section 1 — Continuous Integration & Testing**
- `tests/test_integration.py`: 30 integration tests covering `eval_gate`, `compute_aml_metrics`, `classification_report`, `regression_report`, `build_features`, `save_json`/`load_json`, `set_seed`, and end-to-end pipeline chain
- `.github/workflows/ci.yml`: Python 3.10 × 3.11 matrix strategy; added `mypy` type-check step
- `.github/workflows/docker-build.yml`: Build all 3 Docker images (`aml-api`, `aml-graphsage`, `aml-hf`), smoke-test `/health`, push to GHCR on master/tags
- `README.md`: Testing section with local run commands, CI workflow table, coverage notes

**Section 2 — CML & Docker**
- `.github/workflows/cml.yml`: CML model report workflow — generates metrics comparison plots and posts as PR comment; uploads figures as artifacts
- `scripts/cml_report.py`: Generates AUC-PR and metrics bar charts; writes `reports/cml_report.md` with embedded image links
- `reports/baseline_snapshot.json`: Stable baseline metrics for CML delta comparison (post fusion training checkpoint)
- `docs/cml_docker_guide.md`: CML and Docker workflow setup and customisation guide

**Section 3 — GCP Deployment**
- `deploy/huggingface/app.py`: Gradio UI for HuggingFace Spaces — simplified transaction inputs, live API scoring, stub fallback
- `.github/workflows/deploy-cloudrun.yml`: Build → push to Artifact Registry → deploy to Cloud Run → smoke-test; triggered on master push and version tags
- `scripts/gcs_model_registry.py`: GCS model registry with `upload`, `download`, `list`, `promote` subcommands and versioned manifests
- `scripts/vertex_ai_training.py`: Submit Vertex AI custom training jobs (sync/async); supports fusion, distilbert, graphsage, bilstm
- `tests/locustfile.py`: Locust load test — `AmlApiUser` (weighted normal/suspicious/burst/invalid traffic) and `BurstUser`
- `scripts/setup_gcp.sh`: One-shot GCP bootstrap — enable APIs, create service account + IAM roles, Workload Identity Federation, Artifact Registry, GCS bucket

**Section 4 — Documentation**
- `README.md`: Phase 3 section with CI/CD Mermaid diagram, deployment options table, GCP quick-start, monitoring reference, cost estimation table; updated project structure
- `deploy/DEPLOYMENT.md`: Automated Cloud Run deployment steps, environment variables table, secrets management, rollback procedures, troubleshooting guide
- `docs/api.md`: Complete HTTP endpoint reference — `/health`, `/predict`, `/ping`, `/invocations`, `/metrics`; request/response schemas; curl and Python examples; error codes
- `CLEANUP.md`: Step-by-step GCP resource teardown instructions, cost monitoring, budget alerts, local cleanup
- `CONTRIBUTING.md`: Development setup, branching convention, CI requirements, testing requirements, deployment process, code style, commit message guide
- `CHANGELOG.md`: This file

### Changed

- `ci.yml`: Upgraded from single Python 3.11 to 3.10 × 3.11 matrix; added mypy step
- `deploy/DEPLOYMENT.md`: Expanded from 206 to ~450 lines with Cloud Run automation, env vars table, rollback, and troubleshooting

### Fixed

- `tests/test_integration.py`: Replaced deprecated `Series.ptp()` with `max() - min()` (removed in pandas 2.x)
- `scripts/cml_report.py`: Fixed import sort order (ruff I001)

---

## [0.2.0] — 2026-05 · Phase 2: Containerization & Monitoring

### Added

- Docker multi-stage builds for API, GraphSAGE training, HuggingFace, and SageMaker targets
- Prometheus + Grafana monitoring stack (`docker-compose.yaml`, `grafana/`, `prometheus/`)
- Evidently AI drift reports per modality (`monitoring/drift_report.py`, `monitoring/run_drift.py`)
- MLflow experiment tracking integration across all training scripts
- Hydra configuration management for all model hyperparameters (`conf/`)
- SHAP explainer for late-fusion predictions (`models/shap_explainer.py`)
- BentoML service definition (`serving/bento_service.py`)
- FastAPI scoring API with Prometheus metrics, `/health`, `/predict`, `/metrics` endpoints
- Pre-commit hooks: ruff, ruff-format, mypy, trailing-whitespace
- Debug utilities (`utils/debug.py`) and structured logging (`logging_config.py`)
- Platt calibration on fusion MLP output

### Changed

- GraphSAGE training migrated to Hydra configuration (`train_graphsage_hydra.py`)
- DistilBERT and BiLSTM training scripts refactored for DVC pipeline compatibility

---

## [0.1.0] — 2026-04 · Phase 1: Model Development

### Added

- GraphSAGE encoder (2-layer, 128-dim, focal loss) on Elliptic Bitcoin dataset
- DistilBERT encoder (64-dim CLS projection, fine-tuned 3 epochs) on synthetic memos
- BiLSTM encoder (2-layer, 64-dim, 30-day rolling window) on PaySim behavioral data
- Late-fusion MLP (256 → 128 → 64 → 1) with dropout 0.3 and ReLU
- XGBoost tabular baseline
- Evaluation suite: AUC-PR, Precision@Recall=0.8, FPR, F1, ROC-AUC
- DVC pipeline for data versioning and reproducible training
- Project scaffold: `src/`, `tests/`, `data/`, `models/`, `reports/`

### Results

| Model | AUC-PR | Precision @ R=0.8 |
|---|---|---|
| GraphSAGE | 0.9299 | 0.9463 |
| DistilBERT | 0.8418 | 1.0000 |
| BiLSTM | 0.9324 | 0.9613 |

---

[Unreleased]: https://github.com/jpmartin22/Multimodal_Anti_Money_Laundering/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jpmartin22/Multimodal_Anti_Money_Laundering/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jpmartin22/Multimodal_Anti_Money_Laundering/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jpmartin22/Multimodal_Anti_Money_Laundering/releases/tag/v0.1.0
