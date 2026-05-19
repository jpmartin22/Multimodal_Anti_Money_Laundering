# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## Overview
Phase 2 focuses on scaling and operationalizing Multimodal Anti Money Laundering by implementing containerization, advanced monitoring, profiling, experiment tracking, and comprehensive logging. This phase ensures your model can be reliably deployed, monitored in production, and continuously improved through systematic experimentation.

---

## 1. Containerization

- [x] **Dockerfile Creation**: Multi-stage `python:3.11-slim` images — `dockerfiles/Dockerfile` (general), `dockerfiles/Dockerfile.graphsage` (Member A), `dockerfiles/Dockerfile.sagemaker` (Member D)
- [x] **Base Image Selection**: `python:3.11-slim` (builder + runtime stages) across all images
- [x] **Environment Variables**: Defined in Dockerfile (`MLFLOW_TRACKING_URI`, `AML_THRESHOLD`, `LOG_LEVEL`); documented in `.env.example`
- [x] **Build Instructions**: `dockerfiles/README.md` — build examples, custom tags, CI usage
- [x] **Run Instructions**: `dockerfiles/README.md` + `README.md` — API, training, and debug run modes with volume/network config
- [x] **Container Testing**: Smoke tests via `docker run ... /health` and `/predict` endpoints; verified locally
- [x] **Docker Compose**: `docker-compose.yaml` — `graphsage-train`, `train`, `api`, `prometheus`, `grafana` services with healthchecks and `env_file`
- [x] **Environment Consistency**: `.dockerignore` excludes `.venv`, `data/`, `mlruns/` from build context; `requirements.serve.txt` + `requirements-graphsage.txt` pin serving/training deps separately

---

## 2. Monitoring & Debugging

- [x] **Logging for Debugging**: Rotating file handler (5 MB, 3 backups) + console handler in GraphSAGE and BiLSTM training scripts
- [x] **Model Assertion Checks**: Pre-training assertions — NaN detection, shape validation, label range, class imbalance warnings (`train_graphsage.py`, `train_bilstm.py`)
- [x] **Training Validation**: Sanity checks on features, edge_index, and labels before model construction
- [x] **Production Monitoring**: Evidently AI drift reports for graph, BiLSTM, and text modalities (`monitoring/drift_report.py`)
- [x] **Metrics Dashboard**: Prometheus + Grafana — AUC-PR, F1, FPR gauges per model branch; request counter and latency histogram (`monitoring/metrics_exporter.py`)
- [ ] **Debugging Tools**: pdb/ipdb interactive debugging setup
- [ ] **Debugging Documentation**: Debug guide for containerized environment
- [ ] **Debug Scenarios**: Example scenario + solution documents

---

## 3. Profiling & Optimization

- [x] **CPU Profiling**: cProfile on GraphSAGE training (`profile_graphsage.py`) → top-50 hotspots saved to `reports/profiling/graphsage_cprofile.txt`
- [x] **CPU Profiling (BiLSTM)**: cProfile on BiLSTM training (`profile_bilstm.py`) → `reports/profiling/bilstm_cprofile.txt`
- [x] **Memory Profiling**: `memory_profiler` line-by-line analysis for GraphSAGE and BiLSTM → `graphsage_memory.txt`, `bilstm_memory.txt`
- [x] **Profiling Results**: Bottlenecks identified — SAGEConv forward pass and BCEWithLogitsLoss dominate GraphSAGE runtime
- [x] **Optimization 1**: Reduced hidden channels 256→128 cuts per-epoch time from 0.69s → 0.36s on 8k-node subset
- [x] **Optimization 2**: Gradient clipping (`max_norm=1.0`) stabilises training, allows larger learning rate without divergence
- [x] **Performance Benchmarks**: Before/after documented in `reports/profiling/graphsage_benchmark.json` — **1.92x speedup**
- [x] **Optimization Documentation**: Explained in `README.md` Phase 2 profiling section and profiling script docstrings

---

## 4. Experiment Management & Tracking

- [x] **MLflow Setup**: MLflow tracking initialised; experiments logged to local `mlruns/`; UI at `mlflow ui --port 5000`
- [x] **Metric Logging**: Train loss, val AUC-PR, val F1 logged every `eval_every` epochs for GraphSAGE and BiLSTM
- [x] **Parameter Logging**: All hyperparameters logged via `mlflow.log_params(OmegaConf.to_container(cfg))` — full config captured per run
- [x] **Model Artifact Logging**: `graphsage_best.pt`, `graphsage_encoder.pt`, `graphsage_metrics.json` logged as MLflow artifacts
- [x] **Experiment Comparison**: 3 GraphSAGE runs compared — lr, hidden_dim, dropout sweep; results in `reports/graphsage_experiment_comparison.json`
- [x] **Best Model Selection**: Exp 3 (lr=0.001, h=256, d=0.5) selected — test AUC-PR=0.9299; criteria documented in README
- [x] **Model Registry**: MLflow model registration + staging→production lifecycle promotion (`src/models/register_model.py`)
- [x] **Experiment Documentation**: Results table in `README.md` and `reports/graphsage_experiment_comparison.json`
- [ ] **Visualization**: Performance comparison charts/plots (PR curves across runs)

---

## 5. Application & Experiment Logging

- [x] **Logger Setup**: `logging.getLogger` with `RotatingFileHandler` (5 MB, 3 backups) + `StreamHandler` in all training scripts
- [x] **Log Levels**: DEBUG (file), INFO (console), WARNING/ERROR used appropriately throughout training and serving code
- [x] **Log Messages**: Key events logged — data load, split sizes, pos_weight, per-epoch metrics, best model saves, eval gate results
- [x] **Training Log Example**: Sample output documented in `README.md` Phase 2 section
- [x] **Inference Log Example**: Stub warnings logged when fusion model not loaded (`api.py`)
- [x] **Error Logging**: `logger.exception()` in metrics_exporter; `logger.error()` + `sys.exit(1)` in eval gate
- [x] **Performance Logging**: Epoch timing, total training time, and inference latency (Prometheus histogram) all logged
- [x] **Log Rotation**: `RotatingFileHandler(maxBytes=5*1024*1024, backupCount=3)` configured in GraphSAGE and BiLSTM

---

## 6. Configuration Management

- [x] **Hydra Setup**: `hydra-core==1.3.2` installed; `@hydra.main(version_base="1.3")` decorator on `train_graphsage_hydra.py`
- [x] **Config Files**: YAML configs for model, data, and training in `conf/`
- [x] **Config Structure**: Hierarchical — `conf/config.yaml` composes `model/`, `data/`, `training/` groups
- [x] **Config Example 1**: `conf/model/graphsage_base.yaml` — hidden=128, dropout=0.3 (baseline)
- [x] **Config Example 2**: `conf/model/graphsage_large.yaml` — hidden=256, dropout=0.5 (best, AUC-PR=0.9299)
- [x] **Config Example 3**: `conf/training/fast.yaml` — 5 epochs, 8k nodes (CI smoke test)
- [x] **Override Documentation**: CLI override examples documented in `README.md` and script docstring
- [x] **Config Version Control**: All `conf/` YAML files tracked in git alongside code

---

## 7. Documentation & Repository Updates

- [x] **README Update**:
  - [x] Containerization section — Docker build/run for GraphSAGE, compose stack
  - [x] Profiling guide — cProfile, memory_profiler, benchmark results
  - [x] Experiment tracking — MLflow UI, 3-run comparison table
  - [x] Configuration management — Hydra usage, config hierarchy, CLI overrides
  - [x] Logging — sample training log output, log rotation config
- [x] **Architecture Documentation**: Mermaid diagram in `README.md` showing encoders → fusion → MLOps stack
- [x] **Setup Guide**: Install, PyG extra step, dev hooks documented in `README.md`
- [x] **Tool Integration**: `docker-compose.yaml` wires API → Prometheus → Grafana end-to-end
- [x] **dockerfiles/README.md**: Full Docker build/run reference (Rajani)
- [ ] **Troubleshooting**: Troubleshooting section for common issues
- [ ] **Performance Guide**: Standalone profiling and optimisation guide

---

> **Checklist:** Use this as a guide for documenting your Phase 2 deliverables.
