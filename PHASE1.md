# PHASE 1: Project Design & Model Development

## 1. Project Proposal

### Scope & Objectives

Money laundering costs the global economy an estimated $800 billion to $2 trillion annually (UNODC, 2023). Traditional rule-based AML systems generate false-positive alert rates as high as 95%, overwhelming compliance teams while sophisticated laundering schemes slip through undetected. The core limitation is that these systems examine transactions in isolation — they cannot jointly exploit the complementary signals carried by transaction graph topology, temporal behavioral patterns, and payment description text.

This project builds a multimodal ML system that fuses all three signal types and wraps it in a complete MLOps lifecycle — from data versioning and experiment tracking through CI/CD-gated deployment and continuous drift monitoring.

**Primary objective:** Train a multimodal AML detector (GraphSAGE + DistilBERT + BiLSTM → late-fusion MLP) that achieves ≥ 0.80 AUC-PR on the Elliptic Bitcoin dataset and outperforms every single-modality baseline.

**Secondary objectives:**
- Demonstrate that fusion outperforms each single-modality baseline on AUC-PR.
- Build and operate a full MLOps pipeline: DVC, MLflow, GitHub Actions CI/CD, AWS SageMaker.
- Satisfy regulatory explainability requirements (SHAP force plots, Google Model Card).

**Success metrics:**

| Metric | Target |
|---|---|
| AUC-PR (primary) | ≥ 0.80 |
| Precision @ Recall = 0.8 | ≥ 0.70 |
| False positive rate | ≤ 5% |
| Inference latency P95 | < 200 ms |
| Fusion > each branch alone | Required |

---

### Dataset Selection

Three publicly available datasets were selected to provide complementary modalities. No licensing agreements are required.

**1. Elliptic Bitcoin Dataset (Kaggle)**
- **Modality:** Transaction graph
- **Size:** 203,769 transactions, 234,355 directed edges, 166 node features (local + aggregated)
- **Labels:** 4,545 illicit (2.2%), 42,019 licit (20.6%), 157,205 unknown (77.2%)
- **Time:** 49 time steps spanning real blockchain activity
- **Why:** Only large-scale, real-world labeled Bitcoin AML benchmark; widely used in GNN research (Weber et al., 2019).
- **Preprocessing:** Build NetworkX DiGraph → PyG `Data` object; StandardScaler on node features; focal loss (γ=2) to handle imbalance; 70/15/15 train/val/test split on labeled nodes only.

**2. PaySim (Kaggle)**
- **Modality:** Behavioral time-series
- **Size:** 6.36 million synthetic mobile-money transactions (TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN)
- **Labels:** 8,213 fraudulent transactions (0.13%)
- **Why:** Provides timestamped account-level transaction history, enabling 30-day rolling window feature extraction that a BiLSTM can process.
- **Preprocessing:** 30-day rolling windows per account; extract [amount, hour-of-day, day-of-week, transaction type (encoded), cumulative velocity] per step; SMOTE on the tabular features to upsample the minority class.

**3. Synthetic Payment Memo Text (generated)**
- **Modality:** Natural language (NLP)
- **Size:** ~50,000 payment descriptions
- **Why:** Real payment memo datasets are proprietary. Templates drawn from four categories (consulting invoice, wire transfer, charity donation, generic retail) create realistic variation while injecting suspicious patterns (round amounts, vague counterparties, rapid cycling language) into the illicit subset.
- **Preprocessing:** DistilBERT tokenizer; max length 64 tokens; zero-vector for missing memo fields.

---

### Model Considerations

**Three-branch late-fusion architecture.** Each modality is encoded independently; embeddings are concatenated and passed through a shared MLP fusion head.

**Branch 1 — Transaction graph: GraphSAGE (PyTorch Geometric)**
- 2-layer GraphSAGE (Hamilton et al., 2017); inductive neighborhood sampling.
- Output: 128-dimensional node embedding.
- Loss: focal loss (γ=2) for class imbalance.
- Why GraphSAGE over GCN: inductive — generalizes to unseen nodes at inference without retraining.

**Branch 2 — Payment memo text: DistilBERT (HuggingFace Transformers)**
- DistilBERT fine-tuned 3 epochs on synthetic memo text.
- [CLS] token → 64-dim linear projection.
- Optimizer: AdamW, lr=2e-5.

**Branch 3 — Behavioral time-series: Bidirectional LSTM**
- 2-layer BiLSTM on 30-day rolling windows of 5 features.
- Output: 64-dimensional hidden state.
- Trains in ~15 minutes on CPU.

**Fusion head**
- MLP: [256 → 128 → 64 → 1], ReLU, dropout 0.3, sigmoid output.
- Input: concatenated branch embeddings [128 + 64 + 64 = 256].
- Platt scaling post-training for calibrated probabilities.

**Baseline:** XGBoost on handcrafted tabular features (amount, velocity, graph degree, n-hop neighbor count). All three branches and the fusion model must beat it on AUC-PR.

---

### Open-Source Tools

**Third-party ML frameworks (required by course):**

| Tool | Role | Justification |
|---|---|---|
| **PyTorch Geometric (PyG)** | Graph encoder | Directly listed as a recommended third-party tool. Provides GraphSAGE, inductive mini-batch sampling, and PyG `Data` objects out-of-the-box. |
| **HuggingFace Transformers** | Text encoder | Also listed in course guidelines. DistilBERT checkpoint + tokenizer available via `from_pretrained`; compatible with PyTorch training loop. |

**MLOps stack:**

| Tool | Role |
|---|---|
| MLflow | Experiment tracking, model registry (staging → production promotion) |
| DVC | Data + artifact versioning against S3/Google Drive |
| GitHub Actions | CI/CD: lint → tests → schema check → AUC-PR gate → Docker build → deploy |
| Great Expectations | Data quality validation at ingestion (schema, null rates, class balance) |
| Evidently AI | Daily per-modality drift reports (PSI graph, KS time-series, cosine text) |
| AWS SageMaker | Containerized inference endpoint with canary rollout + autoscaling |
| SHAP | Per-prediction force plots for regulatory compliance |
| BentoML | Inference service packaging + Docker integration |

---

## 2. Code Organization & Setup

- [x] **GitHub Repository:** Initialized from SE 489 MLOps cookiecutter template (`src/` layout).
- [x] **Environment Setup:** Python 3.11+; install via `pip install -e ".[dev]"`.
- [x] **Dependency Management:** `pyproject.toml` (PEP 517) + pinned `requirements.txt`.
- [x] **Project Structure:** `src/`, `tests/`, `data/`, `models/`, `notebooks/`, `reports/`, `dockerfiles/`, `docs/`.
- [x] **Installation Documentation:** README setup section + `make install`.

**Key source modules:**
```
src/multimodal_anti_money_laundering/
├── config.py              # PROJECT_ROOT, paths, GraphSAGEConfig, FusionConfig
├── data/
│   ├── elliptic.py        # Elliptic loader → PyG Data; synthetic data fallback
│   ├── loaders.py         # Generic CSV loaders
│   └── make_dataset.py    # CLI: raw → processed
├── models/
│   ├── baseline.py        # XGBoost baseline
│   ├── graphsage.py       # GraphSAGE encoder (Week 2)
│   └── fusion.py          # Late-fusion MLP (Week 3)
├── evaluation/
│   └── metrics.py         # AUC-PR, P@R=0.8, FPR helpers
└── train_model.py         # Training CLI with MLflow logging
```

---

## 3. Version Control & Collaboration

- [x] **Branching strategy:** GitHub Flow — feature branches off `master`, PR required to merge.
- [x] **Team roles:**
  - Member A (Jaya): Elliptic graph construction, GraphSAGE, late-fusion MLP, SHAP, ablation
  - Member B (Anusooya): DistilBERT fine-tuning, BiLSTM, demo video, model card
  - Member C (Preshita): CI/CD YAML, unit tests, data schema gates, MLflow promotion
  - Member D (Rajani): Docker, SageMaker, Evidently, Grafana, Airflow retrain DAG
- [x] **Pre-commit hooks:** Ruff (lint + format) + mypy via `.pre-commit-config.yaml`.

---

## 4. Data Handling

- [x] **Data cleaning scripts:** `src/.../data/elliptic.py` — loads raw CSV files, maps labels (1→illicit, 2→licit, unknown→-1), filters unknown nodes from train/val/test masks.
- [x] **Normalization:** `StandardScaler` fit on training nodes only; applied to all splits.
- [x] **Class imbalance:** Focal loss (γ=2) for the GNN branch; SMOTE for tabular/BiLSTM branch; AUC-PR as primary metric (not ROC-AUC, which is misleading under 2% imbalance).
- [x] **Data splits:** 70% train / 15% val / 15% test on labeled nodes only; unknown nodes excluded from supervised loss.
- [x] **Data documentation:** See dataset table in Section 1.2 above and `data/README.md`.
- [x] **DVC setup:** `dvc init`; remote configured for data/raw and models/ directories.

**Download instructions:** See `data/README.md` for Kaggle CLI commands. A synthetic fallback in `elliptic.py` allows the pipeline to run without the real dataset for CI smoke tests.

---

## 5. Model Training

- [x] **Baseline model:** XGBoost trained on 166 Elliptic node features + graph degree (in/out) as additional features. Class weight = `scale_pos_weight` set to ratio of licit/illicit.
- [x] **Training environment:** Local CPU/GPU or Kaggle free GPU (P100). GraphSAGE trains in < 2 hours on Kaggle GPU; BiLSTM trains in ~15 minutes CPU.
- [x] **Hyperparameter configuration:** Documented in `src/.../config.py` (`GraphSAGEConfig`, `FusionConfig`); baseline params stored in MLflow run.
- [x] **Evaluation metrics:** AUC-PR (primary), Precision@Recall=0.8, FPR, F1. Computed by `src/.../evaluation/metrics.py`.
- [x] **Model persistence:** `models/baseline_xgb.json` (XGBoost native format); PyTorch checkpoints as `.pt` files.
- [x] **Training reproducibility:** `set_seed(42)` called before all training; seed logged to MLflow.
- [x] **Performance baseline:** See [REPORT.md](REPORT.md).

---

## 6. Documentation & Reporting

- [x] **README:** Team, objectives, Mermaid architecture diagram, setup, tech stack, structure, references.
- [x] **PHASE1.md:** This document.
- [x] **REPORT.md:** Baseline metrics, dataset stats, next steps.
- [x] **Code docstrings:** NumPy/Google style throughout `src/`.
- [x] **Type hints:** All public functions annotated; `mypy` configured in `pyproject.toml`.
- [x] **Ruff:** Line length 88, selects E/F/I/N/W/B/UP rules.
- [x] **Makefile targets:** `install`, `dev`, `data`, `train`, `test`, `lint`, `format`, `clean`, `docker_build`, `docker_run`, `docs`.
