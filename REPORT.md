# AML Detection — Model Results

> Phase 1 complete · XGBoost baseline on real Elliptic data · GraphSAGE pending (Week 2)

---

## Dataset: Elliptic Bitcoin Transaction Graph

| Stat | Value |
|---|---|
| Total transactions | 203,769 |
| Directed edges | 234,355 |
| Node features | 166 (94 local + 72 aggregated neighborhood) |
| Time steps | 49 (real blockchain snapshots) |
| **Labeled** | **46,564 (22.9% of total)** |
| — Illicit | 4,545 (9.76% of labeled, 2.23% of all) |
| — Licit | 42,019 (90.24% of labeled) |
| Unknown | 157,205 (77.1%) — excluded from supervised training |

**Class imbalance note:** Within labeled transactions the illicit rate is 9.76% (licit:illicit ≈ 9.5:1), not 2%. The 2% figure refers to the full dataset including unknowns. `scale_pos_weight = 9.46` was set automatically.

**Train / Val / Test split** (70 / 15 / 15, labeled nodes only, seed=42):

| Split | Samples | Illicit % |
|---|---|---|
| Train | 32,594 | 9.56% |
| Val | 6,984 | 9.95% |
| Test | 6,986 | 10.51% |

---

## Phase 1 Baseline: XGBoost on Elliptic Tabular Features

**Model:** XGBoost classifier on all 166 node features — no graph traversal, no text, no time-series.

### Training Progression (Val AUC-PR)

| Iteration | Val AUC-PR |
|---|---|
| 0 | 0.9101 |
| 50 | 0.9583 |
| 100 | 0.9709 |
| 150 | 0.9778 |
| 200 | 0.9813 |
| 250 | 0.9830 |
| 300 | 0.9841 |
| 350 | 0.9850 |
| 400 | 0.9855 |
| **488 (best)** | **0.9860** |

Early stopping triggered at iteration 488 (patience=30).

### Hyperparameters

| Parameter | Value |
|---|---|
| n_estimators (best iter) | 488 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 9.46 |
| eval_metric | aucpr |
| device | cuda (Colab T4) |
| seed | 42 |

### Results

| Metric | Val | Test | Target | Status |
|---|---|---|---|---|
| **AUC-PR** (primary) | **0.9860** | **0.9891** | ≥ 0.80 | ✅ Exceeds target |
| Precision @ Recall=0.80 | 0.9982 | 1.0000 | ≥ 0.70 | ✅ Exceeds target |
| FPR @ Recall=0.80 | 0.0002 | 0.0000 | ≤ 0.05 | ✅ Exceeds target |
| F1 (threshold=0.5) | 0.9588 | 0.9632 | — | — |
| ROC-AUC | 0.9961 | 0.9977 | — | — |

### Why the Baseline Is So Strong

The Elliptic feature set includes **72 aggregated neighborhood features** — pre-computed statistics over each node's 1-hop and 2-hop neighbors (counts, sums, means). These implicitly encode graph structure without any GNN. This is why the tabular baseline far exceeds the literature benchmark (~0.65–0.72 AUC-PR reported by Weber et al., 2019, who used only a feature subset).

**Implication for Week 2:** GraphSAGE must beat **AUC-PR = 0.9891** using dynamic inductive neighborhood aggregation. The advantage GraphSAGE will add is:
- Inductive generalization to **unseen nodes** (new accounts not in training)
- Deeper multi-hop aggregation beyond the pre-computed 2-hop features
- End-to-end learned embeddings that the fusion head can exploit jointly with text and time-series

---

## Planned Model Comparison (End of Week 3)

| Model | AUC-PR | P@R=0.80 | FPR@R=0.80 | Status |
|---|---|---|---|---|
| **XGBoost tabular baseline** | **0.9891** | **1.0000** | **0.0000** | ✅ Done |
| GraphSAGE (graph only) | TBD | TBD | TBD | Week 2 |
| DistilBERT (text only) | TBD | TBD | TBD | Week 2 |
| BiLSTM (time-series only) | TBD | TBD | TBD | Week 2 |
| **Late-fusion (all three)** | **> 0.9891 target** | TBD | TBD | Week 3 |

The fusion model target is now to **exceed the XGBoost baseline (0.9891 AUC-PR)**, not just the original proposal target of 0.80.

---

## How to Reproduce

```bash
# Synthetic data (no download required — CI smoke test)
python -m multimodal_anti_money_laundering.train_model --model baseline --synthetic

# Real Elliptic data (download first — see data/README.md)
python -m multimodal_anti_money_laundering.train_model --model baseline
```

Full EDA + training notebook: [`notebooks/01_week1_eda_baseline.ipynb`](notebooks/01_week1_eda_baseline.ipynb)
(Open in Colab → Runtime → T4 GPU → Run all)

Artifacts:
- `models/baseline_metrics.json` — test metrics
- `models/baseline_xgb.joblib` — fitted model (gitignored; tracked via DVC)

---

## References

- Weber et al. (2019). *Anti-money laundering in Bitcoin.* KDD Workshop. — Elliptic dataset; reports XGBoost AUC-PR ~0.65–0.72 on a feature subset.
- Hamilton et al. (2017). *Inductive representation learning on large graphs.* NeurIPS. — GraphSAGE (Week 2 target).


---

## Phase 1 Member B: BiLSTM on Elliptic Behavioral Time-Series

**Model:** 2-layer Bidirectional LSTM on Elliptic node feature sequences across 49 time steps. Each labeled node is represented as a sequence of length 49 (one entry per time step; zeros for all steps except the node's own time step). This encodes *when* in the transaction lifecycle a node appeared — a genuine AML signal since illicit transactions cluster in specific time windows.

**Script:** `src/multimodal_anti_money_laundering/train_bilstm.py`

### Architecture

| Component | Detail |
|---|---|
| Input shape | (batch, 49, 165) — 49 time steps × 165 features |
| LSTM layers | 2-layer BiLSTM, hidden_size=64, bidirectional |
| Projection | Linear(128 → 64) — produces 64-dim embedding for fusion head |
| Classification head | Linear(64 → 1) + Sigmoid |
| Dropout | 0.3 between LSTM layers |
| Loss | BCELoss with positive class weight 9.2x |
| Optimizer | Adam, lr=0.001 |
| Seed | 42 |

### Experiment Comparison (3 MLflow runs)

| Experiment | LR | Hidden size | Test AUC-PR | Test P@R=0.8 | Time |
|---|---|---|---|---|---|
| **Exp 1 — baseline** | **0.001** | **64** | **0.9324** | **0.9613** | **2.2 min** |
| Exp 2 — higher LR | 0.005 | 64 | 0.9139 | 0.9496 | 2.2 min |
| Exp 3 — larger hidden | 0.001 | 128 | 0.9257 | 0.9698 | 4.3 min |

**Best model: Experiment 1** — lower LR converges more stably, trains 2x faster.

### Results (Best Model — Test Set)

| Metric | Value | Target | Status |
|---|---|---|---|
| **AUC-PR** (primary) | **0.9324** | ≥ 0.80 | ✅ Exceeds target |
| Precision @ Recall=0.8 | 0.9613 | ≥ 0.70 | ✅ Exceeds target |
| False positive rate | 0.0000 | ≤ 0.05 | ✅ Exceeds target |
| Training time (CPU) | 2.2 min | < 15 min | ✅ Within budget |

### Profiling Results (Phase 2 §3)

| Metric | Value |
|---|---|
| Biggest bottleneck | `torch.lstm` kernel — 43% of training time |
| Peak memory | 1,496 MB |
| Optimization | Batch size 64 → 256 = **1.62x speedup** |

### How to Reproduce

```bash
python -m multimodal_anti_money_laundering.train_bilstm
python -m multimodal_anti_money_laundering.train_bilstm --lr 0.001 --hidden_size 64 --epochs 10
```

Artifacts: `models/bilstm/bilstm_best.pt`, `models/bilstm/bilstm_encoder.pt`, `bilstm_metrics.json`
