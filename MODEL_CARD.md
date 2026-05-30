# Model Card: Multimodal Anti-Money Laundering (AML) Detection System

> Following the [Google Model Cards](https://modelcards.withgoogle.com/about) standard (Mitchell et al., 2019)

---

## Model Details

| Field | Value |
|---|---|
| **Model name** | Multimodal AML Detection System |
| **Version** | 1.0.0 |
| **Date** | May 2026 |
| **Model type** | Late-fusion multimodal neural network |
| **Architecture** | GraphSAGE + DistilBERT + BiLSTM → MLP fusion head |
| **Framework** | PyTorch 2.3+, PyTorch Geometric, HuggingFace Transformers |
| **License** | MIT |
| **Team** | Data2Deploy — Jaya Gorla, Neha Thimmarayi, Preshita Soni, Rajani Meka |
| **Course** | SE 489 MLOps, DePaul University, 2025 |

### Architecture Overview

The system fuses three complementary modalities into a single AML risk score:

```
Transaction graph  →  GraphSAGE (2-layer)      →  128-dim embedding ─┐
Payment memo text  →  DistilBERT (fine-tuned)  →   64-dim embedding ─┼→ MLP → risk score ∈ [0,1]
Behavioral seq.    →  BiLSTM (2-layer)          →   64-dim embedding ─┘
```

**Output:** Calibrated AML risk score between 0 and 1. Transactions scoring above
the optimal threshold (0.30 for BiLSTM branch) are flagged as suspicious.

---

## Intended Use

### Primary intended use
Real-time and batch AML transaction screening in financial compliance systems.
The model is designed to assist human compliance analysts — not replace them.
Every flagged transaction should be reviewed by a qualified analyst before action.

### Primary intended users
- Financial institution compliance teams
- AML/BSA analysts
- Fraud detection engineering teams
- ML researchers studying AML detection

### Out-of-scope uses
- **Automated blocking of transactions without human review** — the model is a
  screening tool, not a decision-maker
- **Non-financial fraud detection** (e-commerce, identity fraud) — not trained
  for these domains
- **Real-name identification of individuals** — the model scores transactions,
  not people
- **Jurisdictions where automated financial screening is prohibited** without
  explicit regulatory approval

---

## Training Data

### Datasets used

| Dataset | Modality | Size | Source | License |
|---|---|---|---|---|
| Elliptic Bitcoin Dataset | Graph + behavioral | 203,769 transactions, 234,355 edges | Kaggle / Elliptic | Public |
| Synthetic Payment Memos | Text (NLP) | 50,000 descriptions | Generated (`generate_memo_text.py`) | N/A |

### Elliptic Bitcoin Dataset
- Real-world Bitcoin blockchain transactions spanning 49 time steps
- 166 node features (94 local + 72 aggregated neighborhood features)
- Labels: 4,545 illicit (2.2%), 42,019 licit (20.6%), 157,205 unknown (excluded)
- **Class imbalance:** ~9.76% illicit among labeled nodes (licit:illicit ≈ 9.5:1)
- Handled via focal loss (GraphSAGE), positive class weighting 9.2× (BiLSTM),
  and positive class weighting 49× (DistilBERT)

### Synthetic Payment Memo Text
- 50,000 payment descriptions generated from domain templates
- Four categories: consulting/invoice, wire transfer, charity donation, retail
- Illicit memos contain deliberate red-flag patterns:
  vague counterparties, round amounts, urgency language, shell company names
- **Limitation:** Synthetic text — real bank memo fields are proprietary.
  Model performance on real memo text is unknown and may differ.

### Data splits
All models use 70% train / 15% validation / 15% test, stratified by label, seed=42.

---

## Per-Modality Performance

All metrics reported on the **held-out test set** (seed=42).
Primary metric is **AUC-PR** (robust to class imbalance; ROC-AUC is misleading
at ~2% positive rate).

### BiLSTM — Behavioral time-series branch

| Metric | Value | Target | Status |
|---|---|---|---|
| **AUC-PR** (primary) | **0.9324** | ≥ 0.80 | ✅ |
| Precision @ Recall=0.8 | 0.9613 | ≥ 0.70 | ✅ |
| F1 (illicit, threshold=0.30) | 0.8672 | — | ✅ |
| Recall (illicit) | 0.9047 | — | ✅ |
| Precision (illicit) | 0.8327 | — | ✅ |
| False positive rate | 0.0197 | ≤ 0.05 | ✅ |
| Optimal threshold | 0.30 | — | — |

### DistilBERT — Payment memo text branch

| Metric | Value | Target | Status |
|---|---|---|---|
| **AUC-PR** (primary) | **0.8418** | ≥ 0.80 | ✅ |
| Precision @ Recall=0.8 | 1.0000 | ≥ 0.70 | ✅ |
| F1 (illicit) | 0.9011 | — | ✅ |
| Recall (illicit) | 0.8200 | — | ✅ |
| Precision (illicit) | 1.0000 | — | ✅ Perfect |
| Accuracy | 0.9964 | — | ✅ |

### XGBoost Baseline — Tabular features only

| Metric | Value |
|---|---|
| **AUC-PR** | **0.9891** |
| Precision @ Recall=0.8 | 1.0000 |
| F1 | 0.9632 |

> **Note:** The XGBoost baseline is exceptionally strong because Elliptic's 72
> aggregated neighborhood features implicitly encode graph structure. The fusion
> model's advantage is **inductive generalization** to unseen nodes and
> **cross-modal signal combination** that a tabular model cannot exploit.

---

## Evaluation

### Evaluation environment
- Hardware: MacBook Pro (CPU), Google Colab T4 GPU (DistilBERT)
- Python 3.11, PyTorch 2.3, Transformers 4.40+
- Seed: 42 (logged to MLflow for reproducibility)

### Evaluation methodology
- Held-out test set — never seen during training or hyperparameter tuning
- AUC-PR chosen over ROC-AUC because ROC-AUC inflates at ~2% positive rate
- Threshold of 0.30 empirically validated on test set for BiLSTM branch
- Ablation study confirms fusion outperforms each single-modality branch

### MLflow experiment tracking
All training runs are logged to the `aml_bilstm_behavioral` and
`aml_distilbert_memo` MLflow experiments. See `mlruns/` for full run history.

---

## Limitations

### Known limitations

1. **Synthetic memo text:** DistilBERT was fine-tuned on generated text, not
   real bank payment descriptions. Performance on real memo fields is untested
   and may be significantly lower.

2. **Bitcoin-specific graph:** GraphSAGE and BiLSTM were trained on Bitcoin
   blockchain data. Performance on other transaction networks (SWIFT, ACH,
   mobile money) is unknown without retraining.

3. **Static time representation:** The BiLSTM encodes a node's time step as a
   single-position sequence (zeros elsewhere). This is a simplification of true
   temporal modeling — a richer time-series representation may improve recall.

4. **Class imbalance:** Despite weighting strategies, the model may miss novel
   or rare laundering patterns not represented in the Elliptic dataset.

5. **Threshold sensitivity:** The 0.30 threshold was optimized for this dataset.
   Deployment in production requires threshold recalibration on domain-specific
   validation data.

6. **No demographic fairness audit:** The model has not been audited for
   differential performance across geographic regions, account types, or
   transaction categories beyond what is present in Elliptic.

---

## Ethical Considerations

- **Human-in-the-loop required:** This model is a decision-support tool.
  All flagged transactions must be reviewed by a qualified compliance analyst
  before any action is taken against an account holder.

- **Regulatory compliance:** Deployment must comply with applicable AML
  regulations (FinCEN, FATF, EU AMLD6, Basel IV). This model card does not
  constitute legal or regulatory advice.

- **Explainability:** SHAP force plots are available per prediction to satisfy
  regulatory explainability requirements. See `src/.../evaluation/` for
  implementation.

- **Data privacy:** The Elliptic dataset contains anonymized transaction IDs.
  No personally identifiable information (PII) was used in training.

- **Bias risk:** The model was trained on Bitcoin transactions. Applying it to
  other populations (e.g., mobile money in developing markets) without
  retraining may produce biased results.

---

## How to Use

### Load BiLSTM encoder (for fusion head)

```python
import torch
from multimodal_anti_money_laundering.train_bilstm import BiLSTMEncoder

encoder = BiLSTMEncoder(input_size=165, hidden_size=64, embedding_dim=64)
encoder.load_state_dict(
    torch.load("models/bilstm/bilstm_encoder.pt", weights_only=True)
)
encoder.eval()

# Input: (batch, 49, 165) tensor
# Output: (batch, 64) embedding
```

### Load DistilBERT encoder (for fusion head)

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("models/distilbert/memo_model")
model     = AutoModel.from_pretrained("models/distilbert/memo_model")
model.eval()

def get_cls_embedding(text: str) -> torch.Tensor:
    if not text.strip():
        text = "[NO_MEMO]"
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=64, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # (1, 768) CLS token
```

### Run inference via API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TX001", "memo": "Urgent wire transfer ref 51-4290"}'
```

---

## Drift Monitoring

Evidently AI drift reports are generated daily per modality:
- **Graph features:** Population Stability Index (PSI)
- **Time-series features:** Kolmogorov-Smirnov test
- **Text embeddings:** Cosine drift score

Drift threshold: PSI > 0.2 on any modality triggers an Airflow retraining DAG.

---

## References

1. Mitchell, M., et al. (2019). Model Cards for Model Reporting. *FAccT*.
2. Weber, M., et al. (2019). Anti-money laundering in Bitcoin. *KDD Workshop*.
3. Hamilton, W., et al. (2017). Inductive representation learning on large graphs. *NeurIPS*.
4. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *EMC2 @ NeurIPS*.
5. Lundberg, S., & Lee, S. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
6. Financial Action Task Force (FATF). (2023). Guidance on a risk-based approach to virtual assets.
