# API Reference — AML Multimodal Scoring API

FastAPI service serving the late-fusion AML risk scorer.

**Base URL (local):** `http://localhost:8000`
**Base URL (HuggingFace Spaces):** `https://<username>-aml-multimodal-scorer.hf.space`
**Base URL (Cloud Run):** `https://aml-multimodal-scorer-<hash>-<region>.a.run.app`

Interactive docs: `<base-url>/docs` (Swagger UI) · `<base-url>/redoc`

---

## Endpoints

### `GET /health`

Liveness probe. Returns service status and whether the fusion model is loaded.

**Response `200 OK`**

```json
{
  "status": "ok",
  "model": "stub"
}
```

| Field | Type | Values |
|---|---|---|
| `status` | string | always `"ok"` |
| `model` | string | `"stub"` (no model loaded) or `"loaded"` (fusion model active) |

**curl example**

```bash
curl http://localhost:8000/health
```

**Python example**

```python
import httpx
r = httpx.get("http://localhost:8000/health")
print(r.json())  # {"status": "ok", "model": "stub"}
```

---

### `GET /ping`

SageMaker-compatible liveness endpoint. Identical behaviour to `/health`.

**Response `200 OK`**

```json
{"status": "ok"}
```

---

### `GET /metrics`

Prometheus metrics scrape endpoint. Returns plain-text metrics in the Prometheus exposition format.

**Response `200 OK`** (Content-Type: `text/plain; version=0.0.4`)

```
# HELP aml_predictions_total Total predictions by flagged status
# TYPE aml_predictions_total counter
aml_predictions_total{flagged="False"} 42.0
aml_predictions_total{flagged="True"} 3.0
# HELP aml_prediction_latency_seconds Prediction latency in seconds
# TYPE aml_prediction_latency_seconds histogram
aml_prediction_latency_seconds_bucket{le="0.005"} 38.0
...
```

**curl example**

```bash
curl http://localhost:8000/metrics
```

---

### `POST /predict`

Score a transaction for AML risk. Main inference endpoint.

**Request body** (`application/json`)

```json
{
  "transaction_id": "tx-001",
  "graph": {
    "node_features": [0.1, 0.2, 0.3, ...]
  },
  "memo_text": "consulting services invoice Q1",
  "time_series": {
    "window": [
      [1000.0, 14.0, 2.0, 0.0, 0.01],
      [1200.0,  9.0, 1.0, 0.0, 0.012]
    ]
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `transaction_id` | string | yes | Unique identifier for this transaction |
| `graph.node_features` | float[] | yes | 166-dimensional Elliptic node feature vector (normalised) |
| `memo_text` | string | yes | Raw payment memo / description text |
| `time_series.window` | float[][] | yes | ≥1 rows of `[amount, hour_of_day, day_of_week, tx_type, cumulative_velocity]` (30-day rolling window) |

**Response `200 OK`**

```json
{
  "transaction_id": "tx-001",
  "aml_risk_score": 0.1243,
  "flagged": false,
  "threshold": 0.5
}
```

| Field | Type | Description |
|---|---|---|
| `transaction_id` | string | Echoed from request |
| `aml_risk_score` | float [0, 1] | Calibrated AML risk probability |
| `flagged` | bool | `true` when `aml_risk_score >= threshold` |
| `threshold` | float | Decision threshold applied (default `0.5`, set via `AML_THRESHOLD` env var) |

**curl example**

```bash
NODE_FEATURES=$(python3 -c "import json; print(json.dumps([0.1]*166))")

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"transaction_id\": \"tx-001\",
    \"graph\": {\"node_features\": ${NODE_FEATURES}},
    \"memo_text\": \"consulting services invoice Q1\",
    \"time_series\": {\"window\": [[1000.0, 14.0, 2.0, 0.0, 0.01]]}
  }"
```

**Python example**

```python
import httpx

payload = {
    "transaction_id": "tx-001",
    "graph": {"node_features": [0.1] * 166},
    "memo_text": "consulting services invoice Q1",
    "time_series": {"window": [[1000.0, 14.0, 2.0, 0.0, 0.01]]},
}

r = httpx.post("http://localhost:8000/predict", json=payload)
result = r.json()
print(f"Risk score: {result['aml_risk_score']:.4f}  Flagged: {result['flagged']}")
```

---

### `POST /invocations`

SageMaker-compatible inference endpoint. Identical behaviour to `POST /predict`.

**Request / Response:** same schemas as `/predict`.

---

## Request Schema Details

### `GraphInput`

```json
{
  "node_features": [float, float, ...]
}
```

- `node_features`: exactly 166 floats. Corresponds to the Elliptic dataset node feature vector. For non-Elliptic deployments, use normalised feature values in [0, 1].

### `TimeSeriesInput`

```json
{
  "window": [
    [amount, hour_of_day, day_of_week, tx_type, cumulative_velocity],
    ...
  ]
}
```

| Column | Index | Range | Description |
|---|---|---|---|
| `amount` | 0 | ≥ 0 | Transaction amount (USD or normalised) |
| `hour_of_day` | 1 | 0–23 | Hour the transaction was initiated |
| `day_of_week` | 2 | 0–6 | Day of week (0 = Monday) |
| `tx_type` | 3 | 0 or 1 | 0 = debit, 1 = credit |
| `cumulative_velocity` | 4 | ≥ 0 | Running total of transaction amounts in the window |

Minimum 1 row, maximum 30 rows (30-day window). Rows are time-ordered oldest → newest.

---

## Error Responses

### `422 Unprocessable Entity`

Returned when the request body fails schema validation.

**Triggers:**
- `node_features` length ≠ 166
- `time_series.window` is empty (`[]`)
- Missing required fields (`transaction_id`, `memo_text`, `graph`, `time_series`)
- Wrong types (e.g. string where float expected)

**Example response**

```json
{
  "detail": [
    {
      "type": "too_short",
      "loc": ["body", "graph", "node_features"],
      "msg": "List should have at least 166 items after validation, not 10",
      "input": [0.1, 0.2, ...],
      "ctx": {"field_type": "List", "min_length": 166, "actual_length": 10}
    }
  ]
}
```

### `500 Internal Server Error`

Returned when an unexpected exception occurs during inference. The response body contains a FastAPI error detail string. Check `/metrics` for the `aml_errors_total` counter and application logs for the traceback.

---

## Python Client Example (full flow)

```python
"""Score multiple transactions and collect flagged ones."""
import httpx

BASE_URL = "http://localhost:8000"

transactions = [
    {
        "id": "tx-001",
        "memo": "payroll disbursement Q2",
        "amount": 5000.0,
    },
    {
        "id": "tx-002",
        "memo": "urgent offshore transfer anonymous",
        "amount": 95000.0,
    },
]


def build_payload(tx: dict) -> dict:
    return {
        "transaction_id": tx["id"],
        "graph": {"node_features": [0.1] * 166},
        "memo_text": tx["memo"],
        "time_series": {
            "window": [[tx["amount"], 14.0, 2.0, 0.0, tx["amount"] / 100_000]]
        },
    }


with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
    # Verify service is up
    health = client.get("/health").json()
    print(f"Service: {health['status']}  Model: {health['model']}")

    flagged = []
    for tx in transactions:
        r = client.post("/predict", json=build_payload(tx))
        r.raise_for_status()
        result = r.json()
        print(
            f"{tx['id']}: score={result['aml_risk_score']:.4f}  "
            f"flagged={result['flagged']}"
        )
        if result["flagged"]:
            flagged.append(result)

print(f"\nFlagged transactions: {len(flagged)}")
```

---

## Python Package API

The package is importable as `multimodal_anti_money_laundering` after `pip install -e .`.

### Core modules

```python
from multimodal_anti_money_laundering.config import PROJECT_ROOT, DATA_DIR, MODELS_DIR
from multimodal_anti_money_laundering.logging_config import get_logger
from multimodal_anti_money_laundering.features import build_features
from multimodal_anti_money_laundering.evaluation.metrics import compute_aml_metrics
from multimodal_anti_money_laundering.utils import set_seed, save_json, load_json
```

### Evaluation

```python
import numpy as np
from multimodal_anti_money_laundering.evaluation.metrics import compute_aml_metrics

y_true = np.array([0, 0, 1, 1, 0, 1])
y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

metrics = compute_aml_metrics(y_true, y_score)
# {
#   "auc_pr": 1.0,
#   "precision_at_r80": 1.0,
#   "fpr_at_r80": 0.0,
#   "f1": 1.0,
#   "roc_auc": 1.0
# }
```

### Training CLIs

```bash
python -m multimodal_anti_money_laundering.train_graphsage --help
python -m multimodal_anti_money_laundering.train_distilbert --help
python -m multimodal_anti_money_laundering.train_bilstm --help
python -m multimodal_anti_money_laundering.train_fusion --help
```

### AUC-PR evaluation gate

```bash
python -m multimodal_anti_money_laundering.evaluation.eval_gate \
    --metrics reports/metrics.json
# Exits 0 if auc_pr >= 0.80, exits 1 otherwise
```
