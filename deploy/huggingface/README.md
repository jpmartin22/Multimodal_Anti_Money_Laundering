---
title: AML Multimodal Scorer
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: GraphSAGE + DistilBERT + BiLSTM Anti-Money Laundering risk scorer
---

# Multimodal AML Detection API

Late-fusion AML risk scorer combining three modalities:

- **GraphSAGE** — transaction graph topology (166-dim node features)
- **DistilBERT** — payment memo text classification
- **BiLSTM** — 30-day behavioral time-series windows

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Score a transaction (returns 0–1 risk score) |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Interactive Swagger UI |

## Example Request

```bash
curl -X POST https://<your-space>.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-001",
    "graph": {"node_features": [0.1, 0.2, ...]},
    "memo_text": "consulting services invoice Q1",
    "time_series": {"window": [[100.0, 14.0, 2.0, 1.0, 500.0]]}
  }'
```

## Response

```json
{
  "transaction_id": "tx-001",
  "aml_risk_score": 0.5,
  "flagged": false,
  "threshold": 0.5
}
```

> **Note:** The fusion model returns a stub score of 0.5 until the Week-3 trained weights are loaded.
> The API, schema validation, and all endpoints are production-ready.
