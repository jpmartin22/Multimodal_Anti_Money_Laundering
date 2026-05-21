# Deployment Guide — AML Multimodal Scorer

Member D (Rajani) — Phase 3 §3 & §4

---

## Option A: HuggingFace Spaces (Docker SDK) — Recommended

Free, no cloud billing required. Ideal for course demo.

### Prerequisites

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate an access token: **Settings → Access Tokens → New token** (write permission)
3. Install the HF Hub client:
   ```bash
   source venv/bin/activate
   pip install huggingface_hub
   ```

### Deploy (first time)

```bash
source venv/bin/activate

# Option 1: interactive login
huggingface-cli login

# Option 2: token via env var
export HF_TOKEN=hf_your_token_here

# Push to HuggingFace Spaces
python deploy/push_to_spaces.py --username <your-hf-username>
```

HuggingFace will build the Docker image automatically (takes 2–5 minutes).

Your Space URL: `https://huggingface.co/spaces/<your-username>/aml-multimodal-scorer`

### Update after code changes

```bash
python deploy/push_to_spaces.py --username <your-hf-username> --update
```

### Test the deployed API

```bash
HF_SPACE="https://<your-username>-aml-multimodal-scorer.hf.space"

# Health check
curl $HF_SPACE/health

# Predict
curl -X POST $HF_SPACE/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-demo-001",
    "graph": {"node_features": [0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.1]},
    "memo_text": "consulting services invoice Q1 wire transfer",
    "time_series": {
      "window": [
        [100.0, 14.0, 2.0, 1.0, 500.0],
        [200.0, 9.0,  1.0, 0.0, 300.0],
        [150.0, 18.0, 3.0, 1.0, 450.0]
      ]
    }
  }'
```

Expected response:
```json
{
  "transaction_id": "tx-demo-001",
  "aml_risk_score": 0.5,
  "flagged": false,
  "threshold": 0.5
}
```

Interactive Swagger UI: `https://<your-username>-aml-multimodal-scorer.hf.space/docs`

---

## Option B: Google Cloud Run

### Prerequisites

- GCP account with billing enabled
- `gcloud` CLI installed: `brew install google-cloud-sdk`
- Docker installed

### Steps

```bash
# 1. Authenticate
gcloud auth login
gcloud config set project <your-gcp-project-id>

# 2. Enable APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# 3. Create Artifact Registry repo
gcloud artifacts repositories create aml-api \
  --repository-format=docker \
  --location=us-central1

# 4. Build and push Docker image
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/<project-id>/aml-api/aml-scorer:latest \
  --dockerfile dockerfiles/Dockerfile .

# 5. Deploy to Cloud Run
gcloud run deploy aml-scorer \
  --image us-central1-docker.pkg.dev/<project-id>/aml-api/aml-scorer:latest \
  --platform managed \
  --region us-central1 \
  --port 8000 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 3
```

### Test

```bash
SERVICE_URL=$(gcloud run services describe aml-scorer \
  --region us-central1 --format 'value(status.url)')
curl $SERVICE_URL/health
```

### Teardown (to avoid charges)

```bash
gcloud run services delete aml-scorer --region us-central1
gcloud artifacts repositories delete aml-api --location us-central1
```

---

## SageMaker (existing config)

See `deploy/canary_rollout.yaml` and `deploy/sagemaker_endpoint.py` for the
AWS SageMaker canary rollout configuration (90/10 traffic split).

```bash
# Deploy
python deploy/sagemaker_endpoint.py --config deploy/canary_rollout.yaml

# Promote canary to 100%
python deploy/sagemaker_endpoint.py --promote

# Teardown
python deploy/sagemaker_endpoint.py --delete
```

---

## Local Docker (for testing before deploy)

```bash
# Build HF variant
docker build -f dockerfiles/Dockerfile.hf -t aml-hf .

# Run on port 7860
docker run --rm -p 7860:7860 aml-hf

# Test
curl http://localhost:7860/health
curl http://localhost:7860/docs
```
