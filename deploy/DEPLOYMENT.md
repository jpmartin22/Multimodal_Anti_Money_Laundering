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

---

## Automated Cloud Run Deployment (GitHub Actions)

The `deploy-cloudrun.yml` workflow handles builds and deploys automatically.

### One-time setup

```bash
# 1. Bootstrap all GCP resources (project, SA, Artifact Registry, GCS, WI Federation)
export GCP_PROJECT=your-project-id
bash scripts/setup_gcp.sh

# 2. Add the following to GitHub → Settings → Secrets and variables → Actions
#    Secrets:
#      GCP_WORKLOAD_IDENTITY_PROVIDER   (printed by setup_gcp.sh)
#      GCP_SERVICE_ACCOUNT              (printed by setup_gcp.sh)
#    Variables:
#      GCP_PROJECT_ID                   your GCP project ID
#      GCP_REGION                       us-central1  (or your preferred region)
#      AR_REPO                          aml-images
```

### Trigger deployment

```bash
# Automatic: push to master
git push origin master

# Manual (redeploy with a specific image tag):
#   GitHub → Actions → "Deploy to Cloud Run" → Run workflow → enter image_tag
```

### Verify after deploy

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe aml-multimodal-scorer \
  --region us-central1 --format='value(status.url)')

echo "Service URL: $SERVICE_URL"
curl "$SERVICE_URL/health"
```

---

## Environment Variables & Secrets

### Runtime environment variables (Cloud Run)

| Variable | Default | Description |
|---|---|---|
| `AML_THRESHOLD` | `0.5` | Decision threshold for `flagged` field |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`) |
| `MLFLOW_TRACKING_URI` | `file:///app/mlruns` | MLflow backend (set to remote URI in production) |
| `PORT` | `8000` | Port the uvicorn server listens on |

Set Cloud Run env vars:
```bash
gcloud run services update aml-multimodal-scorer \
  --region us-central1 \
  --set-env-vars AML_THRESHOLD=0.4,LOG_LEVEL=WARNING
```

### GCS model registry variables

| Variable | Default | Description |
|---|---|---|
| `GCS_MODEL_BUCKET` | `aml-model-registry` | GCS bucket for model artifacts |

### CI/CD secrets (GitHub Actions)

| Secret / Variable | Required | Description |
|---|---|---|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Secret | Full WI provider resource name |
| `GCP_SERVICE_ACCOUNT` | Secret | SA email for impersonation |
| `GCP_PROJECT_ID` | Variable | GCP project ID |
| `GCP_REGION` | Variable | Deployment region |
| `AR_REPO` | Variable | Artifact Registry repo name |

---

## Rollback Procedures

### Cloud Run — roll back to previous revision

```bash
# List revisions
gcloud run revisions list --service aml-multimodal-scorer --region us-central1

# Route 100% traffic to a previous revision
gcloud run services update-traffic aml-multimodal-scorer \
  --region us-central1 \
  --to-revisions <REVISION-NAME>=100

# Example: roll back to specific revision
gcloud run services update-traffic aml-multimodal-scorer \
  --region us-central1 \
  --to-revisions aml-multimodal-scorer-00005-abc=100
```

### Model registry — revert to previous model version

```bash
# Check available versions
python scripts/gcs_model_registry.py list --model fusion

# Promote an older version back to latest
python scripts/gcs_model_registry.py promote --model fusion --version 0.9.0

# Download and re-deploy the older version
python scripts/gcs_model_registry.py download \
  --model fusion --version 0.9.0 --dest models/fusion/
```

### Git — revert a bad commit on master

```bash
# Safe revert (creates a new commit, preserves history)
git revert <bad-commit-sha>
git push origin master
# The deploy-cloudrun.yml workflow will automatically redeploy
```

---

## Troubleshooting

### Cloud Run: container fails to start

```bash
# View logs
gcloud run services logs read aml-multimodal-scorer --region us-central1 --limit 50

# Common causes:
# - Missing model file: ensure models/fusion/fusion_mlp.pt is in the image or mounted
# - Port mismatch: Dockerfile CMD must use --port 8000 (set via PORT env var)
# - Memory exceeded: increase --memory to 4Gi for the fusion model
```

### GitHub Actions: WI authentication fails

```
Error: google-github-actions/auth failed with: the GitHub Actions workflow must be triggered by the master branch
```

Fix: Ensure `GCP_WORKLOAD_IDENTITY_PROVIDER` was created with the correct repository binding. Re-run `bash scripts/setup_gcp.sh` to recreate the attribute condition.

### Artifact Registry: push permission denied

```bash
# Re-configure Docker auth
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# Verify SA has Artifact Registry Writer role
gcloud projects get-iam-policy $GCP_PROJECT \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:aml-ci-sa@*"
```

### High latency on first request

Cloud Run scales to zero by default. The first request after idle triggers a **cold start** (~2–5 s). Options:

```bash
# Keep at least 1 instance warm (increases cost)
gcloud run services update aml-multimodal-scorer \
  --region us-central1 --min-instances 1
```

### AUC-PR gate fails in CI

```
CRITICAL CRASH: Model performance (0.XX) dropped below regulatory compliance target (0.80)
```

The gate reads `reports/metrics.json`. Ensure the file exists and `auc_pr` reflects the latest evaluation run. Check MLflow for the true value, then update `reports/metrics.json` accordingly.
