# Resource Cleanup Guide

Instructions for tearing down all GCP resources created during Phase 3.
Run these commands when the project is complete to avoid ongoing charges.

> **Important:** These commands are irreversible. Verify you no longer need the resources before running.

---

## Quick Teardown (all resources)

```bash
export GCP_PROJECT=your-project-id
export GCP_REGION=us-central1

# Delete Cloud Run service
gcloud run services delete aml-multimodal-scorer \
  --region $GCP_REGION --project $GCP_PROJECT --quiet

# Delete Artifact Registry repository (and all images inside)
gcloud artifacts repositories delete aml-images \
  --location $GCP_REGION --project $GCP_PROJECT --quiet

# Delete GCS model bucket (and all objects)
gsutil rm -r gs://aml-model-registry

# Delete Vertex AI training jobs (does not affect already-finished jobs)
# List first, then delete individually if needed:
gcloud ai custom-jobs list --region $GCP_REGION --project $GCP_PROJECT

# Delete Workload Identity pool (GitHub Actions auth)
gcloud iam workload-identity-pools delete aml-github-pool \
  --location global --project $GCP_PROJECT --quiet

# Delete service account
gcloud iam service-accounts delete \
  aml-ci-sa@${GCP_PROJECT}.iam.gserviceaccount.com \
  --project $GCP_PROJECT --quiet
```

---

## Step-by-Step (selective teardown)

### 1. Cloud Run Service

```bash
# Delete the service
gcloud run services delete aml-multimodal-scorer \
  --region us-central1 --project $GCP_PROJECT

# Verify
gcloud run services list --region us-central1
```

Cost impact: Cloud Run is billed only on requests (free tier: 2M req/month). Deleting the service stops all future requests.

### 2. Artifact Registry

```bash
# List images before deleting
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/$GCP_PROJECT/aml-images

# Delete the entire repository (all images)
gcloud artifacts repositories delete aml-images \
  --location us-central1 --project $GCP_PROJECT

# Or delete a specific image only
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/$GCP_PROJECT/aml-images/aml-api:latest
```

Cost impact: ~$0.10/GB/month. Deleting all images removes storage charges.

### 3. Cloud Storage Bucket

```bash
# List bucket contents first
gsutil ls gs://aml-model-registry/

# Delete all objects, then bucket
gsutil rm -r gs://aml-model-registry

# Verify
gsutil ls | grep aml-model-registry
```

Cost impact: ~$0.02/GB/month for standard storage.

### 4. Vertex AI Training Jobs

Vertex AI training jobs that have already completed do not incur ongoing charges. Only delete if jobs are still running or queued.

```bash
# List jobs
gcloud ai custom-jobs list --region us-central1 --project $GCP_PROJECT

# Cancel a running job
gcloud ai custom-jobs cancel <JOB_ID> --region us-central1 --project $GCP_PROJECT
```

### 5. IAM — Service Account and Workload Identity

```bash
# Remove IAM bindings (optional — automatically removed when SA is deleted)
gcloud projects remove-iam-policy-binding $GCP_PROJECT \
  --member="serviceAccount:aml-ci-sa@${GCP_PROJECT}.iam.gserviceaccount.com" \
  --role="roles/run.admin" --quiet

# Delete WI federation pool (also deletes all providers inside)
gcloud iam workload-identity-pools delete aml-github-pool \
  --location global --project $GCP_PROJECT

# Delete service account
gcloud iam service-accounts delete \
  aml-ci-sa@${GCP_PROJECT}.iam.gserviceaccount.com \
  --project $GCP_PROJECT
```

### 6. Disable APIs (optional)

Disabling unused APIs has no cost impact but is good hygiene.

```bash
gcloud services disable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  --project $GCP_PROJECT
```

> Disabling APIs may affect other services in the same project. Check dependencies before disabling.

### 7. Delete the GCP Project entirely

If the project was created solely for this course:

```bash
gcloud projects delete $GCP_PROJECT
```

This deletes **all** resources in the project after a 30-day grace period. Resources can be recovered within the grace period via the GCP Console.

---

## Cost Monitoring

Before deleting, check your final bill:

```bash
# View current month's costs by service
gcloud billing budgets list --billing-account <BILLING_ACCOUNT_ID>
```

Or visit: **GCP Console → Billing → Cost table**

### Setting a budget alert

```bash
gcloud billing budgets create \
  --billing-account <BILLING_ACCOUNT_ID> \
  --display-name "AML Project Budget" \
  --budget-amount 10USD \
  --threshold-rule percent=0.5 \
  --threshold-rule percent=0.9 \
  --threshold-rule percent=1.0
```

This sends email alerts at 50%, 90%, and 100% of the $10 monthly budget.

---

## GitHub Secrets Cleanup

After tearing down GCP resources, remove the GitHub repository secrets to prevent stale credentials:

1. GitHub → repository → **Settings → Secrets and variables → Actions**
2. Delete:
   - `GCP_WORKLOAD_IDENTITY_PROVIDER`
   - `GCP_SERVICE_ACCOUNT`

---

## Local Cleanup

```bash
# Remove DVC cache (large, not needed after project)
rm -rf .dvc/cache

# Remove venv
rm -rf venv/

# Remove MLflow runs (if not needed)
rm -rf mlruns/

# Remove compiled Python bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
```
