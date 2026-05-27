#!/usr/bin/env bash
# setup_gcp.sh — Bootstrap all GCP resources for the AML project.
#
# Run once per project. Idempotent — safe to re-run.
#
# Prerequisites:
#   gcloud CLI installed and authenticated:
#     gcloud auth login
#     gcloud auth application-default login
#
# Usage:
#   export GCP_PROJECT=your-project-id   (or edit PROJECT below)
#   export GCP_REGION=us-central1        (optional, default below)
#   bash scripts/setup_gcp.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT="${GCP_PROJECT:-}"
REGION="${GCP_REGION:-us-central1}"
AR_REPO="${AR_REPO:-aml-images}"
GCS_BUCKET="${GCS_MODEL_BUCKET:-aml-model-registry}"
SA_NAME="aml-ci-sa"
SA_DISPLAY="AML CI/CD Service Account"
SERVICE_NAME="aml-multimodal-scorer"

if [[ -z "$PROJECT" ]]; then
  echo "ERROR: Set GCP_PROJECT env var or edit the PROJECT variable in this script."
  exit 1
fi

log()  { echo; echo "▶ $*"; }
ok()   { echo "  ✓ $*"; }
info() { echo "  · $*"; }

# ── 0. Set active project ─────────────────────────────────────────────────────
log "Setting active project to $PROJECT"
gcloud config set project "$PROJECT"
ok "Active project: $PROJECT"

# ── 1. Enable required APIs ───────────────────────────────────────────────────
log "Enabling required APIs (this takes ~2 min on first run)"
APIS=(
  artifactregistry.googleapis.com
  run.googleapis.com
  aiplatform.googleapis.com
  storage.googleapis.com
  cloudbuild.googleapis.com
  iam.googleapis.com
  secretmanager.googleapis.com
  monitoring.googleapis.com
  logging.googleapis.com
)
gcloud services enable "${APIS[@]}" --project="$PROJECT"
ok "APIs enabled: ${APIS[*]}"

# ── 2. Service account ────────────────────────────────────────────────────────
log "Creating service account: $SA_NAME"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT" &>/dev/null; then
  info "Service account already exists — skipping creation"
else
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="$SA_DISPLAY" \
    --project="$PROJECT"
  ok "Created: $SA_EMAIL"
fi

# ── 3. IAM roles for the service account ─────────────────────────────────────
log "Granting IAM roles to $SA_EMAIL"
ROLES=(
  roles/artifactregistry.writer        # push Docker images
  roles/run.admin                      # deploy Cloud Run services
  roles/aiplatform.user                # submit Vertex AI jobs
  roles/storage.objectAdmin            # read/write GCS model bucket
  roles/iam.serviceAccountTokenCreator # allow Workload Identity impersonation
  roles/logging.logWriter              # write Cloud Logging entries
  roles/monitoring.metricWriter        # write Cloud Monitoring metrics
)
for ROLE in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE" \
    --quiet
  ok "  $ROLE"
done

# ── 4. Workload Identity Federation for GitHub Actions ───────────────────────
log "Setting up Workload Identity Federation for GitHub Actions"
WI_POOL="aml-github-pool"
WI_PROVIDER="aml-github-provider"
GITHUB_REPO="${GITHUB_REPO:-jpmartin22/Multimodal_Anti_Money_Laundering}"

# Create pool
if ! gcloud iam workload-identity-pools describe "$WI_POOL" \
    --location=global --project="$PROJECT" &>/dev/null; then
  gcloud iam workload-identity-pools create "$WI_POOL" \
    --location=global \
    --display-name="AML GitHub Actions Pool" \
    --project="$PROJECT"
  ok "Created WI pool: $WI_POOL"
else
  info "WI pool already exists: $WI_POOL"
fi

# Create OIDC provider
WI_POOL_ID=$(gcloud iam workload-identity-pools describe "$WI_POOL" \
  --location=global --project="$PROJECT" --format="value(name)")

if ! gcloud iam workload-identity-pools providers describe "$WI_PROVIDER" \
    --workload-identity-pool="$WI_POOL" \
    --location=global --project="$PROJECT" &>/dev/null; then
  gcloud iam workload-identity-pools providers create-oidc "$WI_PROVIDER" \
    --workload-identity-pool="$WI_POOL" \
    --location=global \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition="assertion.repository=='${GITHUB_REPO}'" \
    --project="$PROJECT"
  ok "Created OIDC provider: $WI_PROVIDER"
else
  info "OIDC provider already exists: $WI_PROVIDER"
fi

# Bind SA to WI provider
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${WI_POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  --project="$PROJECT" \
  --quiet
ok "WI binding: $GITHUB_REPO → $SA_EMAIL"

WI_PROVIDER_FULL="${WI_POOL_ID}/providers/${WI_PROVIDER}"
info "Add these to GitHub repo secrets / vars:"
info "  GCP_WORKLOAD_IDENTITY_PROVIDER = ${WI_PROVIDER_FULL}"
info "  GCP_SERVICE_ACCOUNT            = ${SA_EMAIL}"
info "  GCP_PROJECT_ID                 = ${PROJECT}"
info "  GCP_REGION                     = ${REGION}"

# ── 5. Artifact Registry repository ──────────────────────────────────────────
log "Creating Artifact Registry repository: $AR_REPO"
if ! gcloud artifacts repositories describe "$AR_REPO" \
    --location="$REGION" --project="$PROJECT" &>/dev/null; then
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="AML multimodal Docker images" \
    --project="$PROJECT"
  ok "Created: ${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}"
else
  info "Repository already exists: $AR_REPO"
fi

# ── 6. GCS bucket for model registry ─────────────────────────────────────────
log "Creating GCS model registry bucket: gs://$GCS_BUCKET"
if ! gsutil ls -b "gs://${GCS_BUCKET}" &>/dev/null; then
  gsutil mb -p "$PROJECT" -l "$REGION" -b on "gs://${GCS_BUCKET}"
  gsutil versioning set on "gs://${GCS_BUCKET}"
  ok "Created: gs://${GCS_BUCKET} (versioning enabled)"
else
  info "Bucket already exists: gs://${GCS_BUCKET}"
fi

# Grant the SA access to the bucket
gsutil iam ch "serviceAccount:${SA_EMAIL}:objectAdmin" "gs://${GCS_BUCKET}"
ok "IAM: $SA_EMAIL → objectAdmin on gs://${GCS_BUCKET}"

# ── 7. Cloud Run service (initial setup) ──────────────────────────────────────
log "Configuring Cloud Run defaults for service: $SERVICE_NAME"
gcloud config set run/region "$REGION"
gcloud config set run/platform managed

# Allow unauthenticated access for the demo endpoint
# (remove --allow-unauthenticated for production internal services)
info "Cloud Run service will be deployed by the deploy-cloudrun.yml workflow."
info "Service name: $SERVICE_NAME  |  Region: $REGION"
info "Access: unauthenticated (course demo — restrict for production)"

# ── 8. Summary ────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo " GCP Bootstrap Complete"
echo "════════════════════════════════════════════════════════════════"
echo " Project              : $PROJECT"
echo " Region               : $REGION"
echo " Service account      : $SA_EMAIL"
echo " Artifact Registry    : ${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}"
echo " Model bucket         : gs://${GCS_BUCKET}"
echo " Cloud Run service    : $SERVICE_NAME (deployed via GitHub Actions)"
echo ""
echo " Next steps:"
echo "   1. Copy the GCP_WORKLOAD_IDENTITY_PROVIDER and GCP_SERVICE_ACCOUNT"
echo "      values shown above into GitHub → Settings → Secrets and variables"
echo "   2. Push to master or trigger deploy-cloudrun.yml manually"
echo "   3. Run: python scripts/gcs_model_registry.py upload --model fusion"
echo "           --version 1.0.0 --local-dir models/fusion/"
echo "════════════════════════════════════════════════════════════════"
