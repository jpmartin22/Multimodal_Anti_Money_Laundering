# PHASE 3: Continuous Machine Learning (CML) & Deployment

> Every item below needs three pieces of evidence:
> 1. File/dir reference in the repo
> 2. Screenshot, live URL, or command output proving the working result
> 3. 2-4 sentence explanation of what was done and why
>
> Boxes are checked only when the repo implementation exists. Evidence boxes stay open until the screenshot/URL/explanation is attached for grading.

## Current Status Summary

Phase 3 implementation is mostly complete in the repository. The latest GitHub Actions status showed Cloud Run deployment passing, Docker builds passing for `aml-api` and `aml-hf`, and API smoke testing passing. The remaining grading work is evidence collection plus rerunning/fixing two deployment checks: `aml-graphsage` Docker build failed once due to a Docker Hub timeout, and Hugging Face Spaces deployment needed a metadata fix in `deploy/huggingface/README.md`.

---

## 1. Continuous Integration & Testing

- [x] **1.1 Unit Testing with pytest**
  - [x] Test scripts for data processing, model training, serving, and integration
  - Evidence:
    - [x] file/dir ref: `tests/test_model.py`, `tests/test_serving.py`, `tests/test_integration.py`, `tests/conftest.py`
    - [ ] screenshot of test run
    - [ ] explanation
  - Notes: The pytest suite covers model behavior, serving validation, and pipeline integration. Add a screenshot from either a successful local `pytest` run or a green GitHub Actions CI run.

- [x] **1.2 GitHub Actions CI Workflow**
  - [x] Workflow YAML(s) for tests, ruff, coverage, data gate, and evaluation gate
  - Evidence:
    - [x] file/dir ref: `.github/workflows/ci.yml`
    - [ ] screenshot of green run
    - [ ] explanation
  - Notes: The CI workflow installs lightweight dependencies, runs ruff checks, executes pytest with coverage, uploads coverage, runs the AUC-PR evaluation gate, and runs the data quality gate.

- [x] **1.3 Pre-commit Hooks**
  - [x] Pre-commit config and setup instructions
  - Evidence:
    - [x] file/dir ref: `.pre-commit-config.yaml`, `CONTRIBUTING.md`
    - [ ] screenshot of hook running
    - [ ] explanation
  - Notes: The hook configuration includes ruff, ruff-format, mypy, trailing whitespace, EOF fixer, and YAML checks.

---

## 2. Continuous Docker Building & CML

- [x] **2.1 Automated Docker Builds**
  - [x] GitHub Actions workflow builds and pushes Docker images on push
  - Evidence:
    - [x] file/dir ref: `.github/workflows/docker-build.yml`, `dockerfiles/Dockerfile`, `dockerfiles/Dockerfile.graphsage`, `dockerfiles/Dockerfile.hf`
    - [ ] screenshot of GHCR / registry image
    - [ ] explanation
  - Latest status: `aml-api`, `aml-hf`, and `Smoke-test aml-api` passed. `aml-graphsage` failed once during Buildx setup with `registry-1.docker.io/v2/: context deadline exceeded`, which appears to be a transient Docker Hub/network timeout. Rerun the failed `aml-graphsage` job and capture the successful workflow screenshot.

- [x] **2.2 Continuous Machine Learning (CML)**
  - [x] CML integration: PR triggers model report generation and posts metrics
  - Evidence:
    - [x] file/dir ref: `.github/workflows/cml.yml`, `scripts/cml_report.py`, `reports/cml_report.md`, `reports/figures/cml_auc_pr.png`
    - [ ] screenshot of CML PR comment
    - [ ] explanation
  - Notes: The CML workflow generates model metrics and posts the markdown report to PRs using the GitHub token.

---

## 3. Deployment on Google Cloud Platform (GCP)

- [x] **3.1 GCP Artifact Registry**
  - [x] Image pushed to Artifact Registry through Cloud Run deployment workflow
  - Evidence:
    - [x] file/dir ref: `.github/workflows/deploy-cloudrun.yml`
    - [ ] screenshot of Artifact Registry
    - [ ] explanation
  - Notes: The Cloud Run workflow builds and pushes `aml-api` to Artifact Registry before deployment. Capture the Artifact Registry image list from the GCP console.

- [x] **3.2 Custom Training Job on GCP**
  - [x] Vertex AI custom training job script and GCS model registry helper exist
  - Evidence:
    - [x] file/dir ref: `scripts/vertex_ai_training.py`, `scripts/gcs_model_registry.py`, `scripts/setup_gcp.sh`
    - [ ] screenshot of completed Vertex AI job
    - [ ] explanation
  - Notes: The repo includes the Vertex AI submission script, but grading still needs proof of an actual completed job and GCS bucket output if this item is claimed.

- [x] **3.3 FastAPI + GCP Cloud Functions**
  - [x] FastAPI service implemented
  - Evidence:
    - [x] file/dir ref: `src/multimodal_anti_money_laundering/serving/api.py`, `docs/api.md`
    - [ ] live endpoint URL + sample request/response
    - [ ] explanation
  - Notes: The service exposes `/health`, `/predict`, and `/metrics`. Current deployment evidence is strongest for Cloud Run; if Cloud Functions is required separately, add a Cloud Functions deployment artifact or clarify that Cloud Run is the selected GCP serving platform.

- [x] **3.4 Dockerize & Deploy with GCP Cloud Run**
  - [x] Container deployed to Cloud Run
  - Evidence:
    - [x] file/dir ref: `.github/workflows/deploy-cloudrun.yml`, `dockerfiles/Dockerfile`, `docs/screenshots/cloudrun_dashboard.png`
    - [ ] live service URL + sample request/response
    - [ ] explanation
  - Latest status: `Deploy to Cloud Run / Build, push & deploy to Cloud Run` succeeded in the latest run. Capture the successful workflow, the Cloud Run service page, and a `/health` or `/predict` response from the live URL.

---

## 4. Interactive UI

- [x] **4.1 Streamlit or Gradio app on Hugging Face Spaces**
  - [x] Hugging Face Space deployment workflow and Space metadata configured
  - Evidence:
    - [x] file/dir ref: `.github/workflows/deploy-huggingface.yml`, `deploy/huggingface/README.md`, `deploy/huggingface/app.py`, `Dockerfile`
    - [ ] Hugging Face Space URL + screenshot
    - [ ] explanation
  - Latest status: The Hugging Face deployment previously failed because `short_description` exceeded Hugging Face's 60-character metadata limit. It was fixed in `deploy/huggingface/README.md` by changing the value to `Multimodal AML risk scorer`. Rerun the deploy workflow, then capture the Space URL and screenshot.

---

## 5. End-to-End Demo Recording

- [ ] **5.1 Recording in main README**
  - [ ] 2-5 minute recording of deployed app and full use case
  - [ ] Narration or on-screen captions
  - [ ] Link or embed in main `README.md` near the top
  - Recording link/path for graders: _pending_
  - Notes: Record after Cloud Run and/or Hugging Face endpoints are stable. Recommended flow: show `README.md`, show this checklist, show successful GitHub Actions runs, open the deployed endpoint, submit a transaction, and explain the returned AML risk score.

---

## 6. Documentation, Repository Updates & Cleanup

- [x] **6.1 Comprehensive README**
  - [x] Phase links exist in main README
  - Evidence:
    - [x] file/dir ref: `README.md`, `REPORT.md`, `PHASE3.md`
    - [ ] screenshot of README rendered
    - [ ] explanation
  - Notes: Add the final demo recording link near the top of `README.md` once available.

- [x] **6.2 PHASE3.md**
  - [x] Checklist updated with repo references and current status
  - [ ] All evidence screenshots/URLs/explanations added
  - Notes: This file now tracks implementation separately from proof so unchecked evidence is visible before submission.

- [x] **6.3 GCP Resource Cleanup**
  - [x] Cleanup instructions documented
  - Evidence:
    - [x] file/dir ref: `CLEANUP.md`, `deploy/DEPLOYMENT.md`
    - [ ] screenshot of empty/cleaned GCP console
    - [ ] explanation
  - Notes: Capture cleanup evidence after demo recording so the deployed service is available long enough for screenshots and testing.

---

## Member-Specific Remaining Work

| Member | Area | Remaining Work |
|---|---|---|
| Neha | Demo and final README evidence | Record the final 2-5 minute demo after live endpoints are stable; add the recording link to `README.md`. |
| Jaya | Fusion / GraphSAGE / SHAP | Model deliverables are complete; add screenshots/evidence references if graders require proof beyond `REPORT.md`. |
| Preshita | CI/CD, tests, CML | Capture green CI, pytest/pre-commit, and CML PR comment screenshots with short explanations. |
| Rajani | Docker, Cloud Run, Hugging Face, cleanup | Capture Docker/GHCR evidence, Artifact Registry, Cloud Run URL and response, rerun Hugging Face deploy after metadata fix, and document GCP cleanup. |

---

> **Checklist:** This is a guideline, not a scoring sheet. Operational rigor and documented evidence are what get graded; checked implementation boxes without evidence may still be incomplete.
