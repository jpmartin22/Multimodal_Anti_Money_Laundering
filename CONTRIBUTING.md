# Contributing Guide

## Development Setup

```bash
git clone https://github.com/jpmartin22/Multimodal_Anti_Money_Laundering.git
cd Multimodal_Anti_Money_Laundering

python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"      # installs pytest, ruff, mypy, pre-commit
pre-commit install            # installs Git hooks
```

---

## Branching Convention

| Branch prefix | Purpose |
|---|---|
| `Task_<member>-<description>` | Feature or task branch |
| `fix/<description>` | Bug fix |
| `docs/<description>` | Documentation-only change |

Branch from `master`. Open a PR back to `master` when ready.

---

## Before Opening a PR

### 1. Run tests locally

```bash
# Full test suite
pytest tests/ -v

# With coverage (aim for ≥ 80% on lightweight modules)
pytest tests/ --cov=multimodal_anti_money_laundering --cov-report=term-missing
```

All tests must pass. The CI matrix runs Python 3.10 and 3.11 — avoid syntax or library calls that don't work on 3.10.

### 2. Run lint and type checks

```bash
ruff check .
ruff format --check .
mypy src/multimodal_anti_money_laundering --ignore-missing-imports
```

Or run everything at once:

```bash
make lint
```

Fix any errors before pushing. The CI will block merge on lint or test failures.

### 3. Pre-commit hooks

If you installed pre-commit (`pre-commit install`), the following run automatically on `git commit`:

- `ruff` — lint + auto-fix
- `ruff-format` — formatting
- `mypy` — type checking
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`

To run manually across all files:

```bash
pre-commit run --all-files
```

---

## CI Requirements

Every PR triggers three GitHub Actions workflows. All must pass before merge:

| Workflow | What it checks | Failure means |
|---|---|---|
| `ci.yml` | Ruff lint, mypy, pytest (Py 3.10 + 3.11), AUC-PR gate | Tests broken or lint errors — fix before merging |
| `docker-build.yml` | Docker image builds successfully, `/health` responds | Dockerfile or dependency issue — investigate build logs |
| `cml.yml` | CML report generated (non-blocking) | Metrics script error — review `scripts/cml_report.py` |

The `deploy-cloudrun.yml` workflow only runs after `docker-build.yml` succeeds on `master` or a version tag. It is **not** required for PR approval.

---

## Testing Requirements for PRs

- **New features or bug fixes** must include at least one test covering the change.
- **Tests must not require `torch`, `transformers`, or DVC-tracked data.** Use lightweight fixtures or mocks. Heavy-dep code belongs in a `continue-on-error` step or a separate DVC stage, not in `tests/`.
- **Do not use `unittest.mock.patch` to bypass the real implementation** for core logic tests — use it only to mock I/O, external services, or optional heavy imports (see `tests/test_integration.py::TestSetSeed` for an example).
- Integration tests live in `tests/test_integration.py`. Unit tests for a specific module go in a dedicated file (e.g. `tests/test_evaluation.py`).

### Coverage targets

| Module group | Target |
|---|---|
| `evaluation/`, `features/`, `utils/` | ≥ 95% |
| `serving/` (API + schemas) | ≥ 80% |
| Training scripts (`train_*.py`) | Not required in CI (requires GPU + data) |

---

## Deployment Process

### Automatic deployment (recommended)

Push to `master` (or merge a PR):

1. `ci.yml` runs lint, tests, and AUC-PR gate.
2. `docker-build.yml` builds and pushes images to GHCR.
3. `deploy-cloudrun.yml` deploys to Cloud Run (requires GCP secrets — see [deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md)).

### Manual deploy to HuggingFace Spaces

```bash
export HF_TOKEN=hf_your_token
python deploy/push_to_spaces.py --username <hf-username> --update
```

### Adding a new workflow

1. Create `.github/workflows/<name>.yml`.
2. Follow the existing patterns: `actions/checkout@v4`, `setup-python@v4`.
3. Keep CI installs slim — no `torch`/`transformers` unless the job truly needs them.
4. Document the workflow in the CI table in `README.md → Testing → CI workflows`.

---

## Code Style

- **Python version:** `>=3.10` (see `pyproject.toml`)
- **Formatter:** `ruff format` (Black-compatible, 88-char lines)
- **Linter:** `ruff check` — rules: `E, F, I, N, W, B, UP`
- **Type hints:** encouraged on all public functions; required on new modules
- **Comments:** only for non-obvious invariants or workarounds — not for what the code does
- **No emojis** in source code or commit messages

---

## Commit Messages

Follow the existing style:

```
<type>(<scope>): short imperative summary

- bullet point detail (optional)
- another detail
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`

Examples:
```
feat(phase3-s1): add integration tests for eval_gate and metrics pipeline
fix(fusion): correct Platt calibration crash on zero-variance predictions
docs(api): add HTTP endpoint reference with curl and Python examples
```

---

## Questions?

Open an issue on [GitHub](https://github.com/jpmartin22/Multimodal_Anti_Money_Laundering/issues) or contact the team via DePaul course channels.
