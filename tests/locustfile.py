"""
Locust load test for the AML Multimodal Scoring API.

Run locally (against a running container):
    locust -f tests/locustfile.py --host=http://localhost:8000 \
           --users=50 --spawn-rate=5 --run-time=2m --headless

Run against Cloud Run:
    export AML_API_URL=https://<service>-<hash>-<region>.a.run.app
    locust -f tests/locustfile.py --host=$AML_API_URL \
           --users=100 --spawn-rate=10 --run-time=5m --headless \
           --html=reports/load_test_report.html

Install:
    pip install locust
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# ---------------------------------------------------------------------------
# Fixtures — pre-built payloads to avoid per-request generation overhead
# ---------------------------------------------------------------------------

_NODE_FEATURES_LOW = [0.05] * 166
_NODE_FEATURES_HIGH = [0.9] * 166

_TS_NORMAL = {"window": [[500.0, 10.0, 2.0, 0.0, 0.005]] * 5}
_TS_SUSPICIOUS = {"window": [[95000.0, 2.0, 6.0, 1.0, 0.95]] * 5}

_MEMO_NORMAL = [
    "payroll disbursement Q2",
    "rent payment March",
    "utility bill electric",
    "coffee shop receipt",
    "grocery store purchase",
    "invoice #4422 consulting services",
]

_MEMO_SUSPICIOUS = [
    "urgent transfer offshore account",
    "cash equivalent anonymous beneficiary",
    "shell company consulting fee",
    "immediate wire transfer no questions",
]


def _make_payload(high_risk: bool = False) -> dict:
    return {
        "transaction_id": f"load-{random.randint(100_000, 999_999)}",
        "graph": {
            "node_features": _NODE_FEATURES_HIGH if high_risk else _NODE_FEATURES_LOW
        },
        "memo_text": (
            random.choice(_MEMO_SUSPICIOUS)
            if high_risk
            else random.choice(_MEMO_NORMAL)
        ),
        "time_series": _TS_SUSPICIOUS if high_risk else _TS_NORMAL,
    }


# ---------------------------------------------------------------------------
# User behaviours
# ---------------------------------------------------------------------------


class AmlApiUser(HttpUser):
    """Simulates a realistic mix of traffic to the AML scoring API."""

    wait_time = between(0.05, 0.5)  # 100–2000 ms think-time between tasks

    # ── Read-only endpoints (high frequency) ─────────────────────────────────

    @task(10)
    def health_check(self) -> None:
        """Liveness probe — cheap, should always be fast."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                resp.success()
            else:
                resp.failure(f"Unexpected health response: {resp.text}")

    @task(3)
    def metrics_endpoint(self) -> None:
        """Prometheus scrape — verify metrics are exposed."""
        with self.client.get("/metrics", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Metrics unavailable: {resp.status_code}")

    # ── Scoring endpoint — low-risk transactions (majority) ─────────────────

    @task(50)
    def predict_normal(self) -> None:
        """Score a routine low-risk transaction."""
        payload = _make_payload(high_risk=False)
        with self.client.post(
            "/predict",
            json=payload,
            name="/predict [normal]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if "aml_risk_score" in body and 0.0 <= body["aml_risk_score"] <= 1.0:
                    resp.success()
                else:
                    resp.failure(f"Bad response body: {body}")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    # ── Scoring endpoint — high-risk transactions (realistic minority) ───────

    @task(10)
    def predict_suspicious(self) -> None:
        """Score a suspicious high-risk transaction."""
        payload = _make_payload(high_risk=True)
        with self.client.post(
            "/predict",
            json=payload,
            name="/predict [suspicious]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if "flagged" in body:
                    resp.success()
                else:
                    resp.failure(f"Missing 'flagged' key: {body}")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    # ── Validation error path — ensure API rejects bad input quickly ─────────

    @task(2)
    def predict_invalid_input(self) -> None:
        """Confirm 422 validation errors are returned quickly (not 500)."""
        bad_payload = {
            "transaction_id": "bad-001",
            "graph": {"node_features": [0.1] * 10},  # wrong length — must be 166
            "memo_text": "test",
            "time_series": {"window": [[1.0, 2.0]]},
        }
        with self.client.post(
            "/predict",
            json=bad_payload,
            name="/predict [invalid — expect 422]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 422:
                resp.success()  # expected error — not a real failure
            else:
                resp.failure(f"Expected 422, got {resp.status_code}")

    # ── SageMaker-compatible endpoints ────────────────────────────────────────

    @task(2)
    def ping(self) -> None:
        """SageMaker /ping liveness check."""
        with self.client.get("/ping", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Ping failed: {resp.status_code}")


class BurstUser(AmlApiUser):
    """Simulates a short burst of aggressive scoring (e.g. batch reconciliation)."""

    wait_time = between(0.001, 0.01)  # near-zero think time

    @task
    def predict_burst(self) -> None:
        payload = _make_payload(high_risk=random.random() < 0.1)
        self.client.post("/predict", json=payload, name="/predict [burst]")
