"""Smoke tests for the AML scoring API stub."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from multimodal_anti_money_laundering.serving.api import app

client = TestClient(app)

_VALID_PAYLOAD = {
    "transaction_id": "tx-test-001",
    "graph": {"node_features": [0.0] * 166},
    "memo_text": "consulting services invoice Q1",
    "time_series": {"window": [[100.0, 14.0, 2.0, 1.0, 500.0]] * 5},
}
pytest.importorskip("bentoml")

def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_stub_returns_valid_schema():
    r = client.post("/predict", json=_VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "tx-test-001"
    assert 0.0 <= body["aml_risk_score"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert "threshold" in body


def test_predict_score_in_range():
    r = client.post("/predict", json=_VALID_PAYLOAD)
    assert 0.0 <= r.json()["aml_risk_score"] <= 1.0


def test_predict_missing_memo_returns_422():
    bad = {**_VALID_PAYLOAD}
    del bad["memo_text"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_wrong_node_feature_length_returns_422():
    bad = {**_VALID_PAYLOAD, "graph": {"node_features": [0.0] * 10}}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_empty_time_series_returns_422():
    bad = {**_VALID_PAYLOAD, "time_series": {"window": []}}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_bento_service_falls_back_to_local_model(monkeypatch):
    class DummyModel:
        def score(self, request):
            return 0.9

    def fake_load_model(name):
        raise RuntimeError("no BentoML store available")

    monkeypatch.setattr("bentoml.picklable_model.load_model", fake_load_model)
    monkeypatch.setattr(
        "multimodal_anti_money_laundering.serving.bento_service.AMLFusionModel.from_default_paths",
        lambda: DummyModel(),
    )

    from multimodal_anti_money_laundering.serving.bento_service import AMLScoringService
    from multimodal_anti_money_laundering.serving.schemas import PredictRequest

    service = AMLScoringService()
    assert service.healthz()["model"] == "loaded"

    request = PredictRequest(**_VALID_PAYLOAD)
    response = service.predict(request)
    assert response.aml_risk_score == 0.9
    assert response.flagged is True


def test_bento_service_returns_stub_when_no_model(monkeypatch):
    def fake_load_model(name):
        raise RuntimeError("no BentoML store available")

    def fake_from_default_paths():
        raise RuntimeError("local fusion model unavailable")

    monkeypatch.setattr("bentoml.picklable_model.load_model", fake_load_model)
    monkeypatch.setattr(
        "multimodal_anti_money_laundering.serving.bento_service.AMLFusionModel.from_default_paths",
        fake_from_default_paths,
    )

    from multimodal_anti_money_laundering.serving.bento_service import AMLScoringService
    from multimodal_anti_money_laundering.serving.schemas import PredictRequest

    service = AMLScoringService()
    assert service.healthz()["model"] == "stub"

    request = PredictRequest(**_VALID_PAYLOAD)
    response = service.predict(request)
    assert response.aml_risk_score == 0.5
    assert response.flagged is True
