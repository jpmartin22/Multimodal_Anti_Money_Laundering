"""FastAPI application for AML multimodal scoring.

Run locally:
    uvicorn multimodal_anti_money_laundering.serving.api:app --reload --port 8000

The /predict endpoint returns a stub score of 0.5 until the Week-3 fusion model
is loaded via load_model().  All schema validation and routing are production-ready.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import Response

from multimodal_anti_money_laundering.monitoring.metrics_exporter import (
    get_metrics_output,
    latency_histogram,
    load_model_metrics,
    predictions_counter,
)
from multimodal_anti_money_laundering.serving.schemas import (
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AML Multimodal Scoring API",
    description="Late-fusion GraphSAGE + DistilBERT + BiLSTM AML risk scorer",
    version="0.1.0",
)

# Load model metrics into Prometheus gauges on startup
load_model_metrics()

_MODEL = None
_THRESHOLD = 0.5

# Auto-load fusion model if checkpoint exists
_FUSION_PT = Path("models/fusion/fusion_mlp.pt")
if _FUSION_PT.exists():
    try:
        from multimodal_anti_money_laundering.models.fusion import AMLFusionModel
        _MODEL = AMLFusionModel.from_default_paths()
        logger.info("Fusion model loaded — live scoring enabled")
    except Exception:
        logger.exception("Fusion model load failed — falling back to stub score")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "stub" if _MODEL is None else "loaded"}


# SageMaker requires exactly /ping and /invocations
@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/invocations", response_model=PredictResponse)
def invocations(request: PredictRequest) -> PredictResponse:
    return predict(request)


@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=get_metrics_output(), media_type="text/plain; version=0.0.4"
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    start = time.perf_counter()

    if _MODEL is None:
        logger.warning(
            "Fusion model not loaded — returning stub score for %s",
            request.transaction_id,
        )
        score = 0.5
    else:
        score = _MODEL.score(request)

    elapsed = time.perf_counter() - start
    flagged = score >= _THRESHOLD

    latency_histogram.observe(elapsed)
    predictions_counter.labels(flagged=str(flagged)).inc()

    return PredictResponse(
        transaction_id=request.transaction_id,
        aml_risk_score=score,
        flagged=flagged,
        threshold=_THRESHOLD,
    )


def load_model(model_path: Path | None = None) -> None:
    """(Re)load the fusion model. Called at startup or by tests."""
    global _MODEL
    from multimodal_anti_money_laundering.models.fusion import AMLFusionModel
    _MODEL = AMLFusionModel.from_default_paths()
    logger.info("Fusion model loaded from default paths")
