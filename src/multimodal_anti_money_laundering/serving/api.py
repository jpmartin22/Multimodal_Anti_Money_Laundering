"""FastAPI application for AML multimodal scoring.

Run locally:
    uvicorn multimodal_anti_money_laundering.serving.api:app --reload --port 8000

The /predict endpoint returns a stub score of 0.5 until the Week-3 fusion model
is loaded via load_model().  All schema validation and routing are production-ready.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from multimodal_anti_money_laundering.serving.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AML Multimodal Scoring API",
    description="Late-fusion GraphSAGE + DistilBERT + BiLSTM AML risk scorer",
    version="0.1.0",
)

_MODEL = None
_THRESHOLD = 0.5


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "stub" if _MODEL is None else "loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _MODEL is None:
        logger.warning(
            "Fusion model not loaded — returning stub score for %s",
            request.transaction_id,
        )
        score = 0.5
    else:
        score = _MODEL.score(request)  # wired in Week 3

    return PredictResponse(
        transaction_id=request.transaction_id,
        aml_risk_score=score,
        flagged=score >= _THRESHOLD,
        threshold=_THRESHOLD,
    )


def load_model(model_path: Path) -> None:
    """Replace the stub with the trained fusion model at startup."""
    global _MODEL
    logger.info("Loading fusion model from %s", model_path)
    # TODO Week 3: from multimodal_anti_money_laundering.models.fusion import FusionModel
    # TODO Week 3: _MODEL = FusionModel.load(model_path)
