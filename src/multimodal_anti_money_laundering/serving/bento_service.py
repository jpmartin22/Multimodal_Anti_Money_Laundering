"""BentoML service scaffold for the AML multimodal scorer.

Build:  bentoml build
Serve:  bentoml serve aml-scorer:latest

The service returns stub scores until the Week-3 fusion model is saved into the
BentoML model store and loaded in __init__.
"""

from __future__ import annotations

import bentoml

from multimodal_anti_money_laundering.serving.schemas import PredictRequest, PredictResponse

_THRESHOLD = 0.5


@bentoml.service(
    name="aml-scorer",
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class AMLScoringService:
    def __init__(self) -> None:
        # TODO Week 3: load the trained fusion model from the BentoML model store
        # self.model = bentoml.picklable_model.load_model("aml_fusion_model:latest")
        self.model = None

    @bentoml.api
    def predict(self, request: PredictRequest) -> PredictResponse:
        if self.model is None:
            score = 0.5  # stub — replaced when fusion model is available
        else:
            score = self.model.score(request)

        return PredictResponse(
            transaction_id=request.transaction_id,
            aml_risk_score=score,
            flagged=score >= _THRESHOLD,
            threshold=_THRESHOLD,
        )

    @bentoml.api
    def healthz(self) -> dict[str, str]:
        return {"status": "ok", "model": "stub" if self.model is None else "loaded"}
