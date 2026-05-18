"""Pydantic I/O schemas for the AML scoring API.

Input covers all three modalities consumed by the late-fusion model:
  - graph:       166-dim Elliptic node feature vector (GraphSAGE branch)
  - memo_text:   raw payment description string (DistilBERT branch)
  - time_series: 30-day rolling window of per-transaction features (BiLSTM branch)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GraphInput(BaseModel):
    node_features: list[float] = Field(
        ...,
        min_length=166,
        max_length=166,
        description="Elliptic node feature vector (166-dim, normalised)",
    )


class TimeSeriesInput(BaseModel):
    window: list[list[float]] = Field(
        ...,
        min_length=1,
        description=(
            "Rows of [amount, hour_of_day, day_of_week, tx_type, cumulative_velocity] "
            "covering up to a 30-day rolling window"
        ),
    )


class PredictRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    graph: GraphInput
    memo_text: str = Field(..., description="Raw payment memo / description text")
    time_series: TimeSeriesInput


class PredictResponse(BaseModel):
    transaction_id: str
    aml_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated AML risk probability in [0, 1]"
    )
    flagged: bool = Field(..., description="True when risk score >= threshold")
    threshold: float = Field(0.5, description="Decision threshold applied")
