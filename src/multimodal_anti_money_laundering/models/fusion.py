"""Late-fusion MLP for multimodal AML detection (Phase 3).

Combines three encoder embeddings:
  - GraphSAGE  (64-dim)  — graph topology
  - BiLSTM     (64-dim)  — temporal behavior
  - DistilBERT (768-dim) — payment memo text, projected to 64-dim internally

Fusion MLP: 192 → 128 → 64 → 1 (raw logit)
Platt calibration applied post-training to fix threshold instability.

Saved artifacts
---------------
  models/fusion/fusion_mlp.pt        — LateFusionMLP state dict
  models/fusion/fusion_calibrator.joblib — Platt calibrator (optional)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from multimodal_anti_money_laundering.serving.schemas import PredictRequest

logger = logging.getLogger(__name__)

# Encoder output dimensions
GRAPH_DIM = 64
BILSTM_DIM = 64
BERT_HIDDEN = 768
TEXT_PROJ_DIM = 64
FUSED_DIM = GRAPH_DIM + BILSTM_DIM + TEXT_PROJ_DIM  # 192


class LateFusionMLP(nn.Module):
    """MLP fusion head over [graph | bilstm | text] embeddings.

    The text_proj layer (768→64) is included here so it is trained and saved
    alongside the fusion weights — no separate projection checkpoint needed.

    Input
    -----
    graph_emb  : (batch, 64)  from GraphSAGE encoder
    bilstm_emb : (batch, 64)  from BiLSTM encoder
    bert_cls   : (batch, 768) raw DistilBERT [CLS] token

    Output
    ------
    (batch, 1) raw logit — apply sigmoid for probability
    """

    def __init__(
        self,
        bert_hidden: int = BERT_HIDDEN,
        text_proj_dim: int = TEXT_PROJ_DIM,
        fused_dim: int = FUSED_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.text_proj = nn.Linear(bert_hidden, text_proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        graph_emb: torch.Tensor,
        bilstm_emb: torch.Tensor,
        bert_cls: torch.Tensor,
    ) -> torch.Tensor:
        text_emb = self.text_proj(bert_cls)
        fused = torch.cat([graph_emb, bilstm_emb, text_emb], dim=1)
        return self.mlp(fused)


class AMLFusionModel:
    """Full inference wrapper: loads all 3 encoders + fusion MLP + calibrator.

    Usage
    -----
    model = AMLFusionModel.from_default_paths()
    score = model.score(predict_request)
    """

    N_FEATURES = 165  # Elliptic features per node (time_step col excluded)
    SEQ_LEN = 49      # BiLSTM sequence length

    def __init__(
        self,
        graphsage_path: Path,
        bilstm_path: Path,
        distilbert_dir: Path | None,
        fusion_path: Path,
        calibrator_path: Path | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self._load_graphsage(graphsage_path)
        self._load_bilstm(bilstm_path)
        self._load_distilbert(distilbert_dir)
        self._load_fusion(fusion_path)
        self._load_calibrator(calibrator_path)

    @classmethod
    def from_default_paths(cls, device: str = "cpu") -> AMLFusionModel:
        return cls(
            graphsage_path=Path("models/graphsage/graphsage_encoder.pt"),
            bilstm_path=Path("models/bilstm/bilstm_encoder.pt"),
            distilbert_dir=Path("models/distilbert/memo_model"),
            fusion_path=Path("models/fusion/fusion_mlp.pt"),
            calibrator_path=Path("models/fusion/fusion_calibrator.joblib"),
            device=device,
        )

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_graphsage(self, path: Path) -> None:
        from multimodal_anti_money_laundering.train_graphsage import GraphSAGEEncoder

        self.graphsage = GraphSAGEEncoder()
        self.graphsage.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.graphsage.eval().to(self.device)
        logger.info("GraphSAGE encoder loaded from %s", path)

    def _load_bilstm(self, path: Path) -> None:
        from multimodal_anti_money_laundering.train_bilstm import BiLSTMEncoder

        self.bilstm = BiLSTMEncoder()
        self.bilstm.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.bilstm.eval().to(self.device)
        logger.info("BiLSTM encoder loaded from %s", path)

    def _load_distilbert(self, model_dir: Path | None) -> None:
        self.tokenizer = None
        self.distilbert_base = None

        if model_dir is None or not model_dir.exists():
            logger.warning(
                "DistilBERT dir not found (%s) — text branch uses zero embeddings", model_dir
            )
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            bert_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            self.distilbert_base = bert_model.distilbert.eval().to(self.device)
            logger.info("DistilBERT base loaded from %s", model_dir)
        except Exception:
            logger.exception("DistilBERT load failed — text branch uses zero embeddings")

    def _load_fusion(self, path: Path) -> None:
        self.fusion = LateFusionMLP()
        self.fusion.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.fusion.eval().to(self.device)
        logger.info("Fusion MLP loaded from %s", path)

    def _load_calibrator(self, path: Path | None) -> None:
        self.calibrator = None
        if path and path.exists():
            import joblib

            self.calibrator = joblib.load(path)
            logger.info("Platt calibrator loaded from %s", path)

    # ── Embedding extractors ──────────────────────────────────────────────────

    @torch.no_grad()
    def _graph_embedding(self, node_features: list[float]) -> torch.Tensor:
        feats = (
            torch.tensor(node_features[: self.N_FEATURES], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        # Single-node graph at inference — no neighbour edges available
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        return self.graphsage(feats, edge_index)  # (1, 64)

    @torch.no_grad()
    def _bilstm_embedding(self, window: list[list[float]]) -> torch.Tensor:
        seq = np.zeros((self.SEQ_LEN, self.N_FEATURES), dtype=np.float32)
        for i, row in enumerate(window[: self.SEQ_LEN]):
            trimmed = row[: self.N_FEATURES]
            seq[i, : len(trimmed)] = trimmed
        t = torch.tensor(seq).unsqueeze(0).to(self.device)  # (1, 49, 165)
        return self.bilstm(t)  # (1, 64)

    @torch.no_grad()
    def _bert_cls(self, memo_text: str) -> torch.Tensor:
        if self.tokenizer is None or self.distilbert_base is None:
            return torch.zeros(1, BERT_HIDDEN, device=self.device)
        tokens = self.tokenizer(
            memo_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        out = self.distilbert_base(**tokens)
        return out.last_hidden_state[:, 0, :]  # (1, 768) CLS token

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, request: PredictRequest) -> float:
        """Return calibrated AML risk probability in [0, 1]."""
        graph_emb = self._graph_embedding(request.graph.node_features)
        bilstm_emb = self._bilstm_embedding(request.time_series.window)
        bert_cls = self._bert_cls(request.memo_text)

        logit = self.fusion(graph_emb, bilstm_emb, bert_cls)
        prob = float(torch.sigmoid(logit).item())

        if self.calibrator is not None:
            prob = float(self.calibrator.predict_proba([[prob]])[0, 1])

        return round(prob, 4)
