"""
Gradio demo for the Multimodal AML Risk Scorer.

Runs on HuggingFace Spaces (Gradio SDK) at port 7860.
Accepts simplified transaction inputs, constructs the full API payload,
and calls the FastAPI scoring endpoint.

Set AML_API_URL env var to point at a running FastAPI instance.
Defaults to localhost:8000 (works when run alongside the API container).

Deploy to HuggingFace Spaces:
    python deploy/push_to_spaces.py --username <hf-user> --gradio
"""

from __future__ import annotations

import json
import os
import random

import gradio as gr

try:
    import httpx

    _HTTPX = True
except ImportError:
    _HTTPX = False

API_URL = os.getenv("AML_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Pre-built example transactions that illustrate different risk levels
# ---------------------------------------------------------------------------
_EXAMPLES = [
    [500.0, "payroll disbursement Q2", 9, "corp-001", "emp-042"],
    [95000.0, "consulting invoice offshore account", 23, "shell-X", "anon-99"],
    [1200.0, "rent payment March", 10, "tenant-7", "landlord-3"],
    [48000.0, "cash equivalent transfer urgent", 2, "acct-Z", "acct-W"],
    [75.0, "coffee shop receipt", 14, "user-101", "merchant-55"],
]

_RISK_CSS = """
.risk-high { color: #d32f2f; font-size: 1.6em; font-weight: 700; }
.risk-low  { color: #388e3c; font-size: 1.6em; font-weight: 700; }
"""


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _stub_score(amount: float, memo: str, hour: int) -> float:
    """Rule-based fallback when the API is unreachable (demo only)."""
    risk = 0.0
    if amount > 50000:
        risk += 0.4
    if amount > 80000:
        risk += 0.2
    suspicious_terms = {"offshore", "urgent", "cash", "anonymous", "shell", "anon"}
    if any(t in memo.lower() for t in suspicious_terms):
        risk += 0.3
    if hour < 6 or hour > 22:
        risk += 0.1
    return min(round(risk, 4), 1.0)


def _build_payload(
    amount: float,
    memo_text: str,
    hour_of_day: int,
    sender_id: str,
    receiver_id: str,
) -> dict:
    # Deterministically derive node features from sender/receiver IDs so the
    # demo is reproducible per pair without needing real graph embeddings.
    seed_val = abs(hash(f"{sender_id}:{receiver_id}")) % (2**31)
    rng = random.Random(seed_val)
    node_features = [round(rng.random(), 6) for _ in range(166)]

    normalised_amount = min(amount / 100_000.0, 1.0)
    tx_type = 1.0 if amount > 10_000 else 0.0
    velocity = normalised_amount
    window = [[normalised_amount, hour_of_day / 23.0, 0.5, tx_type, velocity]] * 5

    return {
        "transaction_id": f"demo-{rng.randint(10_000, 99_999)}",
        "graph": {"node_features": node_features},
        "memo_text": memo_text,
        "time_series": {"window": window},
    }


def score_transaction(
    amount: float,
    memo_text: str,
    hour_of_day: int,
    sender_id: str,
    receiver_id: str,
) -> tuple[float, str, str]:
    """Return (risk_score, risk_html, full_json_response)."""
    payload = _build_payload(
        amount, memo_text, int(hour_of_day), sender_id, receiver_id
    )
    api_ok = False
    score: float = 0.0
    flagged: bool = False
    threshold: float = 0.5
    raw: dict = {}

    if _HTTPX:
        try:
            resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=8.0)
            resp.raise_for_status()
            raw = resp.json()
            score = float(raw.get("aml_risk_score", 0.0))
            flagged = bool(raw.get("flagged", False))
            threshold = float(raw.get("threshold", 0.5))
            api_ok = True
        except Exception:
            pass

    if not api_ok:
        score = _stub_score(amount, memo_text, int(hour_of_day))
        flagged = score >= threshold
        raw = {
            "transaction_id": payload["transaction_id"],
            "aml_risk_score": score,
            "flagged": flagged,
            "threshold": threshold,
            "_source": "stub — API unreachable",
        }

    label = "HIGH RISK" if flagged else "LOW RISK"
    css_cls = "risk-high" if flagged else "risk-low"
    risk_html = (
        f"<div class='{css_cls}'>{label}</div>"
        f"<p>Score: <strong>{score:.4f}</strong> &nbsp;|&nbsp; "
        f"Threshold: {threshold}</p>"
    )

    return score, risk_html, json.dumps(raw, indent=2)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="AML Multimodal Risk Scorer", css=_RISK_CSS) as demo:
    gr.Markdown(
        """
# Multimodal Anti-Money Laundering Risk Scorer

**DePaul SE489 — Phase 3 Demo**

This interface scores financial transactions for AML risk using a late-fusion
model that combines three signal types:
- **Graph features** (GraphSAGE on the Elliptic transaction network)
- **Memo text** (DistilBERT on payment descriptions)
- **Behavioural time series** (Bi-LSTM on 30-day rolling windows)

Enter transaction details below and click **Score Transaction**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            amount = gr.Number(
                label="Transaction Amount (USD)",
                value=1_000.0,
                minimum=0.01,
            )
            memo_text = gr.Textbox(
                label="Payment Memo",
                placeholder="e.g. payroll disbursement Q2",
                lines=2,
            )
            hour_of_day = gr.Slider(
                label="Transaction Hour (0–23)",
                minimum=0,
                maximum=23,
                step=1,
                value=10,
            )
            sender_id = gr.Textbox(label="Sender ID", value="sender-001")
            receiver_id = gr.Textbox(label="Receiver ID", value="receiver-001")

            score_btn = gr.Button("Score Transaction", variant="primary")

        with gr.Column(scale=1):
            risk_gauge = gr.Slider(
                label="AML Risk Score",
                minimum=0.0,
                maximum=1.0,
                interactive=False,
                value=0.0,
            )
            risk_label = gr.HTML(label="Risk Level")
            raw_output = gr.Code(
                label="Full API Response",
                language="json",
                lines=12,
            )

    gr.Examples(
        examples=_EXAMPLES,
        inputs=[amount, memo_text, hour_of_day, sender_id, receiver_id],
        label="Example Transactions",
    )

    score_btn.click(
        fn=score_transaction,
        inputs=[amount, memo_text, hour_of_day, sender_id, receiver_id],
        outputs=[risk_gauge, risk_label, raw_output],
    )

    gr.Markdown(
        """
---
**Model targets:** AUC-PR ≥ 0.80 · Precision @ Recall=0.8 ≥ 0.70 · P95 latency < 200 ms

**Data:** [Elliptic Bitcoin dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

Source: [GitHub](https://github.com/jpmartin22/Multimodal_Anti_Money_Laundering)
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
