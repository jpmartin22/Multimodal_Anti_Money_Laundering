"""
app.py — Gradio demo for Multimodal AML Detection System
Deployed on Hugging Face Spaces (Docker SDK)
"""

import gradio as gr
import requests
import json

import os
# Use Cloud Run URL if available, otherwise use local API
API_URL = os.getenv(
    "AML_API_URL",
    "https://aml-multimodal-scorer-177887911927.us-central1.run.app"
)

def predict_transaction(transaction_id, memo_text, risk_level):
    """
    Send a transaction to the AML scoring API and return the result.
    risk_level controls the node features (high/low risk pattern)
    """
    # Generate node features based on risk level
    if risk_level == "High Risk":
        node_features = [0.9] * 166
    elif risk_level == "Low Risk":
        node_features = [0.1] * 166
    else:
        node_features = [0.5] * 166

    payload = {
        "transaction_id": transaction_id,
        "memo_text": memo_text,
        "graph": {
            "node_features": node_features
        },
        "time_series": {
            "window": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        }
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )
        result = response.json()

        risk_score = result.get("aml_risk_score", 0)
        flagged = result.get("flagged", False)
        threshold = result.get("threshold", 0.5)

        # Format output
        status = "🚨 FLAGGED - Suspicious Transaction" if flagged else "✅ CLEAR - Normal Transaction"
        color = "red" if flagged else "green"

        output = f"""
**Transaction ID:** {result.get('transaction_id', transaction_id)}

**AML Risk Score:** {risk_score:.3f}

**Threshold:** {threshold}

**Status:** {status}

**Decision:** {"This transaction has been flagged for compliance review." if flagged else "This transaction appears normal."}
        """
        return output, risk_score

    except Exception as e:
        return f"Error calling API: {str(e)}", 0.0


# Example transactions
examples = [
    ["TX001", "Urgent wire transfer consulting fee shell company offshore", "High Risk"],
    ["TX002", "Monthly salary payment employee John Smith", "Low Risk"],
    ["TX003", "Invoice payment for marketing services rendered Q4", "Medium Risk"],
    ["TX004", "Immediate cash transfer no reference anonymous", "High Risk"],
    ["TX005", "Regular grocery store purchase supermarket", "Low Risk"],
]

# Build Gradio interface
with gr.Blocks(
    title="AML Multimodal Detection System",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🏦 Multimodal Anti-Money Laundering Detection System
    
    **GraphSAGE + DistilBERT + BiLSTM → Late-Fusion MLP**
    
    This system uses three AI models to detect suspicious financial transactions:
    - **GraphSAGE** — analyzes transaction graph topology
    - **DistilBERT** — processes payment memo text for red-flag language
    - **BiLSTM** — detects suspicious behavioral time-series patterns
    
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Transaction Input")

            transaction_id = gr.Textbox(
                label="Transaction ID",
                placeholder="e.g. TX001",
                value="TX001"
            )

            memo_text = gr.Textbox(
                label="Payment Memo Text",
                placeholder="Enter payment description...",
                lines=3,
                value="Urgent wire transfer consulting fee shell company"
            )

            risk_level = gr.Radio(
                label="Graph Risk Pattern",
                choices=["Low Risk", "Medium Risk", "High Risk"],
                value="High Risk",
                info="Simulates the graph topology risk signal"
            )

            submit_btn = gr.Button("🔍 Analyze Transaction", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Risk Assessment")

            result_text = gr.Markdown(label="Assessment Result")

            risk_score_bar = gr.Slider(
                label="AML Risk Score",
                minimum=0,
                maximum=1,
                value=0,
                interactive=False,
                info="Score above 0.5 triggers a flag"
            )

    gr.Markdown("### 💡 Try Example Transactions")
    gr.Examples(
        examples=examples,
        inputs=[transaction_id, memo_text, risk_level],
        label="Click any example to load it"
    )

    submit_btn.click(
        fn=predict_transaction,
        inputs=[transaction_id, memo_text, risk_level],
        outputs=[result_text, risk_score_bar]
    )

    gr.Markdown("""
    ---
    **Model Performance:**
    - BiLSTM AUC-PR: **0.9324** | F1: **0.8672**
    - DistilBERT AUC-PR: **0.8418** | F1: **0.9011**
    - XGBoost Baseline AUC-PR: **0.9891**
    
    *Built by Data2Deploy — SE 489 MLOps, DePaul University 2025*
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
