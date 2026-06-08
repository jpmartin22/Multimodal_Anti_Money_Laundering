# Dockerfile for HuggingFace Spaces (Docker SDK)
# Runs FastAPI on port 7860 with Gradio frontend
#
# Build locally to test:
#   docker build -f dockerfiles/Dockerfile.hf -t aml-hf .
#   docker run --rm -p 7860:7860 aml-hf
#   curl http://localhost:7860/health

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

COPY requirements.serve.txt .
RUN pip install --no-cache-dir --user -r requirements.serve.txt
RUN pip install --no-cache-dir --user gradio requests

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="Multimodal AML Detection — HF Space" \
      org.opencontainers.image.description="GraphSAGE + DistilBERT + BiLSTM AML risk scorer on HuggingFace Spaces" \
      org.opencontainers.image.version="0.1.0"

WORKDIR /app

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=file:///app/mlruns \
    AML_THRESHOLD=0.5 \
    # HuggingFace Spaces standard port
    PORT=7860 \
    LOG_LEVEL=INFO

COPY . .

RUN pip install --no-cache-dir -e .


# Start script that runs both FastAPI and Gradio
EXPOSE 7860
EXPOSE 7860

CMD ["python", "app.py"]

