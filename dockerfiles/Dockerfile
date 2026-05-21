
# Multi-stage build for Multimodal Anti Money Laundering

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.serve.txt .
RUN pip install --no-cache-dir --user -r requirements.serve.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="Multimodal AML Detection" \
      org.opencontainers.image.description="GraphSAGE + DistilBERT + BiLSTM AML risk scorer" \
      org.opencontainers.image.source="https://github.com/jpmartin22/Multimodal_Anti_Money_Laundering" \
      org.opencontainers.image.version="0.1.0"

WORKDIR /app

# Copy user-installed packages from builder
COPY --from=builder /root/.local /root/.local

# ── Environment variables (override at runtime via --env or env_file) ─────────
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # MLflow tracking — default to local filesystem inside the container
    MLFLOW_TRACKING_URI=file:///app/mlruns \
    # Inference API settings
    AML_THRESHOLD=0.5 \
    PORT=8000 \
    # Logging level: DEBUG | INFO | WARNING | ERROR
    LOG_LEVEL=INFO

# Copy project code
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE ${PORT}

# CMD allows docker-compose (and SageMaker) to override the entrypoint per service
CMD ["uvicorn", "multimodal_anti_money_laundering.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
