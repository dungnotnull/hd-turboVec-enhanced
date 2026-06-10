# TurboVec Enhanced — Multi-stage Dockerfile
# Base: python:3.12-slim (non-root user agentuser)
# GPU: install CUDA dependencies via pip (torch with CUDA) in gpu stage

# ─── Stage 1: Builder ────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 agentuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=agentuser:agentuser . .

# Create writable directories
RUN mkdir -p .index models agent/memory/data && \
    chown -R agentuser:agentuser .index models agent/memory/data

USER agentuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

EXPOSE 8016

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8016/health')"

CMD ["python", "-m", "agent.main", "serve", "--host", "0.0.0.0", "--port", "8016"]
