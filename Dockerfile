# ─────────────────────────────────────────────
# Stage 1 – Builder (install deps)
# ─────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Stage 2 – Runtime (lean final image)
# ─────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="Rohit Chaurasiya"
LABEL description="Goa AI Recommender – FastAPI MLOps Service"
LABEL version="1.0.0"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY model/ ./model/
COPY frontend/ ./frontend/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]