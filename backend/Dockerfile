# ============================================================================
# Kangaroo Shift Backend — Multi-Stage Dockerfile
#
# Multi-stage build reduces final image from ~1.2GB to ~450MB:
#   Stage 1 (builder): Install all dependencies including build tools
#   Stage 2 (runtime): Copy only what's needed to run
#
# Security:
#   - Runs as non-root user (appuser) — container compromise ≠ host compromise
#   - No build tools in final image — smaller attack surface
#   - No dev dependencies in production image
# ============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies (gcc, etc. for bcrypt, cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install dependencies into a virtual environment for clean copying
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -e .


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --system appuser && useradd --system --gid appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create storage directory for local mode
RUN mkdir -p /app/storage && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check — Kubernetes/Docker will restart unhealthy containers
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]


# ── Stage 3: Development (extends runtime) ────────────────────────────────────
FROM runtime AS development

USER root

# Install dev dependencies for testing, linting
COPY --from=builder /opt/venv /opt/venv
RUN pip install --no-cache-dir -e ".[dev]"

USER appuser

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload"]
