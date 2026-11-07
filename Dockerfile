# ==============================================================================
# Axiom Dispatch - Production Container Image
# Multi-stage, high-performance Python 3.12 runtime with uv package manager
# ==============================================================================

FROM python:3.12-slim-bookworm AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_LINK_MODE=copy

WORKDIR /app

# Install system dependencies (curl for healthchecks / uv installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Astral uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.6.2 /uv /bin/uv

# Copy dependency manifests first for optimal layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code and configurations
COPY app/ app/
COPY config/ config/
COPY scripts/ scripts/
COPY README.md ./

# Create runtime directories for exports and logs
RUN mkdir -p exports/briefings exports/thumbnails logs

# Default entry point: runs the 24-hour autonomous scheduling agent
CMD ["uv", "run", "python", "-m", "app.agent.ranking_agent", "--daemon", "--interval-hours", "24"]
