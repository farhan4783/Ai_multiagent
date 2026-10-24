# ============================================================
# Multi-Stage Dockerfile for STGAT-MARL Autonomous System
# Stage 1: Build React + Vite Web Dashboard SPA
# Stage 2: Python Runtime with Orchestrator Server & Models
# ============================================================

# --- Stage 1: Build Web Frontend Assets ---
FROM node:20-alpine AS frontend-builder
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- Stage 2: Python Runtime Environment ---
FROM python:3.11-slim AS runtime

# Install system dependencies (git is required for git miner and worktree manager)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure git identity for automated agent commits
RUN git config --global user.name "Autonomous Swarm Agent" \
    && git config --global user.email "swarm-agent@stgat-marl.local" \
    && git config --global init.defaultBranch main

WORKDIR /app

# Copy project metadata and install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e ".[api,ml,dev]"

# Copy application source code, scripts, checkpoints, and data
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY checkpoints/ ./checkpoints/
COPY constants.py schemas.py ./

# Copy pre-built frontend SPA assets from Stage 1
COPY --from=frontend-builder /web/dist /app/web/dist

# Expose FastAPI backend and web dashboard port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    ORCHESTRATOR_HOST=0.0.0.0 \
    ORCHESTRATOR_PORT=8000

# Run orchestrator server with dashboard
CMD ["python", "scripts/run_orchestrator_server.py", "--host", "0.0.0.0", "--port", "8000"]
