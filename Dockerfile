# ---------- Stage 1: Build Frontend ----------
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Copy package files and install deps
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build   # outputs /app/frontend/dist

# ---------- Stage 2: Backend + Frontend ----------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates wget curl build-essential libffi-dev libssl-dev pkg-config libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv $VENV_PATH \
    && $VENV_PATH/bin/pip install --upgrade pip setuptools wheel \
    && ln -s $VENV_PATH/bin/python /usr/local/bin/container-python

ENV PATH="$VENV_PATH/bin:$PATH"

# Copy backend requirements and install
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source
COPY backend/ ./backend

# Copy frontend build directly
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Create user and set permissions
RUN useradd --create-home --shell /bin/bash vishwasuser \
    && mkdir -p /secrets \
    && chown -R vishwasuser:vishwasuser /app /secrets

# Switch to non-root user
USER vishwasuser
ENV HOME=/home/vishwasuser

# Expose backend port
EXPOSE 8000

# Run Uvicorn from backend folder to ensure paths work
WORKDIR /app/backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
