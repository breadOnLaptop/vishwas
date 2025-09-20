FROM python:3.11-slim

# avoid interactive prompts from apt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/secrets/google-credentials.json

WORKDIR /app

# Install minimal system deps. We install build deps only temporarily.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       gnupg \
       wget \
       curl \
       build-essential \
       libffi-dev \
       libssl-dev \
       pkg-config \
       libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install PowerShell (optional; keep if your scripts use pwsh)
# Note: this targets Debian-style packages. If base image changes, this step may need adjustment.
RUN wget -q https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb \
    && dpkg -i /tmp/packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends powershell \
    && rm -f /tmp/packages-microsoft-prod.deb \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverage Docker cache)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies as root, then remove build deps to keep image small
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && apt-get purge -y --auto-remove build-essential pkg-config libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# Copy application code
COPY . /app
RUN chmod -R a+rX /app

# Create secrets dir (mount your GCP service account here at runtime)
RUN mkdir -p /secrets && chown -R root:root /secrets

# create non-root user and chown app dir
RUN useradd --create-home --shell /bin/bash vishwasuser \
    && chown -R vishwasuser:vishwasuser /app /secrets

USER vishwasuser
ENV HOME=/home/vishwasuser

EXPOSE 8000

# lightweight healthcheck (adjust to your actual HTTP path if applicable)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

# Default: run PowerShell script if you rely on it (keeps parity with your current setup).
# If you prefer to run FastAPI directly with uvicorn in production, use the CMD line (commented) instead.
ENTRYPOINT ["pwsh", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "/app/scripts/start-backend.ps1"]

# Recommended alternative (uncomment to use uvicorn directly; comment out ENTRYPOINT above)
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
