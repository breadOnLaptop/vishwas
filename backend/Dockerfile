FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

WORKDIR /app

# Install minimal system deps and PowerShell prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    apt-transport-https \
    gnupg \
    ca-certificates \
    build-essential \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PowerShell (Microsoft package for Debian/Ubuntu)
# Note: using packages.microsoft.com package bundle (Debian/Ubuntu). Adjust if base image changes.
RUN wget -q https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb \
    && dpkg -i /tmp/packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends powershell \
    && rm -f /tmp/packages-microsoft-prod.deb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

# Copy application source and scripts
COPY . /app
# ensure scripts are executable
RUN chmod -R a+rX /app

# Create secrets dir for service account mount
RUN mkdir -p /secrets

# Expose port
EXPOSE 8000

# Create non-root user
RUN useradd --create-home --shell /bin/bash vishwasuser && chown -R vishwasuser /app /secrets
USER vishwasuser

# Use PowerShell Core to run the startup script by default
ENTRYPOINT ["pwsh", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "/app/scripts/start-backend.ps1"]
