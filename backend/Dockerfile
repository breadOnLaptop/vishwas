ARG PYTHON_VERSION=3.11
ARG ENV=dev
FROM python:${PYTHON_VERSION}-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

# System deps for building wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates wget curl build-essential libffi-dev libssl-dev pkg-config libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create venv and add to PATH
RUN python -m venv $VENV_PATH \
    && $VENV_PATH/bin/pip install --upgrade pip setuptools wheel \
    && ln -s $VENV_PATH/bin/python /usr/local/bin/container-python

ENV PATH="$VENV_PATH/bin:$PATH"

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . /app
RUN chmod -R a+rX /app

# Secrets dir
RUN mkdir -p /secrets && chown -R root:root /secrets

# Non-root user
RUN useradd --create-home --shell /bin/bash vishwasuser \
    && chown -R vishwasuser:vishwasuser /app /secrets

USER vishwasuser
ENV HOME=/home/vishwasuser

EXPOSE 8000

# CMD: reload in dev, single worker in prod
ARG ENV
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port 8000 $( [ \"$ENV\" = \"dev\" ] && echo \"--reload\" || echo \"--workers 1\" )"]
