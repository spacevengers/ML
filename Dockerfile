# syntax=docker/dockerfile:1

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM needs the OpenMP runtime at import time.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . .

# The default command is the offline test suite. Override it for training or
# prediction, e.g.:
#   docker compose run --rm ml python3 -m terrawatch all --start ... --end ...
CMD ["python3", "-m", "pytest", "-q"]
