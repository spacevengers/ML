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

# Default: the remote harmonizer HTTP service. Override for tests, training,
# or batch prediction:
#   docker compose run --rm ml python3 -m pytest -q
#   docker compose run --rm ml python3 -m terrawatch all --start ... --end ...
EXPOSE 9000
CMD ["python3", "-m", "terrawatch", "serve", "--host", "0.0.0.0", "--port", "9000"]
