# ---- base ligera con Python 3.11 (más rápida y ruedas recientes)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # evita sobre-subscription de BLAS/FFT
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# deps del sistema mínimos para wheels (certs + runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# instala primero dependencias para cacheo
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# copia el resto
COPY . .

# expón el puerto (opcional, informativo)
EXPOSE 5410

# usa PORT si está definido (Coolify lo inyecta), si no, 5410
CMD gunicorn -w 2 -k gthread -t 180 --threads 8 \
    --bind 0.0.0.0:${PORT:-5410} app:app
