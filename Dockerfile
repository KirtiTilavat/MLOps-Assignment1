# Heart Disease Risk API — multi-stage build
#   Stage 1 (trainer): full toolchain, downloads data and trains the model.
#   Stage 2 (runtime): slim image with only the API dependencies + trained model.

# ---------------------------------------------------------------- trainer
FROM python:3.13-slim AS trainer

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scripts/ scripts/
COPY src/ src/
RUN python scripts/download_data.py \
 && python -m src.data_processing \
 && python -m src.train \
 && python -m src.predict

# ---------------------------------------------------------------- runtime
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/ src/
COPY app/ app/
COPY --from=trainer /build/models/model.joblib /build/models/metadata.json models/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# Access logging is done by app/monitoring.py (JSON, one line per request).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
