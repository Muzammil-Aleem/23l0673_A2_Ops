# VisionGuard model API — deployed to the cloud (Task 7). Build from the repo root:
#   python scripts/fetch_model.py                     # puts the champion in models/
#   docker build -f model_api/Dockerfile -t visionguard-api .
# Here the model IS baked into the image: a cloud container service (Cloud Run, Azure
# Container Apps, ...) has no host folder to mount, and an immutable image means the
# exact model that was tested is the one that runs.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080 MODEL_PATH=/app/models/champion.keras MODEL_META=/app/models/champion.json

WORKDIR /app
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY model_api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY visionguard/ visionguard/
COPY model_api/ model_api/
COPY models/champion.keras models/champion.json models/

USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4).status == 200 else 1)"

# shell form so $PORT (set by Cloud Run / Render / etc.) is expanded
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 --access-logfile - model_api.wsgi:app
