# VisionGuard web app (Flask + gunicorn).
# Order matters for the layer cache: things that change rarely (base image, OS user,
# Python dependencies) come first; the source code, which changes on every commit,
# is copied last. Editing a template then only rebuilds the final small layers instead
# of re-downloading ~500 MB of TensorFlow.

# slim = Debian without compilers/docs (~45 MB base instead of ~1 GB for python:3.11)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# true  -> tensorflow-cpu is installed so the model runs inside this container (Task 6)
# false -> lightweight image that calls the cloud model API (Task 7/8, INFERENCE_MODE=remote)
ARG WITH_TF=true

WORKDIR /app
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

# 1) dependencies — cached until a requirements file changes
COPY requirements-serve.txt requirements-model.txt ./
RUN pip install -r requirements-serve.txt \
 && if [ "$WITH_TF" = "true" ]; then pip install -r requirements-model.txt; fi

# 2) source code — changes often, so it comes last
COPY visionguard/ visionguard/
COPY webapp/ webapp/
COPY wsgi.py .

# the model is NOT copied: docker-compose mounts ./models read-only at /app/models
RUN mkdir -p /app/models /app/instance && chown -R appuser:appuser /app/instance
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# gunicorn = production WSGI server; 1 worker so the model is in memory once, threads for concurrency
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", \
     "--access-logfile", "-", "wsgi:app"]
