"""Task 7 — the model as a small JSON API that runs in the cloud.

  GET  /health            -> {"status": "ok", model info}           (no key needed)
  POST /predict           -> {"label", "confidence", "p_ai", ...}  (header X-API-Key required)
       body: multipart/form-data with field "file"  OR  raw image bytes

The key is read from the MODEL_API_KEY environment variable (a cloud secret), never
from code. Comparison uses hmac.compare_digest to avoid timing attacks.
"""
import hmac
import logging
import os
import time
import uuid

from flask import Flask, jsonify, request

from visionguard.inference import KerasPredictor
from visionguard.preprocess import ImageDecodeError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("model_api")

MAX_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))


def create_app(predictor=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(MAX_MB * 1024 * 1024)
    api_key = os.getenv("MODEL_API_KEY", "")
    if predictor is None:
        thr = os.getenv("THRESHOLD")
        predictor = KerasPredictor(os.getenv("MODEL_PATH", "models/champion.keras"), os.getenv("MODEL_META"),
                                   float(thr) if thr else None)  # loaded once per worker
    log.info("model %s v%s loaded, threshold %.3f", predictor.model_name, predictor.model_version,
             predictor.threshold)

    def err(status, message):
        return jsonify({"error": message}), status

    @app.get("/")
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_name": predictor.model_name,
                        "model_version": predictor.model_version, "threshold": predictor.threshold,
                        "img_size": predictor.img_size, "auth_required": bool(api_key)})

    @app.post("/predict")
    def predict():
        rid = uuid.uuid4().hex[:8]
        if api_key and not hmac.compare_digest(request.headers.get("X-API-Key", ""), api_key):
            log.warning("req=%s 401 bad or missing API key from %s", rid, request.remote_addr)
            return err(401, "missing or invalid X-API-Key header")
        f = request.files.get("file")
        raw = f.read() if f else request.get_data()
        if not raw:
            return err(400, "send the image as multipart field 'file' or as the raw request body")
        t0 = time.perf_counter()
        try:
            pred = predictor.predict_bytes(raw)
        except ImageDecodeError:
            log.info("req=%s 422 undecodable image (%d bytes)", rid, len(raw))
            return err(422, "file could not be decoded as an image")
        body = pred.to_dict() | {"request_id": rid,
                                 "server_ms": round((time.perf_counter() - t0) * 1000, 1)}
        log.info("req=%s 200 file=%s bytes=%d label=%s p_ai=%.4f ms=%.1f", rid,
                 f.filename if f else "-", len(raw), pred.label, pred.p_ai, body["server_ms"])
        return jsonify(body)

    @app.errorhandler(413)
    def too_big(_):
        return err(413, f"file larger than {MAX_MB:.0f} MB")

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
