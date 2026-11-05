"""Task 4.2 / 5 / 8 — pages: upload (/), result (/predict, /result/<id>), history (/history),
health check (/health) and a JSON history endpoint (/api/history)."""
import base64
import io

from flask import Blueprint, current_app, jsonify, render_template, request
from PIL import Image
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from visionguard.preprocess import decode_image
from webapp.db import PredictionRecord, db
from webapp.validation import UploadError, validate_upload

bp = Blueprint("main", __name__)

CAVEAT = ("This is a statistical guess from a model trained on about a thousand web images. "
          "It can be wrong — especially for edited, compressed, painted or very small images — "
          "and must not be used as sole proof that a picture is fake or real.")


def thumbnail_data_uri(raw: bytes) -> str:
    """Small preview rendered straight into the HTML; nothing is written to disk."""
    img = decode_image(raw)
    img.thumbnail((360, 360))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@bp.get("/")
def index():
    return render_template("index.html", max_mb=current_app.config["MAX_UPLOAD_MB"],
                           allowed=sorted(current_app.config["ALLOWED_EXTENSIONS"]))


@bp.post("/predict")
def predict():
    predictor = current_app.extensions["predictor"]
    if predictor is None:
        return render_error(503, "The model is not available right now. "
                                 "An administrator can see the reason at /health.")
    try:
        up = validate_upload(request.files.get("image"), current_app.config["ALLOWED_EXTENSIONS"],
                             current_app.config["MAX_CONTENT_LENGTH"])
    except UploadError as e:
        return render_error(e.status, e.message)

    try:
        pred = predictor.predict_bytes(up.raw, up.filename)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("prediction failed")
        return render_error(502, f"The model could not process this image ({type(exc).__name__}).")

    rec = PredictionRecord(filename=up.filename, file_sha256=up.sha256, file_size=len(up.raw),
                           width=up.width, height=up.height, label=pred.label,
                           confidence=pred.confidence, p_ai=pred.p_ai, threshold=pred.threshold,
                           model_name=pred.model_name, model_version=pred.model_version,
                           inference_mode=current_app.config["INFERENCE_MODE"], latency_ms=pred.latency_ms)
    db.session.add(rec)
    db.session.commit()
    current_app.logger.info("prediction id=%s file=%s label=%s p_ai=%.4f", rec.id, rec.filename,
                            rec.label, rec.p_ai)
    return render_template("result.html", r=rec, preview=thumbnail_data_uri(up.raw), caveat=CAVEAT)


@bp.get("/result/<int:rid>")
def result(rid):
    rec = db.get_or_404(PredictionRecord, rid)
    return render_template("result.html", r=rec, preview=None, caveat=CAVEAT)


@bp.get("/history")
def history():
    page = request.args.get("page", 1, type=int)
    label = request.args.get("label")
    q = PredictionRecord.query.order_by(PredictionRecord.created_at.desc())
    if label in ("AI-generated", "Real"):
        q = q.filter_by(label=label)
    pages = q.paginate(page=page, per_page=20, error_out=False)
    total = PredictionRecord.query.count()
    n_ai = PredictionRecord.query.filter_by(label="AI-generated").count()
    return render_template("history.html", pages=pages, label=label, total=total, n_ai=n_ai)


@bp.get("/api/history")
def api_history():
    limit = min(request.args.get("limit", 50, type=int), 500)
    rows = PredictionRecord.query.order_by(PredictionRecord.created_at.desc()).limit(limit).all()
    return jsonify([r.to_dict() for r in rows])


@bp.get("/health")
def health():
    cfg = current_app.config
    predictor = current_app.extensions["predictor"]
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    model = {"loaded": predictor is not None, "mode": cfg["INFERENCE_MODE"],
             "error": current_app.extensions["predictor_error"]}
    if predictor is not None:
        if cfg["INFERENCE_MODE"] == "remote":
            try:
                remote = predictor.health()
                model.update(remote_status=remote.get("status"))
            except Exception as exc:  # noqa: BLE001
                model.update(loaded=False, error=f"remote health failed: {exc}")
        model.update(name=predictor.model_name, version=predictor.model_version,
                     threshold=predictor.threshold)
    ok = db_ok and model["loaded"]
    body = {"status": "ok" if ok else "degraded", "database": "ok" if db_ok else "down", "model": model,
            "debug": current_app.debug,
            "database_backend": db.engine.url.get_backend_name()}
    return jsonify(body), 200 if ok else 503


def render_error(status: int, message: str):
    titles = {400: "Bad request", 404: "Not found", 413: "File too large", 415: "Unsupported file type",
              422: "Unreadable image", 500: "Server error", 502: "Model error", 503: "Service unavailable"}
    return render_template("error.html", status=status, title=titles.get(status, "Error"),
                           message=message), status


def register_error_handlers(app):
    @app.errorhandler(HTTPException)
    def http_error(e):
        if e.code == 413:
            return render_error(413, f"File is larger than the {app.config['MAX_UPLOAD_MB']:.0f} MB limit.")
        return render_error(e.code or 500, e.description)

    @app.errorhandler(Exception)
    def unhandled(e):
        app.logger.exception("unhandled error")
        return render_error(500, "Something went wrong on our side. The error has been logged.")


# keep Pillow's decompression-bomb guard at a sane limit for a web service
Image.MAX_IMAGE_PIXELS = 40_000_000
