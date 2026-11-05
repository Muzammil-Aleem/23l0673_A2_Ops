"""VisionGuard Flask application factory."""
import io
import logging
import time

from flask import Flask, Request
from sqlalchemy.exc import OperationalError

from webapp.config import Config
from webapp.db import db


class InMemoryRequest(Request):
    """Keep uploaded files in RAM. Werkzeug normally spools uploads >500 KB to a temp
    file on disk; the assignment says the uploaded image must never touch the disk."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def build_predictor(cfg):
    if cfg["INFERENCE_MODE"] == "remote":
        from webapp.remote import RemotePredictor
        return RemotePredictor(cfg["MODEL_API_URL"], cfg["MODEL_API_KEY"], cfg["MODEL_API_TIMEOUT"])
    from visionguard.inference import KerasPredictor
    thr = cfg["THRESHOLD"]
    return KerasPredictor(cfg["MODEL_PATH"], cfg["MODEL_META"], float(thr) if thr else None)


def init_db(app, retries=10):
    """Create tables; retry briefly in case the database container is still starting."""
    for attempt in range(1, retries + 1):
        try:
            with app.app_context():
                db.create_all()
            return
        except OperationalError as exc:
            app.logger.warning("database not ready (attempt %d/%d): %s", attempt, retries, exc.orig)
            time.sleep(2)
    raise RuntimeError("could not connect to the database — check DATABASE_URL / POSTGRES_PASSWORD")


def create_app(overrides: dict | None = None, predictor=None) -> Flask:
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config.from_object(Config)
    app.config.update(overrides or {})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db.init_app(app)
    init_db(app)

    app.extensions["predictor"] = predictor
    app.extensions["predictor_error"] = None
    if predictor is None and app.config["LOAD_MODEL"]:
        try:
            app.extensions["predictor"] = build_predictor(app.config)  # loaded ONCE at start-up
            app.logger.info("predictor ready: mode=%s", app.config["INFERENCE_MODE"])
        except Exception as exc:  # noqa: BLE001 - keep serving pages, /health reports the problem
            app.extensions["predictor_error"] = f"{type(exc).__name__}: {exc}"
            app.logger.error("MODEL NOT LOADED: %s", app.extensions["predictor_error"])

    from webapp.routes import bp, register_error_handlers
    app.register_blueprint(bp)
    register_error_handlers(app)
    return app
