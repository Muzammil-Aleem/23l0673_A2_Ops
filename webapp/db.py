"""Task 5 — database trail. One row per successful prediction answers:
what was decided (label, p_ai, confidence), when (created_at, UTC), by which model
(name, version, local/remote) and how confidently. Works with SQLite (local dev) and
PostgreSQL (Compose / cloud) through the same SQLAlchemy model."""
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictionRecord(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True,
                           default=lambda: datetime.now(timezone.utc))
    filename = db.Column(db.String(255), nullable=False)  # sanitised name only — the image is never stored
    file_sha256 = db.Column(db.String(64), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    label = db.Column(db.String(32), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    p_ai = db.Column(db.Float, nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(128), nullable=False)
    model_version = db.Column(db.String(32), nullable=False)
    inference_mode = db.Column(db.String(16), nullable=False)
    latency_ms = db.Column(db.Float)

    def to_dict(self):
        return {c.name: (getattr(self, c.name).isoformat() if c.name == "created_at" else getattr(self, c.name))
                for c in self.__table__.columns}
