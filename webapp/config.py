"""Task 4.4 — every setting comes from environment variables (12-factor style), so
the same image runs on a laptop, in Compose and in the cloud without code changes,
and secrets never live in the source."""
import os


def _bool(name, default=False):
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "").strip() or "sqlite:///visionguard.db"
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}
    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
    MAX_CONTENT_LENGTH = int(MAX_UPLOAD_MB * 1024 * 1024)  # Flask rejects bigger bodies with 413
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}

    INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")  # local | remote
    MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
    MODEL_META = os.getenv("MODEL_META")
    THRESHOLD = os.getenv("THRESHOLD")  # optional override of the registry threshold
    MODEL_API_URL = os.getenv("MODEL_API_URL")  # remote mode: https://<cloud-host>/predict
    MODEL_API_KEY = os.getenv("MODEL_API_KEY")
    MODEL_API_TIMEOUT = float(os.getenv("MODEL_API_TIMEOUT", "30"))
    LOAD_MODEL = _bool("LOAD_MODEL", True)  # tests set this to false and inject a fake
