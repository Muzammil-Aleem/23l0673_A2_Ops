"""Vercel entrypoint: model files are downloaded during the build."""
import os
from pathlib import Path

root = Path(__file__).resolve().parent
os.environ["MODEL_PATH"] = str(root / "models" / "champion.keras")
os.environ["MODEL_META"] = str(root / "models" / "champion.json")
if not os.environ.get("MODEL_API_KEY"):
    raise RuntimeError("Set MODEL_API_KEY in Vercel environment variables")

from model_api.app import create_app  # noqa: E402

app = create_app()
