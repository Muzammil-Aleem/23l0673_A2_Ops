"""Project-wide constants. Anything secret or deployment specific comes from
environment variables instead (see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS_DIR = ROOT / "models"

# Roll number 23l0673 — the assignment wants "<RollNumber>_A2_Ops" everywhere.
PROJECT_NAME = os.getenv("PROJECT_NAME", "23l0673_A2_Ops")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", f"{PROJECT_NAME}_VisionGuard")

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32

# Positive class = 1 = AI-generated. A folder whose name contains "ai" is AI,
# one containing "real" is real (Kaggle ships AiArtData/ and RealArt/).
CLASS_NAMES = {0: "real", 1: "ai_generated"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}


def folder_to_label(folder_name: str):
    name = folder_name.lower()
    if "real" in name:
        return 0
    if "ai" in name:
        return 1
    return None
