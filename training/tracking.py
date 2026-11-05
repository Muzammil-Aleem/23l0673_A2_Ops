"""MLflow / DagsHub helpers.

Credentials are NEVER written in code. MLflow reads these environment variables
itself (put them in .env, which is git-ignored):
  MLFLOW_TRACKING_URI       https://dagshub.com/<user>/<repo>.mlflow   (unset -> local ./mlruns)
  MLFLOW_TRACKING_USERNAME  your DagsHub username
  MLFLOW_TRACKING_PASSWORD  a DagsHub access token
"""
import os
import subprocess
import sys

import keras
import mlflow
import yaml
from dotenv import load_dotenv

from visionguard.config import PROJECT_NAME, ROOT


def setup_mlflow(experiment: str | None = None) -> str:
    load_dotenv(ROOT / ".env")
    uri = os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or PROJECT_NAME)
    print("MLflow tracking URI:", uri)
    return uri


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def dvc_data_version() -> str:
    """md5 of the data/raw directory as recorded in data/raw.dvc (the DVC pointer file)."""
    pointer = ROOT / "data" / "raw.dvc"
    if not pointer.exists():
        return "untracked"
    meta = yaml.safe_load(pointer.read_text())
    return meta["outs"][0]["md5"]


def pip_freeze(path) -> None:
    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    path.write_text(out)


class MlflowEpochLogger(keras.callbacks.Callback):
    """Manual per-epoch logging. `offset` continues the step count across the two
    transfer-learning stages so the curves join up in the MLflow UI."""

    def __init__(self, offset: int = 0, prefix: str = ""):
        super().__init__()
        self.offset = offset
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        lr = self.model.optimizer.learning_rate
        logs["learning_rate"] = float(keras.ops.convert_to_numpy(lr))
        mlflow.log_metrics({self.prefix + k: float(v) for k, v in logs.items()},
                           step=self.offset + epoch)
