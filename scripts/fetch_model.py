"""Deployment plan for the model file (Task 3.6 note / 6.3): the serving code never
talks to DagsHub. Instead this script is run ONCE per release (by you or CI) to
download the current `champion` into models/champion.keras + models/champion.json.
That file is then mounted into the web container (Compose) or baked into the
model-API image (cloud).

Run:  python scripts/fetch_model.py [--alias champion]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from training.tracking import setup_mlflow  # noqa: E402
from visionguard.config import MODELS_DIR, REGISTERED_MODEL_NAME  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion")
    a = ap.parse_args()
    setup_mlflow()
    mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, a.alias)
    local = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_model"))
    src = next(local.glob("*.keras"))
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(src, MODELS_DIR / "champion.keras")
    meta = {"name": REGISTERED_MODEL_NAME, "version": mv.version, "alias": a.alias, "run_id": mv.run_id,
            "threshold": float(mv.tags.get("decision_threshold", 0.5)),
            "img_size": int(mv.tags.get("img_size", 224)), "source_file": src.name}
    (MODELS_DIR / "champion.json").write_text(json.dumps(meta, indent=2))
    print("Saved models/champion.keras", meta)


if __name__ == "__main__":
    main()
