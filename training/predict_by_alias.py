"""Task 3.6 — separate small script: load the model by alias from the registry and classify one image.
Run:  python -m training.predict_by_alias path/to/image.jpg [--alias champion]
"""
import argparse

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from training.tracking import setup_mlflow
from visionguard.config import REGISTERED_MODEL_NAME
from visionguard.preprocess import preprocess_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--alias", default="champion")
    a = ap.parse_args()

    setup_mlflow()
    mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, a.alias)
    threshold = float(mv.tags.get("decision_threshold", 0.5))
    img_size = int(mv.tags.get("img_size", 224))
    uri = f"models:/{REGISTERED_MODEL_NAME}@{a.alias}"
    print(f"Loading {uri} (version {mv.version}, run {mv.run_id}) threshold={threshold:.3f}")
    model = mlflow.tensorflow.load_model(uri)

    x = preprocess_path(a.image, img_size)[None]
    p_ai = float(np.ravel(model.predict(x, verbose=0))[0])
    label = "AI-generated" if p_ai >= threshold else "Real"
    conf = p_ai if label == "AI-generated" else 1 - p_ai
    print(f"{a.image}: {label}  p(AI)={p_ai:.4f}  confidence={conf:.1%}")


if __name__ == "__main__":
    main()
