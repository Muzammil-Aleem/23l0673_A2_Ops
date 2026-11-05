"""Task 4.1 — inference core. Loads the champion Keras model ONCE (in __init__) and
classifies raw image bytes with the same preprocessing used in training.

Why not load per request: loading a .keras file rebuilds the graph and reads ~10-20 MB
of weights (several seconds with TensorFlow start-up). Doing that per request would
make every prediction slow, waste memory and CPU, and fall over under concurrent load.
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from visionguard.preprocess import preprocess_bytes

LABELS = {1: "AI-generated", 0: "Real"}


@dataclass
class Prediction:
    label: str
    confidence: float  # probability of the predicted label
    p_ai: float
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float

    def to_dict(self):
        return asdict(self)


class KerasPredictor:
    def __init__(self, model_path: str, meta_path: str | None = None, threshold: float | None = None):
        import keras  # imported lazily: the web app in "remote" mode never needs TensorFlow

        model_path = Path(model_path)
        meta_path = Path(meta_path) if meta_path else model_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        t0 = time.perf_counter()
        self.model = keras.models.load_model(model_path, compile=False)
        self.load_seconds = time.perf_counter() - t0
        self.img_size = int(meta.get("img_size", self.model.input_shape[1]))
        # env override > registry tag > 0.5
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self.model_name = meta.get("name", model_path.stem)
        self.model_version = str(meta.get("version", "local"))
        # warm-up: the first call traces the graph; do it now, not on the first user
        self.model.predict(np.zeros((1, self.img_size, self.img_size, 3), np.float32), verbose=0)

    def predict_bytes(self, raw: bytes, filename: str | None = None) -> Prediction:
        t0 = time.perf_counter()
        x = preprocess_bytes(raw, self.img_size)[None]
        p_ai = float(np.ravel(self.model(x, training=False))[0])
        is_ai = p_ai >= self.threshold
        return Prediction(label=LABELS[int(is_ai)], confidence=p_ai if is_ai else 1 - p_ai, p_ai=p_ai,
                          threshold=self.threshold, model_name=self.model_name,
                          model_version=self.model_version,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))


def predictor_from_env() -> KerasPredictor:
    thr = os.getenv("THRESHOLD")
    return KerasPredictor(os.getenv("MODEL_PATH", "models/champion.keras"), os.getenv("MODEL_META"),
                          float(thr) if thr else None)


if __name__ == "__main__":
    # console demo for the report: python -m visionguard.inference img1.jpg img2.jpg
    import sys

    pred = predictor_from_env()
    print(f"model loaded once in {pred.load_seconds:.2f}s, threshold={pred.threshold:.3f}")
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            print(path, "->", pred.predict_bytes(f.read()).to_dict())
