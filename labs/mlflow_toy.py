"""Tasks 3.1 + 3.2 — tiny Keras model on MNIST tracked with MLflow.

Local (3.1):   python labs/mlflow_toy.py --lr 0.001
               python labs/mlflow_toy.py --lr 0.01
               mlflow ui --backend-store-uri ./mlruns      -> http://127.0.0.1:5000, tick both runs, "Compare"
DagsHub (3.2): put MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD in .env, then
               python labs/mlflow_toy.py --lr 0.001 --remote
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--remote", action="store_true", help="log to DagsHub instead of ./mlruns")
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if a.remote:
        uri = os.environ["MLFLOW_TRACKING_URI"]  # never hard-coded; comes from .env
    else:
        uri = (ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("toy_mnist")
    keras.utils.set_random_seed(42)

    (x, y), (xt, yt) = keras.datasets.mnist.load_data()
    x, xt = x[:10000] / 255.0, xt / 255.0
    y = y[:10000]
    model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                              keras.layers.Dense(64, activation="relu"),
                              keras.layers.Dense(10, activation="softmax")])
    model.compile(keras.optimizers.Adam(a.lr), "sparse_categorical_crossentropy", metrics=["accuracy"])

    with mlflow.start_run(run_name=f"toy_lr{a.lr:g}"):
        mlflow.set_tags({"purpose": "tool-lab", "dataset": "mnist-10k"})
        mlflow.log_params({"lr": a.lr, "epochs": a.epochs, "hidden_units": 64, "batch_size": 32})
        hist = model.fit(x, y, validation_split=0.2, epochs=a.epochs, callbacks=[EpochLogger()], verbose=2)
        mlflow.log_metric("test_accuracy", model.evaluate(xt, yt, verbose=0)[1])

        fig, ax = plt.subplots()
        ax.plot(hist.history["loss"], label="train")
        ax.plot(hist.history["val_loss"], label="val")
        ax.set_title(f"loss, lr={a.lr}")
        ax.legend()
        with tempfile.TemporaryDirectory() as d:
            fig.savefig(Path(d) / "loss_curve.png")
            mlflow.log_artifact(str(Path(d) / "loss_curve.png"), artifact_path="plots")
        mlflow.tensorflow.log_model(model, artifact_path="model")
    print("logged to", uri)


if __name__ == "__main__":
    main()
