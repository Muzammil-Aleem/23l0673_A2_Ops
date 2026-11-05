"""Bonus — Grad-CAM for 4 correct and 4 wrong test predictions.
Run:  python -m training.gradcam --run reports/runs/<run_name>
"""
import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from visionguard.config import DATA_RAW, FIGURES, MODELS_DIR, SEED  # noqa: E402
from visionguard.preprocess import preprocess_path  # noqa: E402


def gradcam(model, img: np.ndarray, explain_ai: bool) -> np.ndarray:
    """Walk the (linear) model layer by layer, grab the feature maps that enter the
    global-average-pool, and weight them by the gradient of the class score."""
    x = tf.convert_to_tensor(img[None])
    with tf.GradientTape() as tape:
        t, feats = x, None
        for layer in model.layers[1:]:
            if layer.name == "gap":
                feats = t
                tape.watch(feats)
            t = layer(t, training=False)
        score = t[:, 0] if explain_ai else 1 - t[:, 0]
    grads = tape.gradient(score, feats)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return tf.image.resize(cam[..., None], img.shape[:2]).numpy()[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads((run / "metadata.json").read_text())
    model = keras.models.load_model(MODELS_DIR / f"{meta['run_name']}.keras")
    df = pd.read_csv(run / "test_predictions.csv")
    right = df[df.pred == df.label].sample(4, random_state=SEED)
    wrong = df[df.pred != df.label].head(4)
    picks = pd.concat([right.assign(kind="correct"), wrong.assign(kind="wrong")])

    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.4))
    for i, (_, r) in enumerate(picks.iterrows()):
        img = preprocess_path(DATA_RAW / r.path, meta["img_size"])
        cam = gradcam(model, img, explain_ai=bool(r.pred))
        axes[0, i].imshow(img.astype("uint8"))
        axes[0, i].set_title(f"{r.kind}: true {'AI' if r.label else 'real'}\np(AI)={r.p_ai:.2f}", fontsize=9)
        axes[1, i].imshow(img.astype("uint8"))
        axes[1, i].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, i].set_title(f"evidence for '{'AI' if r.pred else 'real'}'", fontsize=9)
        for ax in axes[:, i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "gradcam.png", dpi=110)
    print("saved", FIGURES / "gradcam.png")


if __name__ == "__main__":
    main()
