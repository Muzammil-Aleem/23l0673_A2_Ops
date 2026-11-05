"""Task 2.6 — show 10 misclassified test images with automatic hints that help you
write a hypothesis for each (size, aspect, greyscale, JPEG quality, confidence).
Run:  python -m training.error_analysis --run reports/runs/<run_name>
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from visionguard.config import DATA_RAW, REPORTS  # noqa: E402


def hints(path: Path) -> str:
    with Image.open(path) as im:
        w, h, mode, fmt = im.width, im.height, im.mode, im.format
        arr = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.float32)
    out = []
    if min(w, h) < 300:
        out.append("small/low-res")
    if max(w, h) / min(w, h) > 1.6:
        out.append(f"aspect {w / h:.2f} (crop lost content)")
    if np.abs(arr[..., 0] - arr[..., 1]).mean() < 3 and np.abs(arr[..., 1] - arr[..., 2]).mean() < 3:
        out.append("greyscale")
    if arr.std() < 35:
        out.append("low contrast/flat")
    sharp = np.abs(np.diff(arr.mean(-1), axis=0)).mean()
    if sharp < 4:
        out.append("blurry/smooth")
    if mode in ("P", "RGBA", "LA"):
        out.append(f"mode {mode}")
    out.append(f"{fmt} {w}x{h}")
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="reports/runs/<run_name>")
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    run = Path(a.run)
    df = pd.read_csv(run / "test_predictions.csv")
    thr = json.loads((run / "metadata.json").read_text())["threshold"]
    wrong = df[df.pred != df.label].copy()
    wrong["confidence_wrong"] = np.where(wrong.pred == 1, wrong.p_ai, 1 - wrong.p_ai)
    wrong = wrong.sort_values("confidence_wrong", ascending=False).head(a.n)
    print(f"{len(df[df.pred != df.label])} misclassified of {len(df)} (threshold {thr:.3f})")

    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.6 * rows))
    records = []
    for ax, (_, r) in zip(np.atleast_1d(axes).flat, wrong.iterrows()):
        p = DATA_RAW / r.path
        h = hints(p)
        with Image.open(p) as im:
            ax.imshow(im.convert("RGB"))
        truth = "AI" if r.label == 1 else "real"
        ax.set_title(f"true {truth}, p(AI)={r.p_ai:.2f}\n{h}", fontsize=8)
        ax.axis("off")
        records.append({"path": r.path, "true": truth, "p_ai": round(r.p_ai, 3), "hints": h,
                        "hypothesis": "<write yours>"})
    for ax in np.atleast_1d(axes).flat[len(wrong):]:
        ax.axis("off")
    fig.suptitle("Most confident misclassified test images")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "misclassified.png", dpi=110)
    pd.DataFrame(records).to_csv(REPORTS / "misclassified.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
