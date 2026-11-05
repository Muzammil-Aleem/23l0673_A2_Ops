"""Task 1.3 — visual EDA. Needs reports/audit.csv (run training.audit first).
Outputs reports/figures/eda_samples.png and eda_histograms.png.
Run:  python -m training.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from visionguard.config import DATA_RAW, FIGURES, REPORTS, SEED  # noqa: E402


def sample_grid(df, per_class=8):
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(classes), per_class, figsize=(2 * per_class, 2.3 * len(classes)))
    for r, cls in enumerate(classes):
        sub = df[df["class"] == cls].sample(per_class, random_state=SEED)
        for c, (_, row) in enumerate(sub.iterrows()):
            ax = axes[r, c]
            with Image.open(DATA_RAW / row.path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{row.width}x{row.height}", fontsize=8)
            ax.axis("off")
        axes[r, 0].text(-0.15, 0.5, cls, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=12, weight="bold")
    fig.suptitle("Random samples per class (title = original width x height)")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_samples.png", dpi=120)
    plt.close(fig)


def histograms(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, label in zip(axes, ["width", "height", "file_size"],
                              ["width (px)", "height (px)", "file size (KB)"]):
        for cls, sub in df.groupby("class"):
            vals = sub[col] / 1024 if col == "file_size" else sub[col]
            vals = vals.clip(upper=np.percentile(vals, 99))  # keep extreme outliers from flattening the plot
            ax.hist(vals, bins=40, alpha=0.55, label=f"{cls} (median {np.median(vals):.0f})")
        ax.set_xlabel(label + " — clipped at 99th pct")
        ax.set_ylabel("images")
        ax.legend()
    fig.suptitle("Image dimension and file-size distributions per class")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_histograms.png", dpi=120)
    plt.close(fig)

    # aspect ratio matters for the resize decision in Task 2.1
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, sub in df.groupby("class"):
        ax.hist((sub.width / sub.height).clip(0.3, 3), bins=40, alpha=0.55, label=cls)
    ax.set_xlabel("aspect ratio (width / height)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_aspect_ratio.png", dpi=120)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPORTS / "audit.csv")
    df = df[df.readable & ~df.is_duplicate]
    sample_grid(df)
    histograms(df)
    print("Saved figures to", FIGURES)


if __name__ == "__main__":
    main()
