"""Task 2.4 / 2.5 helpers: threshold choice, metrics and figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score, roc_curve)

# Product decision (Task 2.5): calling a REAL photo fake is the worse error — it accuses a
# real person/journalist of faking evidence and destroys trust in the tool. So the
# threshold is the LOWEST one whose false-positive rate on validation is <= MAX_FPR
# (i.e. catch as many AI images as possible while keeping false accusations rare).
MAX_FPR = 0.10


def choose_threshold(y_val, p_val, max_fpr: float = MAX_FPR) -> dict:
    fpr, tpr, thr = roc_curve(y_val, p_val)
    ok = np.where(fpr <= max_fpr)[0]
    i = ok[np.argmax(tpr[ok])]  # best recall among thresholds that respect the FPR cap
    t = float(min(thr[i], 1.0))
    # reference points to report alongside
    youden = float(thr[np.argmax(tpr - fpr)])
    grid = np.linspace(0.05, 0.95, 91)
    f1_best = float(grid[np.argmax([f1_score(y_val, p_val >= g, zero_division=0) for g in grid])])
    return {"threshold": t, "val_fpr_at_threshold": float(fpr[i]), "val_tpr_at_threshold": float(tpr[i]),
            "youden_threshold": min(youden, 1.0), "f1_best_threshold": f1_best}


def compute_metrics(y, p, threshold: float) -> dict:
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p),
            "false_positive_rate": fp / max(fp + tn, 1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def plot_confusion(y, p, threshold, path, title=""):
    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["real", "AI"])
    ax.set_yticks([0, 1], ["real", "AI"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\nthreshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(y, p, path, threshold=None, title=""):
    fpr, tpr, thr = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    if threshold is not None:
        i = np.argmin(np.abs(thr - threshold))
        ax.scatter(fpr[i], tpr[i], color="red", zorder=3, label=f"chosen threshold {threshold:.2f}")
    ax.set_xlabel("false positive rate (real called AI)")
    ax.set_ylabel("true positive rate (AI caught)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history: dict, path, title="", stage_boundary: int | None = None) -> int:
    """Loss and accuracy curves; marks the best epoch (lowest val_loss). Returns best epoch (1-based)."""
    best = int(np.argmin(history["val_loss"]))
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[key], label=f"train {key}")
        ax.plot(epochs, history[f"val_{key}"], label=f"val {key}")
        ax.axvline(best + 1, color="red", ls=":", label=f"best epoch {best + 1}")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="grey", ls="--", label="start fine-tuning")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return best + 1
