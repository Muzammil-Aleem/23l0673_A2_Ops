"""Task 3.5 — pull every run of the experiment from MLflow/DagsHub into one table and chart.
The best run is chosen on VALIDATION ROC-AUC (threshold-free, so it measures ranking
quality) with validation false-positive rate as tie-breaker; test metrics are shown
but not used for the choice, so the test set stays an unbiased final check.
Run:  python -m training.compare_runs
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from training.tracking import setup_mlflow  # noqa: E402
from visionguard.config import FIGURES, PROJECT_NAME, REPORTS  # noqa: E402

COLS = {"tags.mlflow.runName": "run", "tags.model_type": "model", "tags.logging": "logging",
        "params.lr": "lr", "params.augment": "augment", "params.img_size": "img",
        "params.fine_tune_epochs": "ft_epochs", "metrics.best_epoch": "best_ep",
        "metrics.val_roc_auc": "val_auc", "metrics.val_f1": "val_f1",
        "metrics.val_false_positive_rate": "val_fpr", "metrics.test_accuracy": "test_acc",
        "metrics.test_f1": "test_f1", "metrics.test_roc_auc": "test_auc", "run_id": "run_id"}


def main():
    setup_mlflow()
    df = mlflow.search_runs(experiment_names=[PROJECT_NAME])
    df = df[[c for c in COLS if c in df.columns]].rename(columns=COLS)
    df = df.dropna(subset=["val_auc"]).sort_values(["val_auc", "val_fpr"], ascending=[False, True])
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "run_comparison.csv", index=False)
    print(df.drop(columns=["run_id"]).round(4).to_markdown(index=False))
    best = df.iloc[0]
    print(f"\nBEST: {best.run}  run_id={best.run_id}  val_auc={best.val_auc:.4f}  test_auc={best.test_auc:.4f}")

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df) + 1.5))
    ax.barh(df.run, df.val_auc, label="val ROC-AUC")
    ax.barh(df.run, df.val_f1, alpha=0.6, height=0.4, label="val F1")
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.legend()
    ax.set_title("VisionGuard experiment comparison (sorted by val ROC-AUC)")
    fig.tight_layout()
    fig.savefig(FIGURES / "run_comparison.png", dpi=120)


if __name__ == "__main__":
    main()
