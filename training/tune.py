"""Task 2.7 (apply) — tune learning rate, dropout and number of unfrozen layers for the
MobileNetV2 transfer model with Keras Tuner (random search, >= 8 trials).

Each trial: frozen head warm-up (3 epochs, lr 1e-3) -> unfreeze N layers -> fine-tune
with the trial's learning rate. Objective: val_auc. Every trial is also logged to MLflow
as a nested run so it is visible on DagsHub.
Run:  python -m training.tune --trials 10
"""
import argparse

import keras
import keras_tuner as kt
import mlflow
import pandas as pd

from training.models import build_transfer, unfreeze
from training.pipeline import make_dataset
from training.tracking import setup_mlflow
from training.train import compile_model, set_seeds
from visionguard.config import REPORTS, SEED

WARMUP_EPOCHS = 3


class TransferHyperModel(kt.HyperModel):
    def __init__(self, backbone="mobilenetv2", img_size=224):
        super().__init__()
        self.backbone, self.img_size = backbone, img_size

    def build(self, hp):
        model = build_transfer(self.backbone, self.img_size, dropout=hp.Float("dropout", 0.1, 0.6, step=0.1))
        hp.Float("lr", 1e-6, 1e-3, sampling="log")
        hp.Choice("unfrozen_layers", [0, 10, 20, 40, 60])
        return model

    def fit(self, hp, model, train_ds, validation_data=None, callbacks=None, **kw):
        compile_model(model, 1e-3)
        model.fit(train_ds, validation_data=validation_data, epochs=WARMUP_EPOCHS, verbose=0)
        n = hp.get("unfrozen_layers")
        if n > 0:
            unfreeze(model, self.backbone, n)
        compile_model(model, hp.get("lr"))
        cbs = list(callbacks or []) + [keras.callbacks.EarlyStopping("val_loss", patience=3,
                                                                     restore_best_weights=True)]
        hist = model.fit(train_ds, validation_data=validation_data, callbacks=cbs, verbose=0, **kw)
        best = max(hist.history["val_auc"])
        with mlflow.start_run(run_name=f"trial_{hp.get('lr'):.1e}_d{hp.get('dropout'):.1f}_u{n}", nested=True):
            mlflow.log_params(hp.values)
            mlflow.log_metric("val_auc", best)
            mlflow.log_metric("val_loss", min(hist.history["val_loss"]))
        return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    set_seeds(SEED)
    setup_mlflow()
    train_ds = make_dataset("train", augment="medium")
    val_ds = make_dataset("val")

    tuner = kt.RandomSearch(TransferHyperModel(), objective=kt.Objective("val_auc", "max"),
                            max_trials=a.trials, seed=SEED, directory="reports/_tuner",
                            project_name="visionguard", overwrite=True)
    with mlflow.start_run(run_name="keras_tuner_search"):
        mlflow.set_tag("logging", "tuner")
        tuner.search(train_ds, validation_data=val_ds, epochs=a.epochs)
        best = tuner.get_best_hyperparameters(1)[0]
        mlflow.log_params({f"best_{k}": v for k, v in best.values.items()})

    rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_auc": round(t.score, 4)}
            for t in tuner.oracle.get_best_trials(num_trials=a.trials)]
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "tuner_results.csv", index=False)
    print(df.to_markdown(index=False))
    # crude importance: how much the mean score moves across values of each hyperparameter
    for hp in ("lr", "dropout", "unfrozen_layers"):
        col = pd.qcut(df[hp], 3, duplicates="drop") if hp == "lr" else df[hp]
        spread = df.groupby(col, observed=True).val_auc.mean()
        print(f"\n{hp}: spread of mean val_auc = {spread.max() - spread.min():.4f}\n{spread.round(4)}")
    print("\nBest hyperparameters:", best.values)


if __name__ == "__main__":
    main()
