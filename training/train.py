"""Tasks 2.2-2.5 and 3.3-3.5 — train one VisionGuard model and track it in MLflow.

Examples (each is one MLflow run):
  python -m training.train --model cnn --lr 1e-3                       # baseline CNN, manual logging
  python -m training.train --model cnn --lr 1e-3 --autolog             # same, MLflow autologging
  python -m training.train --model mobilenetv2                          # transfer: frozen -> last block
  python -m training.train --model mobilenetv2 --lr 3e-4                # one change: learning rate
  python -m training.train --model mobilenetv2 --augment none           # one change: augmentation
  python -m training.train --model mobilenetv2 --fine-tune-epochs 0     # one change: never unfreeze
  python -m training.train --model mobilenetv2 --img-size 160           # one change: image size
  python -m training.train --model efficientnetb0                       # one change: backbone
"""
import argparse
import json
import os
import time

import keras
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from training import evaluation as ev
from training.models import build_cnn, build_transfer, trainable_layer_table, unfreeze
from training.pipeline import labels_of, make_dataset, read_split
from training.tracking import MlflowEpochLogger, dvc_data_version, git_commit, pip_freeze, setup_mlflow
from visionguard.config import MODELS_DIR, PROJECT_NAME, REPORTS, SEED, SPLITS_DIR


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mobilenetv2", choices=["cnn", "mobilenetv2", "efficientnetb0"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="LR for the CNN or for stage 1 (frozen backbone)")
    p.add_argument("--fine-tune-lr", type=float, default=1e-5, help="much smaller LR for stage 2")
    p.add_argument("--epochs", type=int, default=30, help="max epochs (CNN) / stage-1 epochs")
    p.add_argument("--fine-tune-epochs", type=int, default=15, help="0 = keep the backbone frozen")
    p.add_argument("--unfreeze-layers", type=int, default=None, help="default: the last block")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--augment", default="medium", choices=["none", "light", "medium", "strong"])
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--autolog", action="store_true", help="use mlflow.tensorflow.autolog instead of manual logging")
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)  # seeds python `random`, numpy and tensorflow together
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # noqa: BLE001
        pass


def compile_model(model, lr):
    # binary cross-entropy: the correct likelihood for one sigmoid output (0=real, 1=AI)
    # Adam: adaptive per-parameter steps, robust default for both CNN and fine-tuning
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc"),
                           keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")])


def make_callbacks(ckpt_path, patience, extra):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True,
                                        save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(patience // 2, 2),
                                          min_lr=1e-7, verbose=1),
        *extra,
    ]


def layer_table(model) -> pd.DataFrame:
    return pd.DataFrame([{"layer": l.name, "type": type(l).__name__,
                          "output_shape": str(tuple(l.output.shape)), "params": l.count_params()}
                         for l in model.layers])


def class_weights(y):
    counts = np.bincount(y, minlength=2)
    return {i: len(y) / (2 * c) for i, c in enumerate(counts)}


def predict(model, split, img_size, batch_size):
    ds = make_dataset(split, img_size, batch_size, augment=None, shuffle=False)
    return model.predict(ds, verbose=0).ravel()


def main(argv=None):
    args = parse_args(argv)
    set_seeds(args.seed)
    if args.dropout is None:
        args.dropout = 0.4 if args.model == "cnn" else 0.3
    augment = None if args.augment == "none" else args.augment
    stamp = time.strftime("%m%d-%H%M")
    run_name = args.run_name or f"{args.model}_lr{args.lr:g}_aug-{args.augment}_img{args.img_size}_{stamp}"
    out = REPORTS / "runs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    setup_mlflow()
    if args.autolog:
        mlflow.tensorflow.autolog(log_models=True, log_datasets=False, checkpoint=False)

    train_ds = make_dataset("train", args.img_size, args.batch_size, augment=augment)
    val_ds = make_dataset("val", args.img_size, args.batch_size)
    y_train = labels_of("train")
    cw = class_weights(y_train)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"git_commit": git_commit(), "dvc_data_version": dvc_data_version(),
                         "seed": args.seed, "model_type": args.model, "project": PROJECT_NAME,
                         "logging": "autolog" if args.autolog else "manual"})
        mlflow.log_params({"n_train": len(y_train), "n_val": len(labels_of("val")),
                           "n_test": len(labels_of("test")), "class_weight": json.dumps(cw)})
        if not args.autolog:
            mlflow.log_params({k: v for k, v in vars(args).items() if k not in ("autolog", "run_name")})

        ckpt = out / "best.weights.h5"
        if args.model == "cnn":
            model = build_cnn(args.img_size, args.dropout)
        else:
            model = build_transfer(args.model, args.img_size, args.dropout)
        compile_model(model, args.lr)
        model.summary(print_fn=lambda s, **_: print(s))
        layer_table(model).to_csv(out / "layer_table.csv", index=False)
        with open(out / "model_summary.txt", "w", encoding="utf-8") as f:
            model.summary(print_fn=lambda s, **_: f.write(s + "\n"))
        stage1_trainable = trainable_layer_table(model)
        pd.DataFrame(stage1_trainable, columns=["layer", "trainable_params"]).to_csv(
            out / "trainable_stage1.csv", index=False)

        extra = [] if args.autolog else [MlflowEpochLogger()]
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=cw,
                       callbacks=make_callbacks(ckpt, args.patience, extra), verbose=2)
        model.load_weights(ckpt)  # best epoch of stage 1
        history = {k: list(v) for k, v in h1.history.items()}
        boundary = None

        if args.model != "cnn" and args.fine_tune_epochs > 0:
            boundary = len(history["loss"])
            trainable = unfreeze(model, args.model, args.unfreeze_layers)
            print(f"Stage 2: {len(trainable)} backbone layers trainable:", trainable)
            pd.DataFrame(trainable_layer_table(model), columns=["layer", "trainable_params"]).to_csv(
                out / "trainable_stage2.csv", index=False)
            compile_model(model, args.fine_tune_lr)  # recompile so the trainable flags take effect
            extra = [] if args.autolog else [MlflowEpochLogger(offset=boundary)]
            h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, class_weight=cw,
                           callbacks=make_callbacks(ckpt, args.patience, extra), verbose=2)
            model.load_weights(ckpt)
            for k, v in h2.history.items():
                history.setdefault(k, []).extend(v)
        if "learning_rate" in history:
            history.pop("learning_rate")

        best_epoch = ev.plot_history(history, out / "curves.png", f"{run_name}", boundary)
        if boundary:  # separate figures for the two stages (Task 2.3)
            ev.plot_history({k: v[:boundary] for k, v in history.items()}, out / "curves_stage1.png",
                            "Stage 1: frozen backbone")
            ev.plot_history({k: v[boundary:] for k, v in history.items()}, out / "curves_stage2.png",
                            "Stage 2: last block unfrozen")
        print(f"Best epoch (lowest val_loss): {best_epoch}  val_loss={min(history['val_loss']):.4f}")

        # ---- threshold from validation, then the test set exactly once ----
        y_val, p_val = labels_of("val"), predict(model, "val", args.img_size, args.batch_size)
        thr = ev.choose_threshold(y_val, p_val)
        val_m = ev.compute_metrics(y_val, p_val, thr["threshold"])
        y_test, p_test = labels_of("test"), predict(model, "test", args.img_size, args.batch_size)
        test_m = ev.compute_metrics(y_test, p_test, thr["threshold"])
        test_m05 = ev.compute_metrics(y_test, p_test, 0.5)
        ev.plot_confusion(y_test, p_test, thr["threshold"], out / "confusion_matrix.png", f"{args.model} — test")
        ev.plot_roc(y_test, p_test, out / "roc_curve.png", thr["threshold"], f"{args.model} — test ROC")

        test_df = read_split("test")[["path", "label"]].assign(p_ai=p_test)
        test_df["pred"] = (test_df.p_ai >= thr["threshold"]).astype(int)
        test_df.to_csv(out / "test_predictions.csv", index=False)

        mlflow.log_metrics({"best_epoch": best_epoch, "best_val_loss": float(min(history["val_loss"])),
                            **{f"val_{k}": float(v) for k, v in val_m.items()},
                            **{f"test_{k}": float(v) for k, v in test_m.items()},
                            **{f"test_{k}_at_0.5": float(v) for k, v in test_m05.items()
                               if k in ("accuracy", "f1", "precision", "recall")},
                            **{k: v for k, v in thr.items()}})
        mlflow.log_param("decision_threshold", round(thr["threshold"], 4))

        # ---- save + artifacts ----
        model_path = MODELS_DIR / f"{run_name}.keras"
        model.save(model_path)
        meta = {"run_id": run.info.run_id, "run_name": run_name, "model_type": args.model,
                "img_size": args.img_size, "threshold": thr["threshold"], "best_epoch": best_epoch,
                "test_metrics": test_m}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        pip_freeze(out / "requirements-freeze.txt")
        mlflow.log_artifacts(str(out), artifact_path="reports")
        mlflow.log_artifacts(str(SPLITS_DIR), artifact_path="splits")
        mlflow.log_artifact(str(out / "requirements-freeze.txt"), artifact_path="environment")
        mlflow.log_artifact(str(model_path), artifact_path="keras_model")
        if not args.autolog:
            sample = next(iter(val_ds))[0][:1].numpy()
            sig = mlflow.models.infer_signature(sample, model.predict(sample, verbose=0))
            mlflow.tensorflow.log_model(model, artifact_path="model", signature=sig)

        print(json.dumps({"run_id": run.info.run_id, "threshold": thr, "test": test_m}, indent=2, default=float))
    return meta


if __name__ == "__main__":
    main()
