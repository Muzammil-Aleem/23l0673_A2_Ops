"""Task 3.6 — register a run's model in the MLflow Model Registry and (optionally) give it an alias.

  python -m training.register --run-id <older_run>                    # -> version 1
  python -m training.register --run-id <best_run> --alias champion     # -> version 2 = champion

The decision threshold and image size are stored as version tags so whoever loads
the model also knows how to use it.
"""
import argparse

import mlflow
from mlflow.tracking import MlflowClient

from training.tracking import setup_mlflow
from visionguard.config import REGISTERED_MODEL_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--alias", default=None)
    ap.add_argument("--name", default=REGISTERED_MODEL_NAME)
    a = ap.parse_args()

    setup_mlflow()
    client = MlflowClient()
    run = client.get_run(a.run_id)
    mv = mlflow.register_model(f"runs:/{a.run_id}/model", a.name)
    for key in ("decision_threshold", "img_size"):
        if key in run.data.params:
            client.set_model_version_tag(a.name, mv.version, key, run.data.params[key])
    client.set_model_version_tag(a.name, mv.version, "val_roc_auc", str(run.data.metrics.get("val_roc_auc")))
    print(f"Registered {a.name} version {mv.version} from run {a.run_id}")

    if a.alias:
        try:
            client.set_registered_model_alias(a.name, a.alias, mv.version)
            print(f"Alias '{a.alias}' -> version {mv.version}")
        except Exception as exc:  # noqa: BLE001 - very old registry servers lack aliases
            print("Alias not supported by this server, falling back to stage 'Production':", exc)
            client.transition_model_version_stage(a.name, mv.version, "Production", archive_existing_versions=True)


if __name__ == "__main__":
    main()
