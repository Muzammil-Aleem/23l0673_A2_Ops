# VisionGuard — AI vs Real image classifier (23l0673_A2_Ops)

![CI](https://github.com/<github-user>/23l0673_A2_Ops/actions/workflows/ci.yml/badge.svg)

VisionGuard tells whether an uploaded image is **AI-generated** or **real**. A TensorFlow/Keras
model (baseline CNN + MobileNetV2/EfficientNetB0 transfer learning) is tracked with MLflow on
DagsHub, data is versioned with DVC, the model is served by a Flask app that logs every request
to PostgreSQL, everything runs with Docker Compose, and the model API is deployed to the cloud.

| | Link |
|---|---|
| GitHub | `https://github.com/<github-user>/23l0673_A2_Ops` |
| DagsHub (code, DVC data, MLflow) | `https://dagshub.com/<dagshub-user>/23l0673_A2_Ops` |
| Model API (cloud) | `https://<service-url>/health` |

## Layout

```
visionguard/      shared code: config, preprocessing (one function for train AND serve), inference core
training/         audit, EDA, splits, tf.data pipeline, models, train, evaluate, tune, register, compare
labs/             tool labs: env check, DVC toy, Keras Tuner toy, MLflow toy, Docker warm-up, GHA toy repo
webapp/           Flask app: upload / result / history / health, validation, SQLAlchemy DB trail
model_api/        cloud model API (Flask + API key) and its Dockerfile
scripts/          DVC tracking, model fetch, compose operations, Cloud Run deploy, sample sender
tests/            pytest suite (runs without TensorFlow, uses a fake predictor)
docs/             RUNBOOK.md (step by step), REPORT.md (report draft), Postman collection
```

## Quick start

```powershell
# training environment (Python 3.11)
py -3.11 -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # fill in DagsHub + DB values

python -m training.download_data
python -m training.audit; python -m training.eda; python -m training.make_splits
python -m training.train --model mobilenetv2

# serve locally (debug off)
python scripts/fetch_model.py       # champion -> models/champion.keras
python run.py                       # http://127.0.0.1:5000

# whole stack in Docker
docker compose up -d --build        # http://localhost:8000
```

Full, ordered instructions for every task are in [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Configuration

All settings come from environment variables; see [.env.example](.env.example). `.env` is
git-ignored and docker-ignored.

| Variable | Meaning |
|---|---|
| `MODEL_PATH`, `MODEL_META` | champion `.keras` file and its JSON (threshold, image size, version) |
| `THRESHOLD` | optional override of the decision threshold |
| `DATABASE_URL` | SQLAlchemy URL; empty means local SQLite |
| `INFERENCE_MODE` | `local` (model in-process) or `remote` (call the cloud API) |
| `MODEL_API_URL`, `MODEL_API_KEY` | cloud model API endpoint and secret key |
| `MLFLOW_TRACKING_URI/USERNAME/PASSWORD` | DagsHub MLflow server and token |

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) installs dependencies, runs `ruff` and
`pytest`, then builds the Docker image on every push.
