# VisionGuard runbook — do this in order

Project name everywhere: **23l0673_A2_Ops** (Compose: `23l0673_a2_ops`, Cloud Run: `vg-23l0673-a2-ops`).
📸 = take a screenshot for the report (terminal prompt with your username + date, or the browser address bar, must be visible).

---

## 0. One-time setup

1. **Python 3.11.** TensorFlow 2.19 publishes wheels for Python 3.9–3.12; this PC only has 3.14.
   `winget install Python.Python.3.11`
2. **Repo on GitHub + DagsHub.**
   ```powershell
   cd C:\Users\BD\Desktop\MlAss
   git init; git branch -M main
   git add .; git commit -m "VisionGuard skeleton"
   # create an EMPTY GitHub repo named 23l0673_A2_Ops, then:
   git remote add origin https://github.com/<github-user>/23l0673_A2_Ops.git
   git push -u origin main
   ```
   On dagshub.com: **Create → New Repository → Connect a repository → GitHub**, pick
   `23l0673_A2_Ops`. DagsHub mirrors the code and gives you DVC storage and an MLflow server.
3. **Token.** DagsHub → avatar → *Settings → Tokens* → copy the default token.
   On the repo page, **Remote → Experiments** shows the MLflow URI
   `https://dagshub.com/<user>/23l0673_A2_Ops.mlflow`.
4. `copy .env.example .env` and fill in `MLFLOW_TRACKING_*`, DB password, `SECRET_KEY`.
   📸 `git check-ignore -v .env` → shows `.gitignore:2:.env` (proof .env is ignored, Task 3.2).

## Task 1 — data

| Step | Command | Evidence |
|---|---|---|
| 1.1 | `py -3.11 -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt` → `python labs/env_check.py` | 📸 versions + GPU line. On native Windows TF has no GPU support (since 2.11): expect `GPUs visible: 0`. That is fine; training this dataset on CPU takes minutes. Use WSL2 if you want a GPU. |
| 1.2 | Kaggle → *Settings → Create New Token* → save `kaggle.json` in `%USERPROFILE%\.kaggle\`. `python -m training.download_data` then `python -m training.audit` | 📸 the printed tables; `reports/audit_summary.md` has them in Markdown |
| 1.3 | `python -m training.eda` | `reports/figures/eda_samples.png`, `eda_histograms.png`, `eda_aspect_ratio.png` |
| splits | `python -m training.make_splits` | train/val/test counts (commit `data/splits/*.csv` to Git) |
| 1.4 toy | run `labs/dvc_toy.ps1` block by block (set `DAGSHUB_USER`, `DAGSHUB_TOKEN`, `TOY_REPO`) | 📸 `git log`, `data.csv.dvc`, checkout of v1, `dvc push`, `dvc pull` |
| 1.4 apply | `scripts/dvc_track_dataset.ps1` (set `DAGSHUB_REPO=23l0673_A2_Ops`) | 📸 `data/raw.dvc`, `dvc push`, the DagsHub file browser showing `data/raw` as DVC-tracked |

## Task 2 — models (each command is one MLflow run)

```powershell
python -m training.pipeline                                   # 2.1 figure of one augmented batch
python -m training.train --model cnn --lr 1e-3                # 2.2 baseline CNN
python -m training.train --model mobilenetv2                  # 2.3 frozen -> last block, 2.4 callbacks
```
Each run writes `reports/runs/<run_name>/` with `model_summary.txt`, `layer_table.csv` (2.2 table),
`trainable_stage1.csv` / `trainable_stage2.csv` (2.3 list), `curves*.png` (best epoch marked, 2.4),
`confusion_matrix.png`, `roc_curve.png`, `metadata.json` (threshold, test metrics, 2.5).

```powershell
python -m training.error_analysis --run reports/runs/<best_run>   # 2.6 -> figures/misclassified.png
python labs/keras_tuner_toy.py                                     # 2.7 toy table
python -m training.tune --trials 10                                # 2.7 apply (takes a while on CPU)
python -m training.gradcam --run reports/runs/<best_run>           # bonus -> figures/gradcam.png
```

## Task 3 — MLflow / DagsHub

| Step | Command | Evidence |
|---|---|---|
| 3.1 | `python labs/mlflow_toy.py --lr 0.001` and `--lr 0.01`, then `mlflow ui --backend-store-uri ./mlruns` → http://127.0.0.1:5000 → experiment `toy_mnist` → tick both → **Compare** | 📸 both run pages + compare view |
| 3.2 | `python labs/mlflow_toy.py --lr 0.001 --remote` | 📸 DagsHub repo page, the toy run in DagsHub → Experiments |
| 3.3 | `python -m training.train --model cnn --lr 1e-3` (manual) and `python -m training.train --model cnn --lr 1e-3 --autolog --run-name cnn_autolog` | 📸 both run pages: Parameters, Metrics (charts), Tags |
| 3.4 | open the best run → **Artifacts** | 📸 `reports/` (confusion matrix, ROC), `splits/`, `environment/requirements-freeze.txt`, `keras_model/`, `model/` |
| 3.5 | at least 6 runs, each one change away from the default `mobilenetv2` run: `--lr 3e-4`, `--augment none`, `--fine-tune-epochs 0`, `--img-size 160`, `--model efficientnetb0` (+ the CNN). Then `python -m training.compare_runs` | 📸 DagsHub/MLflow compare view + `reports/figures/run_comparison.png` |
| 3.6 | `python -m training.register --run-id <second-best>` then `python -m training.register --run-id <best> --alias champion`, then `python -m training.predict_by_alias some_image.jpg` | 📸 registry page (2 versions, alias), terminal output |

Run IDs: shown at the end of each training run and in `reports/run_comparison.csv`.

## Task 4 — Flask app (local, debug off)

```powershell
python scripts/fetch_model.py                       # champion -> models/champion.keras + .json
python -m visionguard.inference real.jpg ai.png     # 4.1 console output for two known images
python run.py                                        # http://127.0.0.1:5000
```
- 4.2 📸 three failure pages: upload `notes.txt` (415 wrong type), an image > 5 MB (413 too
  large), and a text file renamed to `fake.jpg` (422 cannot be decoded).
- 4.3 📸 upload page, result page, history page.
- 4.4 📸 http://127.0.0.1:5000/health JSON (`"debug": false`).
- `pytest -q` and `ruff check .` must pass.

## Task 5 — database trail

Locally it is SQLite (`instance/visionguard.db`); in Compose it is PostgreSQL. Evidence:
📸 history page after several uploads, and the SQL query from `scripts/compose_ops.ps1`
(`docker compose exec db psql ...`).

## Task 6 — Docker

| Step | What | Evidence |
|---|---|---|
| 6.1 | `labs/docker_warmup.ps1` block by block | 📸 each action |
| 6.2 | `Dockerfile` | 📸 `docker compose exec web whoami` → `appuser` |
| 6.3 | `docker compose up -d --build`, `docker compose ps` | 📸 both `healthy`; `Test-NetConnection localhost -Port 5432` → False; `db:5432 reachable from web` |
| 6.4 | `docker images`, `docker history`, build `Dockerfile.unoptimized` | two sizes + largest layers |
| 6.5 | blocks in `scripts/compose_ops.ps1` | 📸 failure log, fix |
| 6.6 toy | new GitHub repo, copy the *contents* of `labs/gha_toy/` to its root, push (green), edit `test_add` to expect 5, push (red), revert (green) | 📸 red + green runs |
| 6.6 apply | push this repo; the CI workflow runs | 📸 green run, badge in README (replace `<github-user>`) |

## Task 7 — model API in the cloud (Google Cloud Run)

1. `python scripts/fetch_model.py` (model must be in `models/`).
2. Test the API image locally first:
   ```powershell
   docker build -f model_api/Dockerfile -t visionguard-api .
   docker run --rm -p 8080:8080 -e MODEL_API_KEY=test123 visionguard-api
   curl.exe -s http://localhost:8080/health
   curl.exe -s -H "X-API-Key: test123" -F "file=@some.jpg" http://localhost:8080/predict
   ```
3. Install the gcloud CLI, `gcloud auth login`, create a project (📸 console), enable billing
   (the free tier covers this workload; set a budget alert).
4. `$env:GCP_PROJECT="..."; $env:MODEL_API_KEY="<random>"; .\scripts\deploy_cloudrun.ps1`
   📸 each step: Artifact Registry image, Secret Manager secret, Cloud Run service page, URL.
5. **Postman:** import `docs/VisionGuard_API.postman_collection.json`, set `base_url` to the Cloud
   Run URL and `api_key`, run *Health*, *Predict* (choose a file), *without key* (401),
   *non-image* (422). 📸 each response + Cloud Run → *Logs* showing the `req=... label=...` lines.
6. Test from **another network** (phone hotspot / mobile data): open `<URL>/health` on the phone,
   or run Postman on the hotspot. 📸

*Alternative (pathway A, Oracle Always Free VM):* create an Ubuntu VM (Ampere A1 = Arm; TF
`tensorflow-cpu` has aarch64 wheels, or choose an x86 E2.1.Micro which is too small for TF), open
port 8080 in the VCN security list **and** in the VM (`sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT`),
install Docker, copy the repo + `models/`, `docker build -f model_api/Dockerfile -t visionguard-api .`,
`docker run -d --restart unless-stopped -p 8080:8080 -e MODEL_API_KEY=... visionguard-api`.

## Task 8 — Flask (localhost) → cloud API

1. In `.env`: `INFERENCE_MODE=remote`, `MODEL_API_URL=https://<cloud-run-url>/predict`,
   `MODEL_API_KEY=<same key>`.
2. `python run.py` → http://127.0.0.1:5000/health shows `"mode": "remote"` and the remote model
   version. 📸
3. Upload several images in the browser (📸 each result), or
   `python scripts/send_samples.py --n 8` (📸 terminal output).
4. 📸 History page — the `Mode` column says `remote`.
5. Optional: run the same thing in Compose with a light image:
   `WITH_TF=false` + `INFERENCE_MODE=remote` in `.env`, `docker compose up -d --build`.

## Final checklist

- `git rev-parse HEAD` → commit hash for the report front matter.
- Report PDF named `23l0673_VisionGuard.pdf` (draft in `docs/REPORT.md`).
- Links: GitHub, DagsHub, API URL.
