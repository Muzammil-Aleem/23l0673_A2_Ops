# VisionGuard — AI vs Real Image Classification and Deployment

**Name:** _______ **Roll number:** 23l0673 **Project:** 23l0673_A2_Ops
**GitHub:** https://github.com/<user>/23l0673_A2_Ops **DagsHub:** https://dagshub.com/<user>/23l0673_A2_Ops
**Live API URL:** https://<service-url> **Graded commit:** `<git rev-parse HEAD>`

**AI-assistant disclosure:** _State honestly which tools you used and for what (e.g. "Claude was
used to scaffold the code structure and draft explanations, which I reviewed, ran and edited")._

> Fill every `[ ]`, insert each 📸 screenshot with a caption ("Figure n: …"), and reference it in
> the text. Rewrite the explanations in your own words — they are a starting point.

---

## Chapter 1 — Data understanding and versioning

### 1.1 Environment
Isolated venv with Python 3.11 (TensorFlow does not ship wheels for every new Python release).
Versions: TensorFlow [ ], Keras [ ], MLflow [ ], DVC [ ]. GPU: [0] — TensorFlow dropped native
Windows GPU support after 2.10, so training ran on CPU; the dataset (~1k images) and a small
backbone make that practical. 📸 Figure 1.

### 1.2 Dataset audit
`training/audit.py` walks `data/raw`, takes the label from the folder name, and for every file
records class, file size, width, height, colour mode, format and SHA-256. Pixel data is fully
decoded (`im.load()`), not just the header, so truncated files are detected. Exact duplicates are
files sharing a SHA-256; duplicates appearing in both classes are flagged separately because their
label is contradictory.

| class | files | corrupt | duplicates |
|---|---|---|---|
| ai_generated | [ ] | [ ] | [ ] |
| real | [ ] | [ ] | [ ] |

Width/height/file-size min/median/max: paste from `reports/audit_summary.md`.
Consequence: corrupt files, extra duplicate copies and cross-class duplicates are removed **before**
splitting so the same picture cannot appear in both train and test (leakage).

### 1.3 Visual EDA
- *Figure: samples per class.* [2–3 sentences: e.g. AI images are often square, highly saturated,
  smooth "painterly" textures; real images include photos with text, watermarks, varied framing.]
- *Figure: width/height/file-size histograms.* [e.g. AI images cluster at generator sizes
  (512, 768, 1024); real images spread widely. File size differs by class, so the model could
  cheat on resolution/compression — one reason every image is resized to the same size.]
- *Figure: aspect ratio.* [Most AI images are 1:1, real ones vary; this motivated centre-crop
  instead of padding (see 2.1).]

### 1.4 DVC
Toy: `labs/dvc_toy.ps1` (📸 per step). The pointer file `data.csv.dvc` holds only the MD5 hash,
size and path of the data; the content lives in `.dvc/cache` and on the remote. Git versions the
small pointer, DVC fetches the matching content (`git checkout <old> -- data.csv.dvc; dvc checkout`).
Large data does not belong in Git because every version of every binary stays in history forever,
cloning becomes slow, diffs are meaningless for images, and GitHub rejects files over 100 MB.
Applied to `data/raw` with DagsHub storage as the remote; `data/raw.dvc` md5 = [ ] — this value is
logged as the `dvc_data_version` tag on every MLflow run.

## Chapter 2 — Model development

### 2.1 Input pipeline
`training/pipeline.py`: split CSV → `from_tensor_slices` → `map(load)` (bytes → shared
`visionguard.preprocess` function: decode, EXIF rotation, RGB, centre-crop, bilinear resize to
224×224, float 0–255) → `cache()` (decoded images stay in RAM; decoding is the slow step) →
`shuffle` (train only, seeded) → `batch(32)` → `map(augment)` (train only: horizontal flip,
±~22° rotation, zoom ±12 %, contrast ±20 %) → `prefetch(AUTOTUNE)` (overlaps data prep with
training). Augmentation is after `cache` so each epoch sees new variants, and never on val/test so
evaluation is deterministic.
**Image size 224:** the native ImageNet resolution of MobileNetV2/EfficientNetB0, so pretrained
filters see objects at the scale they learned.
**Aspect ratio:** centre-crop to a square then resize, so nothing is stretched. Squashing would
distort, and padding adds black bars that mostly real (non-square) images would get — a shortcut
the model could learn. Cost: edges of very wide/tall images are lost. 📸 augmented batch figure.

### 2.2 Baseline CNN
Rescaling(1/255) → 4 × [Conv3×3 (no bias) → BatchNorm → ReLU → MaxPool2] with 32/64/128/256
filters → GlobalAveragePooling → Dropout 0.4 → Dense(1, sigmoid).
- Conv: learns local patterns (edges → textures → artefacts); doubling filters as resolution halves
  keeps compute balanced.
- BatchNorm: stabilises/accelerates training; makes conv bias redundant.
- MaxPool: halves resolution, adds translation tolerance, grows the receptive field.
- GAP: one number per feature map — far fewer parameters than Flatten, less overfitting.
- Dropout: regularisation for a ~1k-image dataset.
- Sigmoid: one probability p(AI).
Loss: binary cross-entropy (the likelihood of a Bernoulli output). Optimiser: Adam lr 1e-3
(adaptive steps, reliable default). Metrics: accuracy, AUC (threshold-free), precision, recall;
class weights compensate the mild imbalance. 📸 model summary + `layer_table.csv`.

### 2.3 Transfer learning
MobileNetV2 (ImageNet, `include_top=False`). Preprocessing: `Rescaling(1/127.5, offset=-1)`
inside the model, identical to `mobilenet_v2.preprocess_input` — the backbone was trained on
inputs in [-1, 1]; feeding 0–255 or 0–1 would shift every activation and make the pretrained
filters useless. Head: GAP → Dropout → Dense 128 ReLU → Dropout → Dense 1 sigmoid.
Stage 1: backbone frozen, lr 1e-3, only the head trains. Stage 2: unfreeze the last block
(`block_16_*` + `Conv_1`), BatchNorm stays frozen and the backbone is called with
`training=False`, lr 1e-5 (100× smaller) so the pretrained weights are nudged, not destroyed.
📸 two curve figures + `trainable_stage1.csv` / `trainable_stage2.csv`.

### 2.4 Training discipline
`keras.utils.set_random_seed(42)` + op determinism; EarlyStopping (val_loss, patience 6, restore
best), ModelCheckpoint (best val_loss weights), ReduceLROnPlateau (×0.3 after 3 flat epochs).
Best epoch: CNN [ ], MobileNetV2 [ ] (red dotted line in the curves).

### 2.5 Evaluation
| model | acc | precision | recall | F1 | ROC-AUC | threshold |
|---|---|---|---|---|---|---|
| CNN | | | | | | |
| MobileNetV2 | | | | | | |

**Which error is worse?** Calling a real photo fake. A false "AI-generated" verdict accuses a
real person/journalist of fabricating evidence, and users stop trusting the tool. Missing a fake
is also bad, but the caveat on the result page already tells users the tool is not proof.
**Threshold:** chosen on the validation set as the lowest threshold whose false-positive rate
(real → AI) is ≤ 10 %, i.e. the highest AI recall under that cap. Youden and F1-optimal thresholds
are logged for comparison. The test set was then evaluated once at that threshold.
📸 confusion matrices + ROC curves.

### 2.6 Error analysis
📸 `figures/misclassified.png`; table from `reports/misclassified.csv` with one hypothesis per
image (hints such as low resolution, greyscale, aspect cropped, smooth/blurry are printed by the
script — e.g. "real studio portrait with heavy retouching looks AI-smooth", "AI image with film
grain / JPEG artefacts", "painting in the real class").

### 2.7 Keras Tuner
Toy: random search, 10 trials, units ∈ [32, 512], lr ∈ [1e-4, 1e-2] (log). Table [ ].
VisionGuard: 10 trials over lr ∈ [1e-6, 1e-3], dropout ∈ {0.1…0.6}, unfrozen layers ∈
{0, 10, 20, 40, 60}; objective val AUC. Table [ ]. Most important: [usually the learning rate —
too high destroys pretrained features, too low barely moves]. **Why best-val can overfit the
validation set:** with many trials, some score high by chance on ~150 val images; picking the max
selects that noise, so the winning val score is optimistically biased. The untouched test set (or
cross-validation) gives the honest estimate.

**Bonus — Grad-CAM:** 📸 `figures/gradcam.png` + comment on where the model looks (faces/skin
texture, backgrounds, edges vs. watermark corners).

## Chapter 3 — Experiment tracking

### 3.1 MLflow locally — definitions
- **Experiment:** a named folder grouping runs that try to solve the same problem.
- **Run:** one execution of training code, with its own ID, start/end time and status.
- **Parameter:** an input setting fixed before the run (lr, batch size); logged once.
- **Metric:** a number measured by the run, optionally per step (val_loss per epoch).
- **Artifact:** any file produced and stored with the run (plots, model, CSVs).
- **Tag:** free-form key/value metadata for searching and context (git commit, author, seed).
📸 two runs + compare view.

### 3.2 DagsHub
Tracking URI, username and token are read from `.env` by `python-dotenv`; MLflow picks them up
from `MLFLOW_TRACKING_URI/USERNAME/PASSWORD`. Nothing secret is in code; `.env` is git-ignored
(📸 `git check-ignore -v .env`).

### 3.3 Manual vs autolog
Manual (`MlflowEpochLogger` callback + explicit `log_params/log_metrics`): exactly the params I
choose (augmentation, unfreeze settings, image size), metrics continue across both stages,
tags (git commit, DVC version, seed), test metrics, threshold. Autolog: zero code, captures
optimiser config, epochs, batch size, every Keras metric per epoch, model summary and the model
itself; misses domain params (augmentation, image size), data version, test metrics and custom
plots, and with two `fit()` calls restarts step numbering and tries to re-log params.
📸 both run pages.

### 3.4 Artifacts
📸 Artifacts tab: `reports/confusion_matrix.png`, `reports/roc_curve.png`, `splits/*.csv`,
`environment/requirements-freeze.txt`, `keras_model/*.keras`, `model/` (MLflow format).

### 3.5 Comparison
| run | change | val AUC | val F1 | val FPR | test AUC |
|---|---|---|---|---|---|
Chosen: [ ] — ranked by validation ROC-AUC (threshold-independent ranking quality), tie-break on
lower validation FPR (the error we care about). Test metrics were not used for selection.
📸 compare view.

### 3.6 Registry
Two versions of `23l0673_A2_Ops_VisionGuard`; version [ ] has alias `champion`. Threshold and image
size are version tags. `training/predict_by_alias.py` loads `models:/…@champion`. 📸 registry +
terminal. **Getting the model to the server:** `scripts/fetch_model.py` downloads the champion
once per release into `models/`; Compose mounts it, the cloud image bakes it in. The app never
calls DagsHub per request.

## Chapter 4 — Flask application

### 4.1 Inference module
`visionguard/inference.py` loads the model once at start-up and warms it up. Loading per request
would re-read the weights and rebuild the graph each time (seconds per request, memory spikes,
collapse under concurrency). Serving calls the exact same `preprocess_bytes` used in training;
any difference (e.g. squash vs crop, 0–1 vs 0–255, RGB vs BGR) silently shifts inputs away from
what the model learned and degrades accuracy without any error. 📸 console output for two images.

### 4.2 Routes and validation
`/` upload, `POST /predict` → result, `/result/<id>`, `/history`, `/health`, `/api/history`.
Extension allow-list (415), size limit via `MAX_CONTENT_LENGTH` + explicit check (413), real
decode with Pillow (422), `secure_filename`, uploads kept in RAM (custom request class; Werkzeug
otherwise spools > 500 KB to a temp file), preview sent as a data-URI, never written to disk.
📸 three error pages.

### 4.3 Frontend
📸 upload, result (verdict, confidence, probability bar with threshold marker, caveat), history.

### 4.4 Configuration
All settings from environment variables (`webapp/config.py`). Debug is off: the Werkzeug debugger
allows arbitrary code execution from the browser, the reloader would load TensorFlow twice, and
tracebacks would leak internals. Production containers use gunicorn. 📸 `/health`.

## Chapter 5 — Database trail
Table `predictions`: id, created_at (UTC), sanitised filename, SHA-256, size, width, height,
label, confidence, p_ai, threshold, model_name, model_version, inference_mode, latency_ms — it
answers *what* (label/p_ai), *when*, *which model* and *how confidently*. Only successful
predictions are logged; the image itself is never stored (privacy) but the hash lets repeated
uploads be recognised. SQLite locally, PostgreSQL in Compose, same SQLAlchemy model.
📸 history page + `psql` query.

## Chapter 6 — Containerisation

### 6.1 Definitions
- **Image:** a read-only template (filesystem + metadata) to start containers from.
- **Container:** a running, isolated process started from an image with a thin writable layer.
- **Layer:** one filesystem diff created by a Dockerfile instruction; cached and shared.
- **Volume:** storage managed outside the container's writable layer that survives removal.
- **Port mapping:** forwarding a host port to a container port (`-p 8080:80`).

### 6.2 Dockerfile
Order: slim base → env → non-root user → copy requirements → `pip install` → copy source → `USER
appuser` → HEALTHCHECK → gunicorn. Dependencies before source so a code change reuses the cached
dependency layer. `tensorflow-cpu` instead of `tensorflow`: no CUDA libraries (hundreds of MB),
and single-image inference does not need a GPU. `.dockerignore` keeps `.env`, data, mlruns and
models out of the image; secrets enter only at run time. 📸 `whoami` → appuser.

### 6.3 Compose
`db` (postgres:16-alpine, named volume `pgdata`, `pg_isready` health check, no `ports`) and `web`
(`depends_on: condition: service_healthy`, `DATABASE_URL` uses host `db`). Passwords from `.env`;
`.env.example` committed. Model **mounted** read-only from `./models`: swapping the champion needs
no rebuild and the image stays small and model-agnostic. 📸 `docker compose ps` (healthy), port
5432 closed from host but reachable from `web`.

### 6.4 Image size
Before (`Dockerfile.unoptimized`): [ ] GB. After: [ ] GB. Largest layer: the pip install
(TensorFlow). Conclusion: [ ].

### 6.5 Troubleshooting
Failure: [missing model / wrong password]. Log excerpt [ ]. Diagnosis [ ]. Fix [ ].

### 6.6 CI
Toy: matrix Python 3.10/3.12; red run, green run 📸. Project: lint + tests + Docker build 📸,
badge in README. Secrets live in GitHub *Settings → Secrets and variables → Actions*, encrypted
and masked in logs; writing them in the workflow would publish them to everyone who can read the
repo and its history.

## Chapter 7 — API deployment (Google Cloud Run)
Why Cloud Run: runs the same container, HTTPS URL out of the box, scales to zero (free tier),
secrets via Secret Manager. Mapping: Compose service → Cloud Run service; `.env` → Secret Manager
+ env vars; image → Artifact Registry; health check → Cloud Run startup probe; DB not needed for
the model API (history stays in the local Flask app's DB).
Step-by-step with 📸: project, APIs, Artifact Registry, docker build/push, secret, deploy, URL,
logs. **API URL:** [ ]. Postman: health, predict (200), no key (401), non-image (422) 📸 + Cloud
Run logs. Tested from mobile data 📸.

## Chapter 8 — Flask ↔ cloud API integration
`INFERENCE_MODE=remote`: `webapp/remote.py` posts the validated bytes with the `X-API-Key` header
to the Cloud Run URL and stores the answer in the local DB. 📸 several results, `/health` showing
remote mode, history page with `remote` rows, `send_samples.py` output.

## What went wrong / lessons
[ e.g. Python 3.14 had no TensorFlow wheels → used 3.11; DB container not ready → health-check
gating; Cloud Run out-of-memory at 512 MiB → raised to 2 GiB ].
