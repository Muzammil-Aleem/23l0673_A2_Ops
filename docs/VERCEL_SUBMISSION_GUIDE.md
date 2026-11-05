# VisionGuard: Vercel deployment and submission

Use this deployment route instead of Cloud Run or Render. Complete tasks 1-6
using RUNBOOK.md first. Keep training and the Flask website/history database on
your computer; host only model_api on Vercel. Use your actual roll number everywhere.

## 1. Local preparation

From the project root, activate .venv. Finish training, DagsHub experiment logging,
model comparison, two registry versions, and the champion alias. Run:

```powershell
python scripts/fetch_model.py
python scripts/prepare_vercel.py
```

The second command prepares deploy/vercel with Python 3.12, API-only dependencies,
an app entrypoint, a model download build command, and function settings. It does
not copy .env, training data, training packages, or model weights. Rerun it whenever
you change visionguard/, model_api/, or scripts/download_deploy_model.py, and commit
the refreshed files. Models are downloaded at build time, never per prediction.

## 2. Publish model attachments

On your public GitHub repository, select Releases -> Draft a new release.
Create tag model-v1. Attach models/champion.keras and models/champion.json, then
publish. Copy each public attachment download URL. They normally look like:

```text
https://github.com/USER/23l0673_A2_Ops/releases/download/model-v1/champion.keras
https://github.com/USER/23l0673_A2_Ops/releases/download/model-v1/champion.json
```

Open the links signed out to check they download files. Do not attach .env or
credentials. Keep the release unchanged so this deployment has fixed model inputs.

```powershell
git add scripts/prepare_vercel.py scripts/download_deploy_model.py deploy/vercel docs/VERCEL_SUBMISSION_GUIDE.md
git commit -m "Prepare model API for Vercel"
git push
```

## 3. Create the Vercel project in your browser

1. Sign up at https://vercel.com with GitHub. Use your personal Hobby account.
2. Add New -> Project -> import your VisionGuard GitHub repository.
3. Project name: 23l0673-a2-ops, or an available variant containing your roll number.
4. Root Directory: **deploy/vercel**. This is important: the main project has a
   different Flask entrypoint and a much larger training requirements file.
5. Framework Preset: **Flask**. Keep automatic dependency installation.
   The committed vercel.json sets the build command. No Gunicorn start command
   and no Vercel CLI are needed for this browser-based route.
6. Add the environment variables below, at least for Production. Add them to
   Preview too if you want preview builds to work. Mark the API key sensitive.

| Variable | Value |
| --- | --- |
| CHAMPION_MODEL_URL | Public champion.keras attachment URL |
| CHAMPION_META_URL | Public champion.json attachment URL |
| MODEL_API_KEY | Your long random secret |
| VERCEL_SUPPORT_LARGE_FUNCTIONS | 1 |
| MAX_UPLOAD_MB | 4 |
| TF_NUM_INTRAOP_THREADS | 1 |
| TF_NUM_INTEROP_THREADS | 1 |
| OMP_NUM_THREADS | 1 |
| TF_CPP_MIN_LOG_LEVEL | 2 |

Generate your API key locally with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

7. Deploy. Check build logs for Python 3.12, serving dependencies, and successful
   download of both model files. Check Functions settings for Fluid Compute and
   Active CPU if the large-function beta reports an eligibility problem.
8. Open the production URL /health. It must return JSON with status ok, your
   model version, and auth_required true. A Vercel sign-in page is not a passing
   health check. Check Deployment Protection settings so the production API is
   public; prediction requests remain protected by your application's API key.
9. If you change variables after deployment, redeploy for them to take effect.

## 4. Verify the public API

Import docs/VisionGuard_API.postman_collection.json into Postman.
Set base_url to https://YOUR-PROJECT.vercel.app (without /predict), and api_key
to the secret. Test Health, Predict with a genuine image, Predict without key
(401), and Predict non-image with the key (422).

Use image files smaller than 4 MB. Vercel limits the entire request to 4.5 MB,
including multipart overhead. Local Task 4 size validation can still be
demonstrated separately. Hosting may reject large requests before Flask handles them.

Save screenshots of deployment settings (secret values hidden), successful build,
public /health, Postman responses, and Runtime Logs showing prediction entries.
Test a prediction while using a different network, such as a phone hotspot.

## 5. Connect local Flask

Edit your local .env:

```dotenv
INFERENCE_MODE=remote
MODEL_API_URL=https://YOUR-PROJECT.vercel.app/predict
MODEL_API_KEY=YOUR_SAME_SECRET
MAX_UPLOAD_MB=4
```

Start `python run.py`, then open http://127.0.0.1:5000. First visit the deployed
/health and wait for model initialization. Local remote requests have a 30-second
timeout, so an initial cold start can fail; wait for health to succeed and retry.
Upload multiple images. Screenshot results, local /health showing remote mode,
and /history with remote entries. History stays in your local database.

The model is initialized once per Vercel worker instance. New instances have cold
starts; it is not one globally permanent process. Explain this in your report.

## 6. Failure checks

- Missing model: check attachment URLs and successful build downloads.
- Unsupported Python: ensure Root Directory deploy/vercel and Python 3.12.
- Bundle too large: check large functions enabled; do not use root training requirements.
- 401 JSON from API: API key mismatch. Login HTML: Vercel Deployment Protection.
- 413: use a smaller image. The platform limit is separate from Flask validation.
- Initialization, memory, or timeout failure: inspect Runtime Logs. A successful
  build alone does not prove the model runs. Do not submit until real predictions pass.

This setup has been prepared and checked locally for syntax and packaging, but
has not been deployed: your trained champion and accounts are still needed.
Remain on Hobby and within its quotas; a paid plan is not part of this guide.

## 7. Finish submission

Complete every required task, including the DVC toy, tuner toy and at least 8
applied trials, local MLflow toy, manual/autolog runs, at least 6 meaningful
experiment runs, Docker warm-up, Compose evidence, and red/green toy CI.

Replace Cloud Run content in docs/REPORT.md with the actual Vercel process.
Include all task evidence, actual results, captions, explanations, problems,
AI-assistant disclosure, GitHub/DagsHub/live API links, and graded commit hash.
Commit and push final source/results, confirm green GitHub CI, run
`git rev-parse HEAD`, and record that hash in the report. Export the PDF as
23l0673_VisionGuard.pdf (substitute your roll number). Follow your instructor's
page limit and submission portal instructions. Keep the production API accessible
and test it from another network before submitting.

Official references:
- https://vercel.com/docs/frameworks/backend/flask
- https://vercel.com/docs/functions/runtimes/python/python-version
- https://vercel.com/docs/functions/limitations
- https://vercel.com/docs/plans/hobby
