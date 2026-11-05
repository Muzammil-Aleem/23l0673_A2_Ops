# Deploy the model API on Render Free

Use this instead of the Google Cloud Run section of RUNBOOK.md. Training,
DagsHub, Docker Compose evidence, and the local Flask application are unchanged.
No Google Cloud account or CLI is needed.

## Before deployment

1. Finish training and model registration. Run `python scripts/fetch_model.py`.
2. Check that `models/champion.keras` and `models/champion.json` exist.
3. Push your source code to a public GitHub repository.
4. On GitHub, open that repository -> Releases -> Draft a new release.
   Use tag `model-v1`, attach those two champion files, and publish the release.
   These are release attachments, not files added to Git history.
5. Copy each attachment's download URL. For example:
   `https://github.com/USER/23l0673_A2_Ops/releases/download/model-v1/champion.keras`
   and the same URL ending in `champion.json`. They must be publicly downloadable.
   Keep credentials and .env out of the release.

## Render browser setup

1. Sign up at https://render.com and connect GitHub.
2. Select New -> Web Service, and select your repository and `main` branch.
3. Name: `23l0673-a2-ops` (or an available name containing your roll number).
4. Select **Python 3** as the language/runtime. A Dockerfile exists in this
   repository, so explicitly check this selection. Leave Root Directory empty.
5. Build command:

   ```sh
   pip install -r model_api/requirements.txt && python scripts/download_deploy_model.py
   ```

6. Start command:

   ```sh
   gunicorn model_api.wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 180 --access-logfile -
   ```

7. Select the **Free ($0)** instance type.
8. Add environment variables:

   | Name | Value |
   | --- | --- |
   | PYTHON_VERSION | 3.11.9 |
   | CHAMPION_MODEL_URL | Public champion.keras release download URL |
   | CHAMPION_META_URL | Public champion.json release download URL |
   | MODEL_PATH | models/champion.keras |
   | MODEL_META | models/champion.json |
   | MODEL_API_KEY | A long random secret; use the same value in local .env |
   | TF_NUM_INTRAOP_THREADS | 1 |
   | TF_NUM_INTEROP_THREADS | 1 |
   | OMP_NUM_THREADS | 1 |
   | TF_CPP_MIN_LOG_LEVEL | 2 |

   Generate the API secret locally with:
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
   Do not include secret values in report screenshots.
9. Set Health Check Path to `/health` and deploy. Read both build and runtime logs.
10. Wait for Live, then open `https://YOUR-SERVICE.onrender.com/health`.
    It must report `status: ok`, your model version, and `auth_required: true`.

## Test and connect local Flask

Import docs/VisionGuard_API.postman_collection.json into Postman. Set base_url
to the Render HTTPS origin, and api_key to your secret. Capture successful health
and prediction responses, missing-key rejection, invalid-image rejection, and
matching server logs. Test from a phone hotspot or mobile data as well.

Set these in your local .env and restart `python run.py`:

```dotenv
INFERENCE_MODE=remote
MODEL_API_URL=https://YOUR-SERVICE.onrender.com/predict
MODEL_API_KEY=YOUR_SAME_SECRET
```

Upload multiple images at http://127.0.0.1:5000 and capture results, /health,
and /history showing remote mode. Keep the history database on your computer;
this API deployment does not need a hosted database.

## Free-plan limits and report

Render Free has 512 MB RAM. TensorFlow and a trained model may exceed that;
this configuration is a candidate, not a verified deployment. If logs show an
out-of-memory kill, the service has not passed deployment. Save the logs and
investigate memory use before choosing another host or a lighter inference runtime.
Do not select a paid upgrade to resolve it if your budget is zero.

Free services sleep after 15 minutes without incoming traffic. Before a demo,
open /health and wait for the service to wake, then submit images. Cold starts
can exceed the local client's 30-second timeout. Keep the service available for
grading and document these limitations. No keep-alive automation is needed.

In report chapters 7 and 8, replace the Cloud Run draft with your actual Render
steps, screenshots, URL, logs, and integration results. Explain that Compose's
database runs locally and only the stateless model API is hosted.

Official references:
- https://render.com/docs/free
- https://render.com/pricing
- https://render.com/docs/deploy-flask
- https://render.com/docs/python-version
