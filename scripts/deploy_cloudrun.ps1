# Task 7 (pathway C: Google Cloud Run) — deploy the model API container.
# Prerequisites: gcloud CLI installed, `gcloud auth login`, a GCP project with billing
# enabled (Cloud Run free tier: 2M requests, 360k GiB-seconds/month — check the current terms),
# and models/champion.keras + champion.json present (python scripts/fetch_model.py).
#
#   $env:GCP_PROJECT = "your-project-id"
#   $env:MODEL_API_KEY = (python -c "import secrets; print(secrets.token_urlsafe(32))")

$ErrorActionPreference = "Stop"
$REGION  = "us-central1"
$SERVICE = "vg-23l0673-a2-ops-api"      # Cloud Run names: lowercase, digits, dashes, must START with a letter
$REPO    = "vg-23l0673-a2-ops"
$IMAGE   = "$REGION-docker.pkg.dev/$env:GCP_PROJECT/$REPO/visionguard-api:v1"

gcloud config set project $env:GCP_PROJECT
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com

# 1. registry for the image
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION 2>$null
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

# 2. build for linux/amd64 (Cloud Run runs x86) and push
docker build --platform linux/amd64 -f model_api/Dockerfile -t $IMAGE .
docker push $IMAGE

# 3. API key as a Secret Manager secret (never as a plain env var in the command history)
$env:MODEL_API_KEY | gcloud secrets create visionguard-api-key --data-file=- 2>$null
if (-not $?) { $env:MODEL_API_KEY | gcloud secrets versions add visionguard-api-key --data-file=- }
$PN = gcloud projects describe $env:GCP_PROJECT --format="value(projectNumber)"
gcloud secrets add-iam-policy-binding visionguard-api-key `
  --member="serviceAccount:$PN-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"

# 4. deploy: 2 GiB RAM for TensorFlow, public HTTPS URL (the API key protects /predict)
gcloud run deploy $SERVICE --image $IMAGE --region $REGION --platform managed `
  --allow-unauthenticated --memory 2Gi --cpu 1 --min-instances 0 --max-instances 2 `
  --timeout 60 --set-secrets "MODEL_API_KEY=visionguard-api-key:latest"

# 5. URL + smoke test
$URL = gcloud run services describe $SERVICE --region $REGION --format "value(status.url)"
Write-Host "API URL: $URL"
curl.exe -s "$URL/health"
gcloud run services logs read $SERVICE --region $REGION --limit 20
