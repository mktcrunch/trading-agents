#!/usr/bin/env bash
# Deploy trading-agents to Cloud Run + Cloud Scheduler.
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT
#   Secrets in Secret Manager (or pass via --set-env-vars)
#
# Usage:
#   export GCP_PROJECT=your-project
#   export GCP_REGION=us-central1
#   export SERVICE_NAME=trading-agents
#   ./deploy/setup_cloud_run.sh

set -euo pipefail

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE_NAME:-trading-agents}"
IMAGE="gcr.io/${PROJECT}/${SERVICE}:latest"
SA_EMAIL="${SERVICE_SA:-${SERVICE}@${PROJECT}.iam.gserviceaccount.com}"

echo "==> Building image ${IMAGE}"
gcloud builds submit --tag "${IMAGE}" .

VERTEX_MODE="${VERTEX_MODE:-false}"
VERTEX_ENV="GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=${PROJECT},GCP_REGION=${REGION},GOOGLE_CLOUD_LOCATION=global,GEMINI_VERTEX_LOCATION=global,GEMINI_FLASH_MODEL=gemini-3.5-flash,USE_ADK=true,USE_ADK_WORKFLOW=true"
IDS_FILE="deploy/agent_engine_ids.env"
if [[ -f "${IDS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${IDS_FILE}"
fi
CHAT_ENV="DASHBOARD_CHAT_ENABLED=true,DASHBOARD_CHAT_READ_ONLY=true"
if [[ -n "${AGENT_ENGINE_BASELINE_ID:-}" && -n "${AGENT_ENGINE_INTERNAL_ID:-}" ]]; then
  CHAT_ENV="${CHAT_ENV},AGENT_ENGINE_BASELINE_ID=${AGENT_ENGINE_BASELINE_ID},AGENT_ENGINE_INTERNAL_ID=${AGENT_ENGINE_INTERNAL_ID},CHAT_BACKEND=vertex"
fi

BASE_ENV="LOG_TO_FILE=false,TRADING_ENABLED=true,DRY_RUN=false,\
GCS_AUDIT_BUCKET=${GCS_AUDIT_BUCKET:-mktcrunch-trading-agents-audit},\
GCS_DATA_BUCKET=${GCS_DATA_BUCKET:-mktcrunch-trading-agents-data},\
GCS_RISK_STATE_BUCKET=${GCS_RISK_STATE_BUCKET:-mktcrunch-trading-agents-data},\
${CHAT_ENV}"

if [[ "${VERTEX_MODE}" == "true" ]]; then
  echo "==> Deploying Cloud Run with Vertex AI Gemini (no GEMINI_API_KEY secret)"
  ENV_VARS="${BASE_ENV},${VERTEX_ENV}"
  SECRETS="\
MC_API_KEY_ID=MC_API_KEY_ID:latest,\
MC_API_SECRET_KEY=MC_API_SECRET_KEY:latest,\
ALPACA_API_KEY_BASELINE=ALPACA_API_KEY_BASELINE:latest,\
ALPACA_SECRET_KEY_BASELINE=ALPACA_SECRET_KEY_BASELINE:latest,\
ALPACA_API_KEY_INTERNAL=ALPACA_API_KEY_INTERNAL:latest,\
ALPACA_SECRET_KEY_INTERNAL=ALPACA_SECRET_KEY_INTERNAL:latest,\
DATABENTO_API_KEY=DATABENTO_API_KEY:latest,\
SCHEDULER_SECRET=SCHEDULER_SECRET:latest"
else
  echo "==> Deploying Cloud Run with GEMINI_API_KEY (Google AI Studio)"
  ENV_VARS="${BASE_ENV}"
  SECRETS="\
MC_API_KEY_ID=MC_API_KEY_ID:latest,\
MC_API_SECRET_KEY=MC_API_SECRET_KEY:latest,\
ALPACA_API_KEY_BASELINE=ALPACA_API_KEY_BASELINE:latest,\
ALPACA_SECRET_KEY_BASELINE=ALPACA_SECRET_KEY_BASELINE:latest,\
ALPACA_API_KEY_INTERNAL=ALPACA_API_KEY_INTERNAL:latest,\
ALPACA_SECRET_KEY_INTERNAL=ALPACA_SECRET_KEY_INTERNAL:latest,\
GEMINI_API_KEY=GEMINI_API_KEY:latest,\
DATABENTO_API_KEY=DATABENTO_API_KEY:latest,\
SCHEDULER_SECRET=SCHEDULER_SECRET:latest"
fi

echo "==> Deploying Cloud Run service ${SERVICE}"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --no-allow-unauthenticated \
  --service-account "${SA_EMAIL}" \
  --memory 1Gi \
  --cpu 1 \
  --timeout 900 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "${ENV_VARS}" \
  --set-secrets "${SECRETS}"

SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" --format='value(status.url)')

echo "==> Service URL: ${SERVICE_URL}"

echo "==> Creating scheduler jobs (America/New_York)"

# Overnight orders — 4:10 PM ET weekdays
gcloud scheduler jobs create http "${SERVICE}-overnight" \
  --location "${REGION}" \
  --schedule "10 16 * * 1-5" \
  --time-zone "America/New_York" \
  --uri "${SERVICE_URL}/jobs/overnight" \
  --http-method POST \
  --oidc-service-account-email "${SA_EMAIL}" \
  2>/dev/null || gcloud scheduler jobs update http "${SERVICE}-overnight" \
  --location "${REGION}" \
  --schedule "10 16 * * 1-5" \
  --time-zone "America/New_York" \
  --uri "${SERVICE_URL}/jobs/overnight" \
  --http-method POST \
  --oidc-service-account-email "${SA_EMAIL}"

# Intraday risk — every 5 min, 9 AM–3:55 PM ET weekdays (market-hours gate inside app)
gcloud scheduler jobs create http "${SERVICE}-risk" \
  --location "${REGION}" \
  --schedule "*/5 9-15 * * 1-5" \
  --time-zone "America/New_York" \
  --uri "${SERVICE_URL}/jobs/risk" \
  --http-method POST \
  --oidc-service-account-email "${SA_EMAIL}" \
  2>/dev/null || gcloud scheduler jobs update http "${SERVICE}-risk" \
  --location "${REGION}" \
  --schedule "*/5 9-15 * * 1-5" \
  --time-zone "America/New_York" \
  --uri "${SERVICE_URL}/jobs/risk" \
  --http-method POST \
  --oidc-service-account-email "${SA_EMAIL}"

echo "==> Done."
echo "Overnight: POST ${SERVICE_URL}/jobs/overnight  (4:10 PM ET Mon-Fri)"
echo "Risk:      POST ${SERVICE_URL}/jobs/risk         (every 5m 9-15 ET Mon-Fri)"
