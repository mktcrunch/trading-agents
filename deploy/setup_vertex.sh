#!/usr/bin/env bash
# Vertex AI setup: enable APIs, IAM, deploy ADK agents to Agent Engine,
# optionally redeploy Cloud Run with Vertex Gemini (no API key).
#
# Prerequisites:
#   gcloud auth login
#   gcloud auth application-default login
#   pip install google-cloud-aiplatform[agent_engines]  (in venv)
#
# Usage:
#   export GCP_PROJECT=turing-course-437219-c0
#   export GCP_REGION=us-central1
#   ./deploy/setup_vertex.sh              # APIs + IAM + Agent Engine only
#   ./deploy/setup_vertex.sh --cloud-run  # also redeploy Cloud Run with Vertex

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

DEFAULT_PROJECT_ID="turing-course-437219-c0"
REQUESTED="${GCP_PROJECT:-$DEFAULT_PROJECT_ID}"

resolve_project_id() {
  case "${1}" in
    MKTCrunch-MVP|mktcrunch-mvp|MKCRUNCH-MVP) echo "turing-course-437219-c0" ;;
    *) echo "${1}" ;;
  esac
}

PROJECT="$(resolve_project_id "${REQUESTED}")"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE_NAME:-trading-agents}"
SA_EMAIL="${SERVICE_SA:-${SERVICE}@${PROJECT}.iam.gserviceaccount.com}"
DEPLOY_CLOUD_RUN=false

for arg in "$@"; do
  case "${arg}" in
    --cloud-run) DEPLOY_CLOUD_RUN=true ;;
  esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Twin Ledger — Vertex AI + Agent Engine Setup                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Project: ${PROJECT}"
echo "Region:  ${REGION}"
echo ""

echo "==> Setting gcloud project"
gcloud config set project "${PROJECT}"

echo "==> Enabling APIs"
gcloud services enable \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT}"

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
RE_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"

echo "==> IAM for Cloud Run service account (${SA_EMAIL})"
for role in \
  roles/aiplatform.user \
  roles/storage.objectAdmin \
  roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${role}" \
    --condition=None \
    --quiet 2>/dev/null || true
done

echo "==> IAM for Agent Engine service account (${RE_SA})"
for role in \
  roles/aiplatform.user \
  roles/storage.objectAdmin \
  roles/logging.logWriter \
  roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${RE_SA}" \
    --role="${role}" \
    --condition=None \
    --quiet 2>/dev/null || true
done

echo "==> Bundling src/ for Agent Engine"
chmod +x deploy/sync_agent_src.sh
./deploy/sync_agent_src.sh

if ! command -v adk &>/dev/null; then
  echo "ERROR: adk CLI not found. Activate venv: source venv/bin/activate"
  exit 1
fi

echo "==> Checking vertexai SDK (required for Agent Engine deploy)"
if ! python -c "import vertexai" 2>/dev/null; then
  echo "Installing google-cloud-aiplatform[agent_engines]..."
  pip install "google-cloud-aiplatform[agent_engines]>=1.88.0"
fi
python -c "import vertexai; print('    vertexai OK')"

IDS_FILE="deploy/agent_engine_ids.env"
if [[ -f "${IDS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${IDS_FILE}"
  echo "==> Loaded existing engine IDs from ${IDS_FILE} (will update in place if set)"
fi
: > "${IDS_FILE}"

deploy_agent() {
  local agent_path="$1"
  local display_name="$2"
  local var_name="$3"
  local engine_id="${!var_name:-}"
  local log_file="deploy/logs/${var_name}_deploy.log"

  mkdir -p deploy/logs

  echo ""
  echo "==> Deploying Agent Engine: ${display_name}"
  echo "    Path: ${agent_path}"
  echo "    ⏳ First deploy often takes 10–20 min (Cloud Build + container push). Output streams below."
  if [[ -n "${engine_id}" ]]; then
    echo "    Reusing existing engine ID: ${engine_id}"
  else
    echo "    Tip: if console shows UNKNOWN shells, delete them and re-run, or set ${var_name}=<id> to update in place."
  fi

  local -a adk_args=(
    deploy agent_engine
    --project="${PROJECT}"
    --region="${REGION}"
    --display_name="${display_name}"
  )
  # OTEL can trigger monitoring permission noise; enable once deploy is stable.
  if [[ "${DEPLOY_OTEL_TO_CLOUD:-false}" == "true" ]]; then
    adk_args+=(--otel_to_cloud)
  fi
  if [[ -n "${engine_id}" ]]; then
    adk_args+=(--agent_engine_id="${engine_id}")
  fi
  adk_args+=("${agent_path}")

  set +e
  adk "${adk_args[@]}" 2>&1 | tee "${log_file}"
  DEPLOY_STATUS=${PIPESTATUS[0]}
  set -e

  if [[ ${DEPLOY_STATUS} -ne 0 ]] || grep -qiE 'Deploy failed|Failed to deploy to Agent Platform|Traceback' "${log_file}"; then
    echo "ERROR: Agent Engine deploy failed for ${display_name} (see ${log_file})"
    if grep -qE 'failed to start and cannot serve traffic|INTERNAL' "${log_file}"; then
      echo "    Check Cloud Logging for container errors:"
      echo "    https://console.cloud.google.com/logs/query?project=${PROJECT}"
      echo "    Query: resource.type=\"aiplatform.googleapis.com/ReasoningEngine\" severity>=ERROR"
      echo "    Common fixes: missing a2a-sdk (google-adk[a2a]), retry on transient INTERNAL"
    fi
    exit 1
  fi

  RESOURCE=$(grep -oE 'projects/[^ ]+/locations/[^ ]+/reasoningEngines/[0-9]+' "${log_file}" | tail -1 || true)
  if [[ -n "${RESOURCE}" ]]; then
    ENGINE_ID="${RESOURCE##*/}"
    echo "${var_name}=${ENGINE_ID}" >> "${IDS_FILE}"
    echo "${var_name}_RESOURCE=${RESOURCE}" >> "${IDS_FILE}"
    echo "    ✓ Saved ${var_name}=${ENGINE_ID}"
  else
    echo "WARNING: Could not parse engine ID from log — check console manually"
  fi
}

deploy_agent "agents/twin_ledger_baseline" "Twin Ledger Baseline" "AGENT_ENGINE_BASELINE_ID"
deploy_agent "agents/twin_ledger_internal" "Twin Ledger Internal" "AGENT_ENGINE_INTERNAL_ID"

echo ""
echo "==> Agent Engine IDs written to ${IDS_FILE}"
cat "${IDS_FILE}"

if [[ "${DEPLOY_CLOUD_RUN}" == "true" ]]; then
  echo ""
  echo "==> Redeploying Cloud Run with Vertex AI (no GEMINI_API_KEY)"
  export GCP_PROJECT="${PROJECT}"
  export GCP_REGION="${REGION}"
  export VERTEX_MODE=true
  chmod +x deploy/setup_cloud_run.sh
  ./deploy/setup_cloud_run.sh
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Vertex setup complete                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Local Vertex (ADC):"
echo "  export GOOGLE_GENAI_USE_VERTEXAI=true"
echo "  export GCP_PROJECT=${PROJECT}"
echo "  export GCP_REGION=${REGION}"
echo "  gcloud auth application-default login"
echo "  python main.py --baseline"
echo ""
echo "ADK web with Vertex:"
echo "  GOOGLE_GENAI_USE_VERTEXAI=true adk web agents"
echo ""
echo "Agent Engine console:"
echo "  https://console.cloud.google.com/vertex-ai/agents/agent-engines?project=${PROJECT}"
echo ""
if [[ "${DEPLOY_CLOUD_RUN}" != "true" ]]; then
  echo "Cloud Run + Vertex:"
  echo "  ./deploy/setup_vertex.sh --cloud-run"
fi
