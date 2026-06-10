#!/usr/bin/env bash
# Set up Cloud Scheduler jobs to trigger Vertex AI Reasoning Engines directly.
#
# Usage:
#   export GCP_PROJECT=your-gcp-project-id
#   export GCP_REGION=us-central1
#   ./deploy/setup_scheduler.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

: "${GCP_PROJECT:?Set GCP_PROJECT to your GCP project ID}"
PROJECT="${GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE_NAME:-trading-agents}"
SA_EMAIL="${SERVICE_SA:-${SERVICE}@${PROJECT}.iam.gserviceaccount.com}"

IDS_FILE="deploy/agent_engine_ids.env"
if [[ -f "${IDS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${IDS_FILE}"
else
  echo "ERROR: ${IDS_FILE} not found. Deploy agents first using ./deploy/setup_vertex.sh"
  exit 1
fi

BASELINE_ID="${AGENT_ENGINE_BASELINE_ID:?Missing AGENT_ENGINE_BASELINE_ID in deploy/agent_engine_ids.env}"
INTERNAL_ID="${AGENT_ENGINE_INTERNAL_ID:?Missing AGENT_ENGINE_INTERNAL_ID in deploy/agent_engine_ids.env}"

echo "╔══════════════════════════════════════════════════════════════╗"
# Only use emojis if explicitly asked; we avoid emojis in script output as well.
echo "║  Twin Ledger — Direct Cloud Scheduler Setup                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Project:     ${PROJECT}"
echo "Region:      ${REGION}"
echo "Service SA:  ${SA_EMAIL}"
echo "Baseline ID: ${BASELINE_ID}"
echo "Internal ID: ${INTERNAL_ID}"
echo ""

create_or_update_job() {
  local job_name="$1"
  local schedule="$2"
  local engine_id="$3"
  local query="$4"

  local uri="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/reasoningEngines/${engine_id}:streamQuery"
  local body="{\"classMethod\": \"stream_query\", \"input\": {\"message\": \"${query}\", \"user_id\": \"scheduler\"}}"

  echo "==> Configuring job: ${job_name}"
  echo "    Schedule: ${schedule}"
  echo "    Query:    ${query}"

  set +e
  gcloud scheduler jobs describe "${job_name}" --location="${REGION}" &>/dev/null
  local exists=$?
  set -e

  if [[ ${exists} -eq 0 ]]; then
    echo "    Deleting existing job..."
    gcloud scheduler jobs delete "${job_name}" --location="${REGION}" --quiet
  fi

  echo "    Creating new job..."
  gcloud scheduler jobs create http "${job_name}" \
    --location="${REGION}" \
    --schedule="${schedule}" \
    --time-zone="America/New_York" \
    --uri="${uri}" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body="${body}" \
    --oauth-service-account-email="${SA_EMAIL}" \
    --quiet
  echo "    ✓ Success"
  echo ""
}

# 1. Overnight Trading Workflow (4:10 PM ET Mon-Fri)
create_or_update_job "baseline-overnight-direct" "10 16 * * 1-5" "${BASELINE_ID}" "Run daily trading workflow."
create_or_update_job "internal-overnight-direct" "10 16 * * 1-5" "${INTERNAL_ID}" "Run daily trading workflow."

# 2. Intraday Risk Check (Every 15 Minutes, 9:30 AM – 4:00 PM ET Mon-Fri)
create_or_update_job "baseline-risk-direct" "*/15 9-15 * * 1-5" "${BASELINE_ID}" "Run intraday risk check."
create_or_update_job "internal-risk-direct" "*/15 9-15 * * 1-5" "${INTERNAL_ID}" "Run intraday risk check."

# 3. Post-Market-Open Chase (9:45 AM, 12:45 PM, and 2:45 PM ET Mon-Fri)
create_or_update_job "baseline-chase-direct" "45 9,12,14 * * 1-5" "${BASELINE_ID}" "Run post-open chase."
create_or_update_job "internal-chase-direct" "45 9,12,14 * * 1-5" "${INTERNAL_ID}" "Run post-open chase."

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Scheduler setup complete                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
