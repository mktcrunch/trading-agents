#!/usr/bin/env bash
# Upload required Cloud Run secrets from .env to Secret Manager.
#
# Usage:
#   export GCP_PROJECT=your-gcp-project-id
#   ./deploy/sync_secrets.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

: "${GCP_PROJECT:?Set GCP_PROJECT to your GCP project ID}"
PROJECT="${GCP_PROJECT}"
ENV_FILE="${ENV_FILE:-.env}"
SERVICE="${SERVICE_NAME:-trading-agents}"
SA_EMAIL="${SERVICE_SA:-${SERVICE}@${PROJECT}.iam.gserviceaccount.com}"

REQUIRED=(
  MC_API_KEY_ID
  MC_API_SECRET_KEY
  ALPACA_API_KEY_BASELINE
  ALPACA_SECRET_KEY_BASELINE
  ALPACA_API_KEY_INTERNAL
  ALPACA_SECRET_KEY_INTERNAL
  DATABENTO_API_KEY
  SCHEDULER_SECRET
)

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found"
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

upsert_secret() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "ERROR: ${name} is empty in ${ENV_FILE}"
    exit 1
  fi
  if gcloud secrets describe "${name}" --project="${PROJECT}" &>/dev/null; then
    echo "==> Updating secret ${name}"
    printf '%s' "${value}" | gcloud secrets versions add "${name}" \
      --project="${PROJECT}" \
      --data-file=-
  else
    echo "==> Creating secret ${name}"
    printf '%s' "${value}" | gcloud secrets create "${name}" \
      --project="${PROJECT}" \
      --replication-policy=automatic \
      --data-file=-
  fi
  gcloud secrets add-iam-policy-binding "${name}" \
    --project="${PROJECT}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet &>/dev/null || true
}

if [[ -z "${SCHEDULER_SECRET:-}" ]]; then
  SCHEDULER_SECRET="$(openssl rand -hex 24)"
  echo "==> Generated SCHEDULER_SECRET (add to .env if you want a stable value)"
fi

for key in "${REQUIRED[@]}"; do
  upsert_secret "${key}" "${!key}"
done

echo "==> Done. Secrets ready for Cloud Run deploy."
