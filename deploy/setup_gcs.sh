#!/usr/bin/env bash
# Create GCS buckets for trading-agents persistence and upload local history.
#
# Usage (from trading-agents/):
#   gcloud auth login
#   gcloud auth application-default login
#   ./deploy/setup_gcs.sh
#
#   gcloud projects list --format='table(projectId,name)'

set -euo pipefail

if [[ -z "${GCP_PROJECT:-}" ]]; then
  echo "ERROR: Set GCP_PROJECT to your GCP project ID."
  echo "  gcloud projects list --format='table(projectId,name)'"
  exit 1
fi

case "${GCP_PROJECT}" in
  MKTCrunch-MVP|mktcrunch-mvp|MKCRUNCH-MVP)
    echo "ERROR: GCP_PROJECT=${GCP_PROJECT} is a console display name, not a project ID."
    exit 1
    ;;
esac

PROJECT="${GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
AUDIT_BUCKET="${GCS_AUDIT_BUCKET:-mktcrunch-trading-agents-audit}"
DATA_BUCKET="${GCS_DATA_BUCKET:-mktcrunch-trading-agents-data}"

echo "==> Project ID: ${PROJECT}  Region: ${REGION}"

if ! gcloud projects describe "${PROJECT}" --format='value(projectId)' &>/dev/null; then
  echo "ERROR: '${PROJECT}' is not a valid GCP project ID."
  echo "List projects:  gcloud projects list --format='table(projectId,name)'"
  echo "Then run:       GCP_PROJECT=<projectId> ./deploy/setup_gcs.sh"
  exit 1
fi

gcloud config set project "${PROJECT}" --quiet
gcloud auth application-default set-quota-project "${PROJECT}" 2>/dev/null || true

create_bucket() {
  local name="$1"
  if gcloud storage buckets describe "gs://${name}" --project="${PROJECT}" &>/dev/null; then
    echo "    exists: gs://${name}"
  else
    echo "    creating: gs://${name}"
    gcloud storage buckets create "gs://${name}" \
      --project="${PROJECT}" \
      --location="${REGION}" \
      --uniform-bucket-level-access
  fi
}

echo "==> Creating buckets"
create_bucket "${AUDIT_BUCKET}"
create_bucket "${DATA_BUCKET}"

echo "==> Uploading local history"
export GCS_AUDIT_BUCKET="${AUDIT_BUCKET}"
export GCS_DATA_BUCKET="${DATA_BUCKET}"
export GCS_RISK_STATE_BUCKET="${DATA_BUCKET}"
export LOG_TO_FILE=false
python3 scripts/sync_history_to_gcs.py

echo ""
echo "==> Add to .env (if not already):"
echo "GCP_PROJECT=${PROJECT}"
echo "GCS_AUDIT_BUCKET=${AUDIT_BUCKET}"
echo "GCS_DATA_BUCKET=${DATA_BUCKET}"
echo "GCS_RISK_STATE_BUCKET=${DATA_BUCKET}"
echo ""
echo "==> Done. Audit: gs://${AUDIT_BUCKET}/audit/audit_events.jsonl"
echo "    Data:  gs://${DATA_BUCKET}/data/ and risk_state/"
