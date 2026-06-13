#!/usr/bin/env bash
# Load deployment settings from project .env (single source of truth).
#
# Required in .env:
#   GCP_PROJECT
#   AGENT_ENGINE_BASELINE_ID
#   AGENT_ENGINE_INTERNAL_ID
#
# Usage (from any deploy/*.sh):
#   source "$(dirname "$0")/load_deploy_env.sh"
#   load_deploy_env

load_deploy_env() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi

  local env_file="${root}/.env"
  if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: Missing ${env_file}. Copy .env.template and set engine IDs."
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a

  if [[ -z "${GCP_PROJECT:-}" ]]; then
    echo "ERROR: Set GCP_PROJECT in .env"
    exit 1
  fi
  if [[ -z "${AGENT_ENGINE_BASELINE_ID:-}" || -z "${AGENT_ENGINE_INTERNAL_ID:-}" ]]; then
    echo "ERROR: Set AGENT_ENGINE_BASELINE_ID and AGENT_ENGINE_INTERNAL_ID in .env"
    exit 1
  fi

  export GCP_PROJECT
  export AGENT_ENGINE_BASELINE_ID
  export AGENT_ENGINE_INTERNAL_ID

  echo "==> Engine IDs from .env (deploy in place)"
  echo "    Baseline: ${AGENT_ENGINE_BASELINE_ID}"
  echo "    Internal: ${AGENT_ENGINE_INTERNAL_ID}"
}
