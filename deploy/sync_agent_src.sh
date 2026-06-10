#!/usr/bin/env bash
# Bundle src/ + deps into agent folders for Vertex Agent Engine deploy.
# Agent Engine only packages agents/<name>/ — src must live inside that folder.
#
# Usage: ./deploy/sync_agent_src.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

AGENTS=(twin_ledger_baseline twin_ledger_internal)

echo "==> Syncing src/ into agent folders for Agent Engine"
for name in "${AGENTS[@]}"; do
  dest="agents/${name}/src"
  rm -rf "${dest}"
  rsync -a --exclude '__pycache__' --exclude '*.pyc' src/ "${dest}/"
  echo "    ${dest}/"
done

echo "==> Writing agent requirements.txt"
for name in "${AGENTS[@]}"; do
  cp deploy/agent_engine_requirements.txt "agents/${name}/requirements.txt"
done

if [[ -f .env ]]; then
  echo "==> Writing Agent Engine .env (Vertex-safe: no API keys in remote env)"
  for name in "${AGENTS[@]}"; do
    # Drop API keys when using Vertex ADC; avoids Gemini routing conflicts on Agent Engine.
    grep -v -E '^(GEMINI_API_KEY|GOOGLE_API_KEY)=' .env > "agents/${name}/.env"
  done
else
  echo "⚠️  No .env found — create one before Agent Engine deploy"
fi

echo "==> Done. Agent folders ready for: adk deploy agent_engine agents/<name>"
