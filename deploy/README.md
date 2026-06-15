# Cloud Run Deployment

## ADK (Agent Development Kit)

Twin Ledger runs on **Google ADK** for competition compliance:

| Layer | Implementation |
|-------|----------------|
| Orchestration | ADK `Workflow` + `LlmAgent` multi-agent coordinators |
| Tools | ADK `FunctionTool` wrappers + optional MCP stdio server |
| LLM | Gemini 3.5 Flash (`gemini-3.5-flash`, Vertex `global` endpoint) |
| Local dev | `adk web agents` → `twin_ledger_baseline` / `twin_ledger_internal` |

```bash
# MCP tool server (stdio)
python -m src.adk.mcp.server

# ADK web UI
adk web agents
```

Env: `USE_ADK`, `USE_ADK_WORKFLOW`, `USE_ADK_MCP`, `GOOGLE_GENAI_USE_VERTEXAI`

## Vertex AI + Agent Engine

Competition stack: **Gemini on Vertex** + **ADK on Agent Engine** + **Cloud Run** for scheduler/dashboard.

This deploys:
- `twin_ledger_baseline` → Vertex AI Agent Engine
- `twin_ledger_internal` → Vertex AI Agent Engine

**Engine IDs:** set `AGENT_ENGINE_BASELINE_ID` and `AGENT_ENGINE_INTERNAL_ID` in `.env` before deploy. All deploy scripts read `.env` only and pass `--agent_engine_id` so updates happen in place (no new shells). After deploy, run `./deploy/setup_scheduler.sh` if IDs changed.

### One-time setup

```bash
gcloud auth login
gcloud auth application-default login

# In .env: GCP_PROJECT, AGENT_ENGINE_BASELINE_ID, AGENT_ENGINE_INTERNAL_ID
chmod +x deploy/setup_vertex.sh deploy/sync_agent_src.sh deploy/setup_scheduler.sh
./deploy/setup_vertex.sh              # APIs + IAM + deploy both agents to Agent Engine
./deploy/setup_scheduler.sh           # point Cloud Scheduler at .env engine IDs
./deploy/setup_vertex.sh --cloud-run  # also redeploy Cloud Run with Vertex (no API key)
```

### Local Vertex (ADC)

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GCP_PROJECT=your-gcp-project-id
export GCP_REGION=us-central1
gcloud auth application-default login

python main.py --baseline
GOOGLE_GENAI_USE_VERTEXAI=true adk web agents
```

### Cloud Run + Vertex

`VERTEX_MODE=true ./deploy/setup_cloud_run.sh` sets `GOOGLE_GENAI_USE_VERTEXAI=true` and uses the service account for Gemini (no `GEMINI_API_KEY` secret). The service account needs `roles/aiplatform.user`.

**Gemini 3.5 on Vertex:** model calls use `GEMINI_VERTEX_LOCATION=global` (required for `gemini-3.5-flash`). Agent Engine, Cloud Run, and GCS stay in `GCP_REGION=us-central1`.

### A2A verification

Agent Engine deploys include `google-adk[a2a]` (see `deploy/agent_engine_requirements.txt`). After deploy:

```bash
python scripts/verify_a2a.py
```

Checks requirements, local ADK `sub_agents` tree, and live Reasoning Engine REST resources (`stream_query` exposed).

### Grounding

- **Both signal agents:** Grounding with Google Search (`SIGNAL_GOOGLE_SEARCH_GROUNDING=true`, default)
- **Internal Signal (incremental):** MC predictions + gate-approved DataBento features + learning memory

## Architecture

```
Cloud Scheduler (4:10 PM ET)  ──POST──►  /jobs/overnight
                                              │
                                              ├─ discovery (if stale; cache fallback on failure)
                                              ├─ baseline → OPG limit orders
                                              └─ internal → OPG limit orders + Kelly

Cloud Scheduler (*/5 9-15 ET) ──POST──►  /jobs/risk
                                              │
                                              ├─ baseline risk (fixed stops, trailing)
                                              └─ internal risk (ATR, trailing, 15m pred, EOD)
```

## Risk agent differences

| Feature | Baseline | Internal |
|---------|----------|----------|
| Base stop | Fixed -1% | ATR × 1.5 (fallback -1%) |
| Trailing | **Pure LLM** activation + profit lock | **Scripted 1%/70% + LLM** (hybrid — never looser than scripted) |
| 15-min prediction gate | No | Yes — defers exit if model disagrees |
| EOD exit (~3:54 PM) | -0.95% losers | ATR-implied or -0.95% |

Trailing stop state persists in `data/risk_state_{system}.json` locally, or in GCS when `GCS_RISK_STATE_BUCKET` is set (required for Cloud Run).

## GCS persistence (do this first)

Create buckets in your GCP project and upload local history:

```bash
gcloud auth login
gcloud auth application-default login
./deploy/setup_gcs.sh
```

**Note:** Use the gcloud *project ID*, not the console display name. List IDs with `gcloud projects list --format='table(projectId,name)'`.

Buckets (default names — override with `GCS_*` env vars):
- `gs://mktcrunch-trading-agents-audit` — `audit/audit_events.jsonl`
- `gs://mktcrunch-trading-agents-data` — `data/*.json`, `risk_state/*.json`, `learning/*.json`

Add the printed `GCS_*` vars to `.env`. Re-upload anytime:

```bash
python scripts/sync_history_to_gcs.py
```

The dashboard server hydrates from GCS on startup; jobs sync back after each trace.

## Deploy

```bash
export GCP_PROJECT=your-project-id
export GCP_REGION=us-central1

# Create secrets in Secret Manager first (MC keys, Alpaca keys, GEMINI, SCHEDULER_SECRET, etc.)
chmod +x deploy/setup_cloud_run.sh
./deploy/setup_cloud_run.sh
```

## Dashboard & audit trail

After deploy, open:

```
https://YOUR-SERVICE-URL/dashboard
```

### Memorable URL (custom domain)

The default Cloud Run URL (`trading-agents-XXXX-uc.a.run.app`) cannot be shortened — the hash is assigned by GCP.

Map a subdomain you own (recommended: **`twin-ledger.marketcrunch.ai`**):

```bash
export GCP_PROJECT=your-project-id
export GCP_REGION=us-central1
export CUSTOM_DOMAIN=twin-ledger.marketcrunch.ai
./deploy/setup_custom_domain.sh
```

Then add the DNS records from `gcloud beta run domain-mappings describe` at your registrar. Google provisions HTTPS automatically.

Dashboard: `https://twin-ledger.marketcrunch.ai/dashboard`

One-time: verify root domain ownership at [Cloud Run → Domain mappings](https://console.cloud.google.com/run/domains).

Every action is recorded to `data/audit_events.jsonl` with a `trace_id` linking full job sessions:

| Event type | Source |
|------------|--------|
| `job_started` / `job_completed` | Overnight, risk, discovery jobs |
| `ledger_decision` | Baseline + internal signal agents |
| `signal_gemini_query` | Exact Gemini signal input (`payload.query_text`, per-section ticker coverage) |
| `order_placed` | Execution agent |
| `discovery_probe` | Discovery agent |
| `risk_stop_exit` / `risk_eod_exit` / `risk_held` | Risk monitor |
| `agent_action` | All other agent logs |

Set `GCS_AUDIT_BUCKET` on Cloud Run to persist audit log across container restarts.

API endpoints: `GET /api/summary`, `GET /api/events`, `GET /api/trace/{id}`, `GET /api/agent-activity`, `GET /api/learning`, `POST /api/chat`

## Manual triggers

```bash
# Local
python main.py --overnight
python main.py --risk
python main.py --risk --internal --dry-run

# HTTP (with scheduler secret)
curl -X POST -H "X-Scheduler-Secret: $SECRET" https://SERVICE_URL/jobs/overnight
curl -X POST -H "X-Scheduler-Secret: $SECRET" https://SERVICE_URL/jobs/risk
```

### Force overnight run (bypass market-day calendar)

Cron jobs (`internal-overnight-direct`, etc.) use the default calendar gate — they skip weekends and exchange holidays. To **manually re-trigger** after a bug (safe: open-order dedup + risk caps prevent duplicates), call Agent Engine `streamQuery` with **`force: true`**:

```bash
# Load from .env
source deploy/load_deploy_env.sh && load_deploy_env .

PROJECT="${GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
INTERNAL_ID="${AGENT_ENGINE_INTERNAL_ID}"

curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/reasoningEngines/${INTERNAL_ID}:streamQuery" \
  -d '{
    "classMethod": "stream_query",
    "input": {
      "message": "Run daily trading workflow.",
      "user_id": "manual",
      "force": true
    }
  }'
```

Alternatives: set `"skip_calendar": true` instead of `"force"`, or include `force` in the message text. Without one of these, a weekend/holiday trigger returns `skipped: true` in the audit log.

## Required env vars / secrets

- `MC_API_KEY_ID`, `MC_API_SECRET_KEY`
- `ALPACA_API_KEY_BASELINE`, `ALPACA_SECRET_KEY_BASELINE`
- `ALPACA_API_KEY_INTERNAL`, `ALPACA_SECRET_KEY_INTERNAL`
- `GEMINI_API_KEY`
- `DATABENTO_API_KEY` (discovery)
- `SCHEDULER_SECRET` (HTTP job auth)
- `GCS_RISK_STATE_BUCKET` (trailing stop persistence on Cloud Run)
- `DB_*` (optional — only if using Postgres features)

## IAM

The Cloud Run service account needs:
- `roles/run.invoker` (for Scheduler OIDC)
- `roles/secretmanager.secretAccessor`
- `roles/storage.objectAdmin` on the risk state bucket (if using GCS)
