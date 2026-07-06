# MarketCrunch Trading Agents

Licensed under the [Apache License 2.0](LICENSE).

**Twin Ledger** runs where static benchmarks fall short: live paper markets on Alpaca, with real prices, real competition, and new decisions every session. Two autonomous agents trade head-to-head and generate outcomes you can audit end to end.

**Baseline** trades on technicals and **Gemini 3.5 Flash** structured reasoning. **Internal** runs the same model, enriched with MC Internal predictions, agentic data discovery, confidence sizing, and hybrid scripted+LLM risk. The scoreboard answers one question: does prediction data actually win?

**LLM:** [Google Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash) — Vertex model ID **`gemini-3.5-flash`** on endpoint **`global`** (`GEMINI_VERTEX_LOCATION=global`; Agent Engine stays in `GCP_REGION=us-central1`). Used via **Google ADK** for Twin Ledger decisions, intraday trailing-stop planning, and DataBento discovery planners.

**Orchestration:** [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) — multi-agent coordinators, `FunctionTool` wrappers for Alpaca/MarketCrunch/DataBento, optional MCP stdio server for external tool connections.

**A2A verification:** `python scripts/verify_a2a.py` — checks ADK sub-agents, Agent Engine REST, and `google-adk[a2a]` wiring.

---

## Systems at a glance

| | Baseline | Internal |
|---|----------|----------|
| **Account** | Alpaca paper #1 | Alpaca paper #2 |
| **Signals** | Gemini 3.5 Flash ledger decisions, Alpaca OHLCV only | Same + **MarketCrunch ensemble forecasts** + discovered features |
| **Sizing** | `size_pct` from LLM (max 25%/position) | Ledger decisions + Kelly on BUY and SHORT entries |
| **Discovery** | — | Agentic DataBento catalog scan, LLM feature formulas |
| **Overnight orders** | OPG limit ±0.5% from close | Same |
| **Intraday risk** | Pure LLM base stop + trailing | Hybrid base stop (scripted + LLM), hybrid trailing, 15-min gate, EOD |

Both agents see a live leaderboard (portfolio value vs competitor) and aligned quant head-to-head metrics (excess return, Sharpe, drawdown, significance) in every competition context and overnight signal payload.

---

## Architecture

```
main.py
├── Discovery (DataBento) ──► approved_datasources.json
├── BaselineSystem ──────────► ADK Workflow → Signal (LlmAgent) → Risk → Execute
└── InternalSystem ──────────► ADK Workflow → Signal (LlmAgent) → Kelly → Execute

ADK (agents/)
├── twin_ledger_baseline/ ── chat coordinator + data/signal task agents
├── twin_ledger_internal/ ─── same + MarketCrunch + DataBento tools
└── src/adk/mcp/server.py ─── MCP tools (Alpaca, MC, DataBento, leaderboard)

Cloud Scheduler (production)
├── 8 jobs → Vertex Agent Engine :streamQuery (overnight, risk, chase-open, chase-midday × 2 traders)

Cloud Run
├── /dashboard  → audit UI, read-only chat, learning panel
└── /api/*      → agent-activity, learning, performance (no LLM required)
```

### ADK agent map

| Agent | Type | Role |
|-------|------|------|
| `twin_ledger_{baseline,internal}` | `LlmAgent` chat | Coordinator — scheduler callbacks + dashboard tools |
| `{system}_data` | `LlmAgent` task | Fetch account, OHLCV, news, MC/discovery context |
| `{system}_signal` | `LlmAgent` task | Structured BUY/SELL/HOLD/CLOSE (+ Google Search grounding on both) |
| Risk / Execution / Monitor | Deterministic Python | Invoked via `FunctionTool` on scheduler path |
| Discovery | Internal only | DataBento probes when stale; GCS cache fallback on failure |

### A2A & Agent Engine

Each trader deploys to **Vertex AI Agent Engine** with `google-adk[a2a]` in requirements. Cloud Scheduler and operators invoke `stream_query` on the engine REST API. In-engine crews delegate via ADK `sub_agents`. Run `python scripts/verify_a2a.py` to confirm engines, requirements, and local ADK tree.

### Grounding & RAG

| Source | Used by |
|--------|---------|
| **Google Search** (`GoogleSearch` tool) | Both signal agents — macro/ETF news (`SIGNAL_GOOGLE_SEARCH_GROUNDING=true`) |
| **Alpaca news API** | Data agents |
| **MarketCrunch ensemble** (50M+ params, 1B+ datapoints) | Internal Signal + sizing + intraday risk gate (private RAG) |
| **DataBento discovery features** | Internal Signal (`approved_datasources.json`) |
| **Learning memory** | Signal + Risk prompts (`learning/{role}_{system}.json`) |
| **GCS audit log** | Dashboard chat tools (retrieval, no vector DB) |

### ADK local dev

```bash
adk web agents                       # interactive multi-agent UI (agents dir is positional)
python -m src.adk.mcp.server         # MCP stdio tool server
```

Env: `USE_ADK=true` (default), `USE_ADK_WORKFLOW=true`, `USE_ADK_MCP=false`, `GOOGLE_GENAI_USE_VERTEXAI=false`

### Baseline daily workflow

1. Fetch Alpaca prices and positions  
2. Gemini structured decisions (`BUY` / `SELL` / `HOLD` / `CLOSE`) with competition context  
3. Pre-trade risk validation (max 25% weight, exposure caps)  
4. Overnight OPG limit orders  
5. Portfolio monitoring  

### Internal daily workflow

Same as baseline, plus:

- MarketCrunch `/analyze` per ticker — proprietary ensemble forecasts (direction, expected move, confidence)  
- DataBento feature enrichment from discovery output  
- Confidence-based sizing on BUY orders from the prediction snapshot  

### MarketCrunch prediction stack (Internal only)

MarketCrunch’s production ensemble is trained at scale — **50M+ parameters** across **1B+ datapoints** — and served via the MarketCrunch API. Twin Ledger does not retrain models; Internal pulls nightly forecasts and injects them into data, signal, sizing, and risk agents. Baseline never calls this layer — that isolation is the experiment control.

### Agentic discovery

Runs daily (or when stale >24h), or on overnight internal with cache fallback if discovery fails:

1. Scan DataBento catalog for equity OHLCV datasets (bar period under 15m excluded; lookback: 90d daily, 45d hourly)  
2. LLM planner picks probe targets; LLM proposes feature formulas per schema  
3. Pre-flight each probe with DataBento `get_billable_size` / `get_cost` — skip downloads **>10 MB** or **>$1** and continue to the next dataset  
4. Three-gate evaluation (MI, IC+t-stat, incremental alpha)  
5. Merge approved sources into `data/approved_datasources.json`  
6. Registry memory in `data/discovery_registry.json`  

Overnight internal: try discovery → on failure use GCS cache → else continue without DataBento enrichment.

### Intraday risk

| Feature | Baseline | Internal |
|---------|----------|----------|
| Base stop | Pure LLM return threshold | ATR × 1.5 scripted floor (fallback -1%) merged with LLM tightening |
| Trailing | Pure LLM activation + profit lock | Scripted 1%/70% floor merged with LLM tightening |
| Overnight risk | Pure LLM entry approval + hard caps | Scripted caps + LLM reject gate |
| Prediction gate | No | 15-min MC API can defer exits |
| EOD exit | -0.95% losers ~3:54 PM ET | ATR-implied or -0.95% |

### Order safety

Before placing overnight orders:

- **Reconcile** — cancel duplicate OPG orders (same symbol/side/price)  
- **Delta placement** — only buy the gap vs pending open orders  
- **Exact duplicate skip** — same symbol/side/qty/price/TIF  
- **Buying power** — skip or scale buys when cash is insufficient  
- **Risk caps** — per-ticker and gross exposure include held + pending + proposed weights (internal: scripted first, then LLM; baseline: LLM then hard caps)

**Re-triggering overnight runs:** by default the pipeline skips non-market days (weekends, Alpaca holidays). Pass **`skip_calendar=true`** (or **`force=true`**) to bypass — safe to re-run because dedup + risk caps prevent duplicate orders. Ways to force:

- Agent Engine `streamQuery` input: `"force": true` alongside the message (see `deploy/README.md`)
- Coordinator tool: `run_daily_trading_workflow(system="internal", skip_calendar=true)`
- Message keyword: include `force` or `retry` in the text

**Post-open chase** (9:35 AM ET): same calendar gate — no chase on weekends/holidays. Overnight limit lookback uses the **last equity session close** (Alpaca calendar), not calendar Friday. Volatility is checked **before** cancelling open limits; limits stay live if the gate fails.

---

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env   # fill in API keys
python test_connections.py   # live API smoke (requires .env)
pytest tests/ -q             # offline unit/smoke tests (no API keys)
```

### Run locally

```bash
python main.py                    # discovery + both systems in parallel
python main.py --baseline         # baseline only
python main.py --baseline --dry-run   # full pipeline, no Alpaca orders
python main.py --internal         # discovery (if stale) + internal
python main.py --internal --dry-run
python main.py --overnight --dry-run  # both traders, no orders
python main.py --discovery        # DataBento discovery only
python main.py --overnight        # EOD job: discovery + both systems
python main.py --risk             # intraday risk (both accounts)
python main.py --reconcile-orders # cancel duplicate OPG orders
python main.py --serve            # HTTP server + dashboard on :8080
```

Dashboard: `http://localhost:8080/dashboard`

---

## Audit trail

Every action is logged to `data/audit_events.jsonl` with trace IDs linking full job sessions:

- `ledger_decision`, `order_placed`, `order_skipped`, `order_cancelled_duplicate`
- `ledger_decision` on no-trade nights includes `no_action_rationale` (ADK structured output) plus optional per-ticker `hold_decisions`
- `signal_gemini_query` — exact Gemini input for the signal step (`query_text`, `coverage` counts)  
- `discovery_probe`, `risk_rejected`, `risk_stop_exit`, `risk_eod_exit`, `job_started` / `job_completed`  

API: `GET /api/summary`, `GET /api/events`, `GET /api/trace/{id}`, `GET /api/performance`, `GET /api/market-clock`, `GET /api/agent-activity`, `GET /api/learning`, `POST /api/chat`

`GET /api/performance` returns live Alpaca equity, portfolio history, and aligned quant metrics (mean daily alpha, Sharpe with 4.25% risk-free rate, max drawdown, total-return bootstrap significance). Excess return card shows cumulative returns plus compound-annualized cumulative per desk, **β vs SPY**, and ann. excess `(1+r)^(252/n)−1`, grounded with **SPY ann. return from Alpaca daily bars since first trade (June 9, 2026)**. Both signal agents are instructed to beat SPY and read `metrics.benchmark.spy` / `beta_spy`. Daily delta card highlights mean daily alpha (paired closes); boxes show each desk’s avg daily and run-rate annualized return (avg × 252); live today returns are secondary. All comparison metrics are stored as **Internal − Baseline**; agents read `for_you` / `perspectives.*` (positive = favorable to that desk). Dashboard chat exposes the same via `get_performance_metrics(hours=720, perspective="baseline"|"internal")`.

`order_placed` rows returned by `/api/events`, `/api/trace/{id}`, and `get_recent_trading_activity` are annotated with live Alpaca fields (`alpaca_status`, `alpaca_is_active`, …) so manual cancels show as `canceled` even though the audit only records placement.

Persistent storage uses two GCS buckets — create them with `./deploy/setup_gcs.sh`, then set `GCS_AUDIT_BUCKET` and `GCS_DATA_BUCKET` in `.env`.

---

## Project structure

```
trading-agents/
├── main.py                 # CLI entry
├── server.py               # Cloud Run HTTP + dashboard
├── src/
│   ├── agents/             # Data, signal, risk, execution, discovery agents
│   ├── systems/            # baseline_system.py, internal_system.py
│   ├── strategies/         # Allocator, order manager, order dedup
│   ├── apis/               # Alpaca, MarketCrunch, DataBento, grounding, price fetcher
│   ├── learning/           # Per-agent learning loops (audit → reflection → prompt)
│   ├── discovery/          # Catalog, planner, evaluator, feature formulas
│   ├── risk/               # Intraday monitor, trailing planner
│   ├── audit/              # Event tracer + dashboard store
│   ├── analytics/          # Twin Ledger quant metrics (Sharpe, drawdown, significance)
│   ├── dashboard/          # Audit UI
│   ├── models/             # TradingDecision, Position, Order, Signal
│   ├── config.py
│   └── logger.py
├── data/                   # approved_datasources.json, audit_events.jsonl, risk state
├── deploy/                 # Cloud Run + Scheduler setup
├── tests/                  # pytest smoke tests (config, ADK imports, grounding)
├── docs/                   # optional local notes (not in public repo)
├── scripts/                # verify_a2a.py, GCS sync helpers
├── LICENSE
├── logs/
└── requirements.txt
```

---

## Configuration

Environment variables (`.env`):

| Variable | Purpose |
|----------|---------|
| `MC_API_KEY_ID`, `MC_API_SECRET_KEY` | MarketCrunch API |
| `MC_API_CACHE_TTL_SEC` | In-process MC `/analyze` cache TTL in seconds (default `900`; `0` disables) |
| `ALPACA_API_KEY_BASELINE`, `ALPACA_SECRET_KEY_BASELINE` | Baseline paper account |
| `ALPACA_API_KEY_INTERNAL`, `ALPACA_SECRET_KEY_INTERNAL` | Internal paper account |
| `GEMINI_API_KEY` | Optional — only if not using Vertex (`GOOGLE_GENAI_USE_VERTEXAI=false`) |
| `GEMINI_FLASH_MODEL` | Default `gemini-3.5-flash` |
| `GEMINI_VERTEX_LOCATION` | Default `global` (Vertex Gemini endpoint; separate from `GCP_REGION`) |
| `DATABENTO_API_KEY` | Discovery pipeline |
| `DATABENTO_DISCOVERY_ENABLED` | `true` (default). Set `false` to skip catalog scans and paid probes; internal uses cached `approved_datasources.json` from GCS |
| `DB_*` | PostgreSQL (optional) |
| `SCHEDULER_SECRET` | Cloud Run job auth |
| `GCS_RISK_STATE_BUCKET`, `GCS_AUDIT_BUCKET` | Cloud Run persistence |
| `LEARNING_ENABLED`, `LEARNING_LOOKBACK_DAYS` | Agent learning loops |
| `SIGNAL_GOOGLE_SEARCH_GROUNDING` | Google Search on both signal agents (default `true`) |
| `DASHBOARD_CHAT_READ_ONLY` | Dashboard chat cannot place orders |
| `AGENT_ENGINE_BASELINE_ID`, `AGENT_ENGINE_INTERNAL_ID` | Vertex Agent Engine resource IDs |

Trading universe and thresholds live in `src/config.py`:

```python
TRADING_UNIVERSE in `src/config.py` — 12 liquid ETFs across equity, sector, intl, rates, credit, commodities
BASELINE_CONFIG / INTERNAL_CONFIG  # twin_ledger, confidence, Kelly
ORDER_CONFIG                       # limit offset, dedup, reconcile
DISCOVERY_CONFIG                   # gates, probes/day, LLM planner
BASELINE_RISK_CONFIG / INTERNAL_RISK_CONFIG
```

---

## Cloud deployment

See [deploy/README.md](deploy/README.md) for Cloud Run, Cloud Scheduler (4:10 PM overnight + 5-min risk window), secrets, and IAM.

```bash
export GCP_PROJECT=your-project
./deploy/setup_cloud_run.sh
```

---

## Troubleshooting

```bash
python test_connections.py
```

| Issue | Fix |
|-------|-----|
| `pytest` failures on model ID | Ensure `.env` has `GEMINI_FLASH_MODEL=gemini-3.5-flash` |
| `ModuleNotFoundError: src` | Run from repo root with venv active |
| Duplicate open orders | `python main.py --reconcile-orders` |
| Stale discovery | `python main.py --discovery --force` |
| Logs | `logs/trading_system.log` (local) or Cloud Logging (deployed) |
