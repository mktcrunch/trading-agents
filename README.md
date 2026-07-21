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

Twin Ledger runs as three scheduled control loops that share one agent backend: session manager, controller/state machine, context assembly, model gateway, tool registry, and policy layer. Side effects go to broker and data APIs; durable state lives in GCS.

Each loop follows the same control cycle: **plan → think → act → observe → verify → repeat**. LLM components produce structured decisions (think); deterministic tools apply policy and execute (act/verify).

### Shared harness stack

```
┌─────────────────────────────────────────────┐
│  Frontend / IDE                             │
│  adk web · Cloud Run /dashboard · CLI       │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│  Agent Backend (ADK + Vertex Agent Engine)  │
│                                             │
│  1. Task/session manager                    │
│     Cloud Scheduler → streamQuery phrases   │
│     main.py jobs · InMemorySessionService   │
│                                             │
│  2. Agent controller / state machine        │
│     ADK Workflow nodes · RiskMonitor        │
│     Discovery probe loop · TwinLedgerState  │
│                                             │
│  3. Context engine                          │
│     Account, OHLCV, news, leaderboard,      │
│     MC forecasts, approved features,        │
│     learning/{role}_{system}.json           │
│                                             │
│  4. Model gateway                           │
│     Gemini 3.5 Flash (Vertex global)        │
│     Signal / discovery / trailing planners  │
│                                             │
│  5. Tool registry                           │
│     FunctionTools + optional MCP            │
│     Alpaca · MarketCrunch · DataBento       │
│                                             │
│  6. Policy / approval engine                │
│     Risk caps · order dedup · calendar      │
│     Discovery MI / IC / alpha gates         │
│     Dashboard chat = read-only tools        │
└─────────────┬────────────────┬──────────────┘
              │                │
              ▼                ▼
     GCS + audit JSONL     External workers
     learning / registry   (not local shell)
     approved_datasources        │
                                 ├── Alpaca paper
                                 ├── MarketCrunch API
                                 ├── DataBento
                                 └── Google Search
```

Deployment topology: Cloud Scheduler (8 jobs) → Vertex Agent Engine `:streamQuery` for overnight / risk / chase × 2 desks; Cloud Run serves `/dashboard` and `/api/*`. Local: `adk web agents`, `python main.py`, `python -m src.adk.mcp.server`. Deploy detail: [`deploy/README.md`](deploy/README.md).

### Gymnasium correspondence

The loops map onto the [Gymnasium](https://gymnasium.farama.org/) MDP interface: market or catalog as environment, LLM + policy as policy network, scheduler ticks as discrete time steps.

| Gymnasium | Twin Ledger |
|-----------|-------------|
| `env` | Live Alpaca paper market, or DataBento catalog + bars |
| `observation` / `reset()` | Context fetch: account, positions, OHLCV, news, MC, features, learning |
| `action_space` | Overnight: ledger actions (`BUY`/`SELL`/`HOLD`/`CLOSE`/`SHORT`/`COVER` + size). Intraday: hold / exit / trail update. Discovery: probe targets + feature formulas |
| `agent.act(obs)` | Gemini structured output (signal / planner / trailing planner) |
| `env.step(action)` | ExecutionAgent / RiskMonitor closes / DataBento download+eval |
| Reward | PnL, Sharpe, leaderboard / quant H2H; discovery: gate pass (MI, IC, alpha) |
| Episode | One overnight session, one 15-min risk tick, or one discovery run |
| Policy constraints | RiskAgent, order dedup, calendar, cost/size preflight (action masking / safety wrappers) |
| Replay / learning | Audit JSONL + `learning/*.json` reflection into subsequent observations (outcome memory, not gradient RL) |

---

### Harness A — Overnight trading (Baseline + Internal)

**Trigger:** Scheduler ~2:00 PM PT (`Run daily trading workflow`) or `main.py --overnight` / `--baseline` / `--internal`.

**Controller:** ADK `Workflow` — `fetch_context → signal_decisions → risk_and_execute → monitor` (`src/adk/workflows/*_daily.py`).

```
PLAN     Scheduler / coordinator picks desk + job; calendar gate
THINK    {system}_signal LlmAgent → structured ledger (+ Google Search)
ACT      Kelly (internal) / size_pct (baseline) → ExecutionAgent OPG limits
OBSERVE  MonitorAgent metrics; fills land next open
VERIFY   RiskAgent caps, reconcile/dedup/BP before place; audit end_trace
REPEAT   Next market day; learning memory injected into next THINK
```

| Slot | Baseline | Internal |
|------|----------|----------|
| Obs extras | Alpaca OHLCV + news | + MC `/analyze` + DataBento from `approved_datasources.json` |
| Act sizing | LLM `size_pct` (max 25%) | Ledger + Kelly on BUY/SHORT |
| Verify | LLM then hard caps | Scripted caps then LLM reject gate |
| Prefetch | — | Discovery if stale (>24h); GCS cache fallback on failure |

**MarketCrunch (Internal only):** production ensemble (~50M+ params, 1B+ datapoints) via API — injected into data, signal, sizing, and intraday gate. Baseline never calls it (experiment control).

**Order safety (policy before `step`):** reconcile duplicate OPG; delta placement vs pending; exact dup skip; buying-power scale; exposure includes held + pending + proposed.

**Force re-run:** default skips weekends/holidays. Bypass with `force` / `skip_calendar` on `streamQuery`, coordinator tool, or message keywords (`force`/`retry`) — safe because dedup + caps prevent duplicate orders. See `deploy/README.md`.

---

### Harness B — Discovery (Internal only)

**Trigger:** Stale approved sources (>24h) before Internal overnight, or `main.py --discovery`.

**Controller:** Catalog scan → LLM/heuristic planner → per-probe feature planner → download → three-gate evaluator → merge registry (`src/discovery/`).

```
PLAN     DiscoveryPlanner picks dataset/schema targets (registry memory)
THINK    FeaturePlanner proposes formulas per schema
ACT      Preflight billable size/cost → download bars → compute features
OBSERVE  Probe metrics (MI, IC, t-stat, incremental alpha, coverage)
VERIFY   Auto-approve only if all three gates pass (no human step)
REPEAT   Write approved_datasources.json + discovery_registry.json;
         next run prefers never-probed / cooldown-elapsed / re-eval approved
```

Constraints: bar period under 15m excluded; lookback 90d daily / 45d hourly; skip download if **>10 MB** or **>$1**; max probes/day from `DISCOVERY_CONFIG`. Overnight: try discovery → on failure use GCS cache → else trade without DataBento enrichment.

**MDP view:** catalog + registry form state; a probe is an action; gate pass/fail is a sparse reward; the registry is episodic memory for the next planning policy.

---

### Harness C — Intraday risk (+ chase)

**Trigger:** Risk every ~15 min (9-15 ET); chase at open (9:35) and midday. `run_intraday_risk_check` / `run_post_open_chase`.

**Controller:** `RiskMonitor.run_check` (`src/risk/risk_monitor.py`) — observe positions → plan stops → optional LLM trail tighten → verify exit gates → act market closes.

```
PLAN     Load risk_state; identify stop / trail / EOD candidates
THINK    Base-stop + trailing planners (LLM; Internal hybrid with scripted floor)
ACT      Cancel open limits on symbol → market close (or hold)
OBSERVE  Live returns, ATR, MC 15-min (Internal), audit risk_* events
VERIFY   Prediction gate may defer exit (Internal); EOD once/day; vol gate on chase
REPEAT   Next Scheduler tick until session end
```

| Feature | Baseline | Internal |
|---------|----------|----------|
| Base stop | Pure LLM return threshold | ATR × 1.5 scripted floor (fallback -1%) merged with LLM tighten |
| Trailing | Pure LLM activation + lock | Scripted 1%/70% floor merged with LLM tighten |
| Prediction gate | No | 15-min MC can defer exit |
| EOD | -0.95% losers ~3:54 PM ET | ATR-implied or -0.95% |

**Post-open chase:** same calendar gate as overnight. Lookback from **last equity session close** (Alpaca calendar). Volatility checked **before** cancelling limits; calm fails → limits stay live.

**MDP view:** each 15-min job is one `step` in a continuous trading episode; `risk_state` is partial environment state; EOD is a terminal transition for the day.

---

### ADK agent map

| Agent | Type | Role |
|-------|------|------|
| `twin_ledger_{baseline,internal}` | `LlmAgent` chat | Coordinator — scheduler callbacks + dashboard tools |
| `{system}_data` | `LlmAgent` task | Fetch account, OHLCV, news, MC/discovery context |
| `{system}_signal` | `LlmAgent` task | Structured ledger (+ Google Search grounding) |
| Risk / Execution / Monitor | Deterministic Python | Workflow nodes / FunctionTools on scheduler path |
| Discovery | Internal only | Harness B — DataBento probes when stale |

```
main.py / Scheduler
├── Harness B Discovery ──► approved_datasources.json
├── Harness A Baseline ───► ADK Workflow → Signal → Risk → Execute → Monitor
├── Harness A Internal ───► ADK Workflow → Signal → Kelly → Execute → Monitor
└── Harness C Intraday ───► RiskMonitor (+ chase) per desk
```

### A2A & Agent Engine

Each trader deploys to **Vertex AI Agent Engine** (`google-adk[a2a]`). Scheduler/operators call `stream_query`; in-engine crews use ADK `sub_agents`. Verify with `python scripts/verify_a2a.py`. Console: Vertex AI → Agent Engines (project from `.env` `GCP_PROJECT`).

### Grounding & RAG

| Source | Used by |
|--------|---------|
| **Google Search** (`GoogleSearch` tool) | Both signal agents (`SIGNAL_GOOGLE_SEARCH_GROUNDING=true`) |
| **Alpaca news API** | Data agents |
| **MarketCrunch ensemble** | Internal Signal + sizing + intraday gate |
| **DataBento discovery features** | Internal Signal (`approved_datasources.json`) |
| **Learning memory** | Signal + Risk prompts (`learning/{role}_{system}.json`) |
| **GCS audit log** | Dashboard chat tools (retrieval, no vector DB) |

### ADK local dev

```bash
adk web agents                       # interactive multi-agent UI (agents dir is positional)
python -m src.adk.mcp.server         # MCP stdio tool server
```

Env: `USE_ADK=true` (default), `USE_ADK_WORKFLOW=true`, `USE_ADK_MCP=false`, `GOOGLE_GENAI_USE_VERTEXAI=false`

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

`GET /api/performance` returns live Alpaca equity, portfolio history, and aligned quant metrics (mean daily alpha, Sharpe with 4.25% risk-free rate, max drawdown, total-return bootstrap significance). Excess return card shows cumulative returns plus compound-annualized cumulative per desk, **β vs SPY**, and ann. excess `(1+r)^(252/n)−1` with **shared `n` = trading days since first trade (June 9, 2026)** — desk equity history is normalized to US/Eastern session dates and fetched from first trade so it lines up with SPY daily bars (live replaces today’s incomplete bar when the market is open). Both signal agents are instructed to beat SPY and read `metrics.benchmark.spy` / `beta_spy`. Daily delta card highlights mean daily alpha (paired closes); boxes show each desk’s avg daily and run-rate annualized return (avg × 252); live today returns are secondary. All comparison metrics are stored as **Internal − Baseline**; agents read `for_you` / `perspectives.*` (positive = favorable to that desk). Dashboard chat exposes the same via `get_performance_metrics(hours=720, perspective="baseline"|"internal")`.

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

See [deploy/README.md](deploy/README.md) for Cloud Run, Cloud Scheduler (2:00 PM PT overnight + 5-min risk window), secrets, and IAM.

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
