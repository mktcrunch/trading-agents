# MarketCrunch Trading Agents

**Twin Ledger** runs where static benchmarks fall short: live paper markets on Alpaca, with real prices, real competition, and new decisions every session. Two autonomous agents trade head-to-head and generate outcomes you can audit end to end.

**Baseline** trades on technicals and **Gemini 2.5 Flash** structured reasoning. **Internal** runs the same model, enriched with MarketCrunch predictions, agentic DataBento discovery, Kelly sizing, and hybrid scripted+LLM risk. The scoreboard answers one question: does prediction data actually win?

**LLM:** [Google Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models) (`gemini-2.5-flash`) via **Google ADK** (`LlmAgent` + `Workflow`) — Twin Ledger decisions, intraday trailing-stop planning, and DataBento discovery planners.

**Orchestration:** [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) — multi-agent coordinators, `FunctionTool` wrappers for Alpaca/MarketCrunch/DataBento, optional MCP stdio server for external tool connections.

---

## Systems at a glance

| | Baseline | Internal |
|---|----------|----------|
| **Account** | Alpaca paper #1 | Alpaca paper #2 |
| **Signals** | Gemini 2.5 Flash ledger decisions, Alpaca OHLCV only | Same + MC predictions + discovered features |
| **Sizing** | `size_pct` from LLM (max 10%/position) | Ledger decisions + Kelly on BUYs |
| **Discovery** | — | Agentic DataBento catalog scan, LLM feature formulas |
| **Overnight orders** | OPG limit ±0.5% from close | Same |
| **Intraday risk** | Fixed -1% stop, pure LLM trailing | ATR stops, hybrid trailing, 15-min prediction gate, EOD exit |

Both agents see a live leaderboard (portfolio value vs competitor) in every Twin Ledger prompt.

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

Cloud Run (optional)
├── 4:10 PM ET  → /jobs/overnight
├── */5 9–15 ET → /jobs/risk
└── /dashboard  → audit trail UI
```

### ADK local dev

```bash
adk web agents                       # interactive multi-agent UI (agents dir is positional)
python -m src.adk.mcp.server         # MCP stdio tool server
```

Env: `USE_ADK=true` (default), `USE_ADK_WORKFLOW=true`, `USE_ADK_MCP=false`, `GOOGLE_GENAI_USE_VERTEXAI=false`

### Baseline daily workflow

1. Fetch Alpaca prices and positions  
2. Gemini structured decisions (`BUY` / `SELL` / `HOLD` / `CLOSE`) with competition context  
3. Pre-trade risk validation (max 10% weight, exposure caps)  
4. Overnight OPG limit orders  
5. Portfolio monitoring  

### Internal daily workflow

Same as baseline, plus:

- MarketCrunch `/analyze` per ticker  
- DataBento feature enrichment from discovery output  
- Kelly allocator sizes BUY orders from MC confidence  

### Agentic discovery

Runs daily (or when stale >24h):

1. Scan DataBento catalog for equity OHLCV datasets  
2. LLM planner picks probe targets; LLM proposes feature formulas per schema  
3. Three-gate evaluation (MI, IC+t-stat, incremental alpha)  
4. Merge approved sources into `data/approved_datasources.json`  
5. Registry memory in `data/discovery_registry.json`  

### Intraday risk

| Feature | Baseline | Internal |
|---------|----------|----------|
| Base stop | Fixed -1% | ATR × 1.5 (fallback -1%) |
| Trailing | Pure LLM activation + profit lock | Scripted 1%/70% floor merged with LLM tightening |
| Prediction gate | No | 15-min MC API can defer exits |
| EOD exit | -0.95% losers ~3:54 PM ET | ATR-implied or -0.95% |

### Order safety

Before placing overnight orders:

- **Reconcile** — cancel duplicate OPG orders (same symbol/side/price)  
- **Delta placement** — only buy the gap vs pending open orders  
- **Exact duplicate skip** — same symbol/side/qty/price/TIF  
- **Buying power** — skip or scale buys when cash is insufficient  

---

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env   # fill in API keys
python test_connections.py
```

### Run locally

```bash
python main.py                    # discovery + both systems in parallel
python main.py --baseline         # baseline only
python main.py --internal         # discovery (if stale) + internal
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
- `discovery_probe`, `risk_stop_exit`, `risk_eod_exit`, `job_started` / `job_completed`  

API: `GET /api/summary`, `GET /api/events`, `GET /api/trace/{id}`

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
│   ├── apis/               # Alpaca, MarketCrunch, DataBento, price fetcher
│   ├── discovery/          # Catalog, planner, evaluator, feature formulas
│   ├── risk/               # Intraday monitor, trailing planner
│   ├── audit/              # Event tracer + dashboard store
│   ├── dashboard/          # Audit UI
│   ├── models/             # TradingDecision, Position, Order, Signal
│   ├── config.py
│   └── logger.py
├── data/                   # approved_datasources.json, audit_events.jsonl, risk state
├── deploy/                 # Cloud Run + Scheduler setup
├── logs/
└── requirements.txt
```

---

## Configuration

Environment variables (`.env`):

| Variable | Purpose |
|----------|---------|
| `MC_API_KEY_ID`, `MC_API_SECRET_KEY` | MarketCrunch API |
| `ALPACA_API_KEY_BASELINE`, `ALPACA_SECRET_KEY_BASELINE` | Baseline paper account |
| `ALPACA_API_KEY_INTERNAL`, `ALPACA_SECRET_KEY_INTERNAL` | Internal paper account |
| `GEMINI_API_KEY` | Gemini 2.5 Flash (decisions, trailing, discovery) |
| `DATABENTO_API_KEY` | Discovery pipeline |
| `DB_*` | PostgreSQL (optional) |
| `SCHEDULER_SECRET` | Cloud Run job auth |
| `GCS_RISK_STATE_BUCKET`, `GCS_AUDIT_BUCKET` | Cloud Run persistence |

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
| `ModuleNotFoundError: src` | Run from repo root with venv active |
| Duplicate open orders | `python main.py --reconcile-orders` |
| Stale discovery | `python main.py --discovery --force` |
| Logs | `logs/trading_system.log` (local) or Cloud Logging (deployed) |
