# Twin Ledger — Competition Submission Notes

Use this file for hackathon / Track 3 writeups, video narration, and judge-facing copy.
Run `python scripts/verify_a2a.py` before submitting to refresh Agent Engine resource names.

---

## One-line pitch

**Can proprietary prediction data beat technicals-only LLM trading in live paper markets?** Twin Ledger runs two autonomous traders head-to-head on Alpaca with full GCS audit replay.

---

## Technical implementation (ADK + A2A + GCP)

Twin Ledger uses **Google ADK** end-to-end:

| Component | Implementation |
|-----------|----------------|
| **Coordinator** | Chat-mode `LlmAgent` (`twin_ledger_baseline` / `twin_ledger_internal`) |
| **Specialists** | Task-mode `LlmAgent` sub-agents: Data, Signal |
| **Tools** | ADK `FunctionTool` — Alpaca, MarketCrunch, DataBento, audit/dashboard |
| **Scheduler path** | `before_agent_callback` phrase routing → deterministic workflows (reliable cron) |
| **Production LLM** | `gemini-3.5-flash` via Vertex **`global`** endpoint (`GEMINI_VERTEX_LOCATION=global`) |
| **Runtime** | Vertex AI **Agent Engine** × 2 (Baseline + Internal) |
| **A2A packaging** | `google-adk[a2a]` in Agent Engine requirements; engines expose `stream_query` + session APIs |

**A2A & interoperability:** Each competing trader is a separately deployable Agent Engine resource invoked by Cloud Scheduler and the dashboard. External systems discover and message agents through Agent Engine's standard APIs (same surface used by the A2A-on-Agent-Engine pattern). In-engine specialist crews use ADK `sub_agents` for delegation.

Verify: `python scripts/verify_a2a.py`

---

## Grounding & RAG

| Layer | Baseline | Internal |
|-------|----------|----------|
| **Google Search grounding** | Baseline Signal — macro/ETF news via `GoogleSearch` tool | — |
| **Private data RAG** | Alpaca news API + technicals + learning memory | MC predictions + DataBento discovery features + learning memory |
| **Audit retrieval** | Dashboard chat tools read GCS `audit_events.jsonl` | Same + discovery registry |

**Baseline overnight path:** `prefer_direct=True` Gemini call with `types.Tool(google_search=types.GoogleSearch())` plus injected learning block from prior outcomes.

**Internal overnight path:** Gate-approved vendor features from `approved_datasources.json` and MC `/analyze` predictions are retrieved and injected into the signal prompt — private-data RAG without a separate vector DB.

---

## Agent crew (6 + Discovery)

| Role | Baseline | Internal |
|------|----------|----------|
| Coordinator | Job routing, chat | Same |
| Data | OHLCV, technicals, news | + MC predictions, vendor features |
| Signal | Gemini + Google Search | + prediction context |
| Risk | Deterministic path + LLM trailing | + prediction gate on stops |
| Execution | Limit orders, chase | + confidence sizing |
| Monitor | Portfolio snapshots | Same |
| Discovery | — | MC Internal IP catalog probes |

Learning memory refreshes for **all crew roles** at the start of each overnight pipeline (`learning/{role}_{system}.json` in GCS).

---

## Business case

- **Problem:** Portfolio managers manually A/B test strategies; outcomes are slow and non-reproducible.
- **Solution:** Twin Ledger automates a controlled experiment — identical infra, different signal stacks, immutable audit trail.
- **Metric:** Live paper-trading leaderboard (portfolio value vs competitor) with per-decision rationale in GCS.
- **Takeaway (fill with live numbers from dashboard):** Internal's MC-enriched stack vs Baseline technicals-only — quantify from Performance tab and `/api/summary` before submission.

---

## Video checklist (≤ 2 min, English + subtitles)

1. **Problem (20s):** Prediction data vs technicals-only.
2. **Platform (70s):** Dashboard Performance tab (both curves) → Decisions trace → Agent chip query → Architecture tab (scheduler + crew).
3. **GCP proof (20s):** Cloud Scheduler jobs or Agent Engine console + Cloud Run dashboard URL.
4. **Result (10s):** Quantified paper-trading delta from your audit window.

---

## Suggested submission paragraph (copy-paste)

> Twin Ledger is a live head-to-head paper-trading experiment on Google Cloud. Two Vertex AI Agent Engine deployments (Baseline and Internal) compete on separate Alpaca accounts using Google ADK multi-agent coordinators, FunctionTools, and deterministic scheduler workflows. Baseline uses Gemini 3.5 Flash with **Grounding with Google Search** and technical indicators; Internal adds proprietary prediction data and agentic DataBento discovery. Every action is logged to GCS with trace IDs and replayed in a Cloud Run dashboard with read-only ADK chat. The system answers one question: does prediction-enriched reasoning deliver measurable alpha over technicals-only trading?
