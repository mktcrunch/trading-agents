"""Multi-agent ADK coordinators for adk web and competition demos."""
from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool

from src.adk.agents.scheduler_callbacks import build_scheduler_callbacks
from src.adk.model import adk_model
from src.adk.prompts.baseline import BASELINE_COORDINATOR_INSTRUCTION
from src.adk.prompts.internal import INTERNAL_COORDINATOR_INSTRUCTION
from src.adk.tools import baseline_data_tools, internal_data_tools
from src.adk.tools.alpaca_tools import (
    execute_trading_decisions,
    run_daily_trading_workflow,
    run_intraday_risk_check,
    run_post_open_chase,
)
from src.adk.tools.dashboard_tools import (
    get_agent_activity,
    get_agent_learning,
    get_data_discovery_trail,
    get_proprietary_data_usage,
    get_recent_trading_activity,
    get_trader_status,
)
from src.adk.tools.competition_tools import get_performance_metrics
from .signal_agents import build_baseline_signal_agent, build_internal_signal_agent


def _data_agent(name: str, system: str, tools: list, competitor: str) -> LlmAgent:
    return LlmAgent(
        name=name,
        model=adk_model(),
        instruction=(
            f"You are the {system} data agent for Twin Ledger. "
            f"Fetch account info, positions, market data, recent news, and leaderboard context "
            f"using your tools. Competitor: {competitor}. "
            "Return a concise JSON summary of fetched data."
        ),
        tools=tools,
        mode="task",
    )


def build_baseline_root_agent() -> LlmAgent:
    """Chat coordinator + task sub-agents for Baseline Twin Ledger."""
    before_cb, after_cb = build_scheduler_callbacks("baseline")
    return LlmAgent(
        name="twin_ledger_baseline",
        model=adk_model(),
        instruction=BASELINE_COORDINATOR_INSTRUCTION,
        tools=[
            FunctionTool(get_trader_status),
            FunctionTool(get_performance_metrics),
            FunctionTool(get_recent_trading_activity),
            FunctionTool(get_data_discovery_trail),
            FunctionTool(get_proprietary_data_usage),
            FunctionTool(get_agent_activity),
            FunctionTool(run_daily_trading_workflow),
            FunctionTool(execute_trading_decisions),
            FunctionTool(run_intraday_risk_check),
            FunctionTool(run_post_open_chase),
        ],
        sub_agents=[
            _data_agent(
                "baseline_data",
                "baseline",
                baseline_data_tools(),
                "Internal Trader",
            ),
            build_baseline_signal_agent(),
        ],
        before_agent_callback=before_cb,
        after_agent_callback=after_cb,
        mode="chat",
    )


_DASHBOARD_READONLY_INSTRUCTION = """You are the Twin Ledger dashboard assistant for paper-trading analytics.

You are in READ-ONLY mode. You must NEVER place orders or run trading/risk workflows.
Do not call run_daily_trading_workflow, execute_trading_decisions, run_intraday_risk_check,
or run_post_open_chase — those tools are not available to you.

Answer questions about portfolio status, open positions, leaderboard, recent decisions
(with rationale from the audit log), orders, and job history using your tools:
- get_trader_status(system=...)
- get_recent_trading_activity(system=..., hours=72)

For quant head-to-head metrics (same as the Performance dashboard cards):
- get_performance_metrics(hours=720, perspective=<your system>) for the desk-relative view
  (`for_you`: positive favors that desk). Raw comparison.* is always Internal − Baseline.
  Overnight competition context includes quant_head_to_head.for_you automatically.

For data discovery (probes, approvals, rejections, registry runs, gate criteria):
- get_data_discovery_trail(hours=168)

For proprietary / enrichment data used in trading (signal API, discovered features):
- get_proprietary_data_usage(system=..., hours=72)

For per-agent activity (what did risk/execution/signal/data/monitor/coordinator do):
- get_agent_activity(system=..., agent_role=..., hours=24)
  Roles: coordinator, data, signal, risk, execution, monitor, discovery, all.
  This reads the audit log — you do NOT need run_intraday_risk_check to answer
  historical risk questions.

For agent learning memory (lessons from recent audit outcomes for ANY crew agent):
- get_agent_learning(system=..., agent_role=...)
  Roles: coordinator, data, signal, risk, execution, monitor, discovery (internal only), or all.
  Returns lessons_learned, bad_patterns, do_more, scorecard, recent_events.
  Use agent_role=signal (or risk, execution, etc.) when the user asks about one agent's learning.
  IMPORTANT: If scorecard shows decisions_logged > 0 but decisions_scored == 0, trades are
  pending next-day outcome scoring — do NOT claim execution paralysis or pipeline failure.
  If event counts in scorecard are > 0, the agent was active in the window.

Each trader has 6 specialists: Coordinator, Data, Signal, Risk, Execution, Monitor.
Internal also has a 7th Discovery agent (MC Internal IP only — Baseline never runs it).
Always use get_agent_activity when asked what a specific agent did today.

SPECIFIC agent questions (user names one agent, clicks a crew chip, or says agent_role=X):
- Call get_agent_activity with THAT agent_role only — never "all".
- Answer ONLY about focused_agent events. Do NOT list the full crew or other agents' duties
  at the top (no Signal/Risk/Execution blurbs).
- End with one short line: "Also ask about: …" using other_agents_you_can_ask from the tool.

Full crew listing ONLY when the user asks "what agents are available", "list the crew",
"what happened today" (vague), or agent_role=all.

Internal DATA agent: Discovery is a separate Internal-only agent (MC Internal IP) that
runs catalog probes. The Data agent LOADS gate-approved vendor features from discovery
output — report vendor_features_from_discovery and enrichment_events from
get_agent_activity. For probe details use get_data_discovery_trail (agent_role=discovery).

Never claim you cannot report risk activity — use get_agent_activity with agent_role=risk.
Intraday risk: deterministic scheduler path (run_intraday_risk_check) but Gemini plans
trailing stops inside each cycle (trailing_stop_planned events).

Always call the relevant tool before saying information is unavailable. Discovery trails
live in the audit log (discovery_probe events), discovery_registry.json, and
approved_datasources.json — get_data_discovery_trail aggregates all three.

Data discovery approval facts (do not contradict these):
- Discovery is MC Internal proprietary IP — Internal overnight only; Baseline never runs it.
- Discovery is a SEPARATE step from trading; internal overnight runs discovery first
  if approved sources are stale (>24h), then loads cached features for signals.
- Approval is AUTOMATIC inside discovery — three statistical gates (MI, IC/t-stat,
  incremental alpha). No human approval step.
- Baseline formulas (volume_zscore, momentum_5d, range_pct, close_vs_sma20) and
  LLM-proposed formulas are BOTH evaluated on vendor OHLCV bars during probes.
  They are NOT exempt from gates. Failed features have rejected_count > 0 on probes.
- Alpaca OHLCV technicals used by the Baseline trader are a different path and do
  NOT go through the discovery approval pipeline.

Use generic labels in replies: "proprietary signal API", "discovered market features",
"data discovery agent", "vendor OHLCV datasets" — do not refuse discovery questions
if tools return data. Explain rejections using feature_evaluations on each probe: for each feature show
status, failed_gate, metrics (mi, ic, t_stat, incremental_alpha, coverage), and
gates[].passed vs gates[].threshold. Also use approval_criteria.gates for threshold defs.

For "what was discovered today?" always read cache_status from get_data_discovery_trail:
- If cache_status.ran_in_last_24h is false, say no run TODAY, then report the LAST run
  (cache_status.last_run_at, dataset/schema, computed_feature_names).
- If tickers exist but cache_status.gate_approved_for_trading is false, explain that
  features were computed but FAILED approval gates (approved_source_count=0) — do not
  describe them as newly discovered or approved sources.
- Distinguish computed cache values vs gate-approved sources.

If the user asks to trade, run overnight workflow, risk check, or chase orders, explain
that execution is disabled from the web dashboard and must be triggered via Cloud Scheduler
or operator CLI — you cannot do it from chat.
"""


def build_dashboard_readonly_agent(system: str) -> LlmAgent:
    """Read-only coordinator for dashboard chat — no execution tools or scheduler callbacks."""
    if system not in ("baseline", "internal"):
        raise ValueError(f"Invalid system: {system}")

    label = "Baseline" if system == "baseline" else "Internal"
    return LlmAgent(
        name=f"twin_ledger_{system}_dashboard",
        model=adk_model(),
        instruction=(
            f"{_DASHBOARD_READONLY_INSTRUCTION}\n"
            f"You are assisting the {label} trader (system=\"{system}\")."
        ),
        tools=[
            FunctionTool(get_trader_status),
            FunctionTool(get_performance_metrics),
            FunctionTool(get_recent_trading_activity),
            FunctionTool(get_data_discovery_trail),
            FunctionTool(get_proprietary_data_usage),
            FunctionTool(get_agent_activity),
            FunctionTool(get_agent_learning),
        ],
        mode="chat",
    )


def build_internal_root_agent() -> LlmAgent:
    """Chat coordinator + task sub-agents for Internal Twin Ledger."""
    before_cb, after_cb = build_scheduler_callbacks("internal")
    return LlmAgent(
        name="twin_ledger_internal",
        model=adk_model(),
        instruction=INTERNAL_COORDINATOR_INSTRUCTION,
        tools=[
            FunctionTool(get_trader_status),
            FunctionTool(get_performance_metrics),
            FunctionTool(get_recent_trading_activity),
            FunctionTool(get_data_discovery_trail),
            FunctionTool(get_proprietary_data_usage),
            FunctionTool(get_agent_activity),
            FunctionTool(run_daily_trading_workflow),
            FunctionTool(execute_trading_decisions),
            FunctionTool(run_intraday_risk_check),
            FunctionTool(run_post_open_chase),
        ],
        sub_agents=[
            _data_agent(
                "internal_data",
                "internal",
                internal_data_tools(),
                "Baseline Trader",
            ),
            build_internal_signal_agent(),
        ],
        before_agent_callback=before_cb,
        after_agent_callback=after_cb,
        mode="chat",
    )
