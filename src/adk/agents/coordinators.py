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
    get_recent_trading_activity,
    get_trader_status,
)
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
            FunctionTool(get_recent_trading_activity),
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
            FunctionTool(get_recent_trading_activity),
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
            FunctionTool(get_recent_trading_activity),
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
