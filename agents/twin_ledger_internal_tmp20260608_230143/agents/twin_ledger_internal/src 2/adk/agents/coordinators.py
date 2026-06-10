"""Multi-agent ADK coordinators for adk web and competition demos."""
from google.adk.agents import LlmAgent

from src.adk.model import adk_model
from src.adk.prompts.baseline import BASELINE_COORDINATOR_INSTRUCTION
from src.adk.prompts.internal import INTERNAL_COORDINATOR_INSTRUCTION
from src.adk.tools import baseline_data_tools, internal_data_tools
from .signal_agents import build_baseline_signal_agent, build_internal_signal_agent


def _data_agent(name: str, system: str, tools: list, competitor: str) -> LlmAgent:
    return LlmAgent(
        name=name,
        model=adk_model(),
        instruction=(
            f"You are the {system} data agent for Twin Ledger. "
            f"Fetch account info, positions, market data, and leaderboard context "
            f"using your tools. Competitor: {competitor}. "
            "Return a concise JSON summary of fetched data."
        ),
        tools=tools,
        mode="task",
    )


def build_baseline_root_agent() -> LlmAgent:
    """Chat coordinator + task sub-agents for Baseline Twin Ledger."""
    return LlmAgent(
        name="twin_ledger_baseline",
        model=adk_model(),
        instruction=BASELINE_COORDINATOR_INSTRUCTION,
        sub_agents=[
            _data_agent(
                "baseline_data",
                "baseline",
                baseline_data_tools(),
                "Internal Trader",
            ),
            build_baseline_signal_agent(),
        ],
        mode="chat",
    )


def build_internal_root_agent() -> LlmAgent:
    """Chat coordinator + task sub-agents for Internal Twin Ledger."""
    return LlmAgent(
        name="twin_ledger_internal",
        model=adk_model(),
        instruction=INTERNAL_COORDINATOR_INSTRUCTION,
        sub_agents=[
            _data_agent(
                "internal_data",
                "internal",
                internal_data_tools(),
                "Baseline Trader",
            ),
            build_internal_signal_agent(),
        ],
        mode="chat",
    )
