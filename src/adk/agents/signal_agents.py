"""ADK LlmAgent builders for Twin Ledger signal generation."""
from google.adk.agents import LlmAgent
from google.adk.tools.google_search_tool import GoogleSearchTool

from src import config
from src.adk.model import adk_model
from src.adk.prompts.baseline import BASELINE_SIGNAL_INSTRUCTION
from src.adk.prompts.internal import INTERNAL_SIGNAL_INSTRUCTION
from src.adk.schemas import TradingDecisionsResponse
from src.adk.mcp.toolset import optional_mcp_tools
from src.adk.tools import baseline_data_tools, internal_data_tools


def _with_google_search_tools(tools: list) -> list:
    if config.SIGNAL_GOOGLE_SEARCH_GROUNDING:
        tools = list(tools)
        tools.append(GoogleSearchTool(bypass_multi_tools_limit=True))
    return tools


def _baseline_signal_tools() -> list:
    return _with_google_search_tools(baseline_data_tools() + optional_mcp_tools())


def build_baseline_signal_agent() -> LlmAgent:
    """LlmAgent that emits structured Twin Ledger decisions (baseline)."""
    return LlmAgent(
        name="baseline_signal",
        model=adk_model(),
        instruction=BASELINE_SIGNAL_INSTRUCTION,
        tools=_baseline_signal_tools(),
        output_schema=TradingDecisionsResponse,
        mode="task",
    )


def build_internal_signal_agent() -> LlmAgent:
    """LlmAgent that emits structured Twin Ledger decisions (internal)."""
    return LlmAgent(
        name="internal_signal",
        model=adk_model(),
        instruction=INTERNAL_SIGNAL_INSTRUCTION,
        tools=_with_google_search_tools(internal_data_tools() + optional_mcp_tools()),
        output_schema=TradingDecisionsResponse,
        mode="task",
    )
