"""Programmatic ADK Runner for production Twin Ledger jobs."""
from typing import Any, Dict, List

from src.adk.model import configure_genai_env
from src.agents.ledger_utils import SignalLedgerResult
from src.logger import setup_logger

logger = setup_logger(__name__)


async def run_signal_agent(
    system: str,
    user_payload: Dict[str, Any],
    session_id: str,
    valid_tickers: List[str],
) -> SignalLedgerResult:
    """Run signal step and return parsed TradingDecision list.

    Uses direct Gemini (prefer_direct=True) — ADK Runner requires chat-mode root
    agents; task-mode signal LlmAgents are used as coordinator sub_agents only.
    """
    configure_genai_env()

    if system == "baseline":
        from src.agents.signal_agent_baseline import BaselineSignalAgent

        agent = BaselineSignalAgent()
        return await agent.make_trading_decisions(
            user_payload.get("technical_data") or {},
            user_payload.get("competition"),
            prefer_direct=True,
            news_data=user_payload.get("news_data"),
        )

    if system == "internal":
        from src.agents.signal_agent_internal import InternalSignalAgent

        agent = InternalSignalAgent()
        return await agent.make_trading_decisions(
            user_payload.get("technical_data") or {},
            user_payload.get("mc_predictions") or {},
            user_payload.get("competition"),
            databento_sources=user_payload.get("databento_sources"),
            prefer_direct=True,
            news_data=user_payload.get("news_data"),
        )

    raise ValueError(f"Unknown system: {system}")
