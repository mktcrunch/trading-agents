"""Programmatic ADK Runner for production Twin Ledger jobs."""
import json
from typing import Any, Dict, List, Optional

from google.adk.apps.app import App
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.adk.agents.signal_agents import build_baseline_signal_agent, build_internal_signal_agent
from src.adk.model import configure_genai_env
from src.adk.schemas import TradingDecisionsResponse
from src.models.trading_decision import TradingDecision
from src.logger import setup_logger

logger = setup_logger(__name__)


def _extract_structured_response(events: list) -> Optional[TradingDecisionsResponse]:
    """Pull TradingDecisionsResponse from ADK runner events."""
    for event in reversed(events):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts or []:
                if part.text:
                    try:
                        data = json.loads(part.text)
                        return TradingDecisionsResponse.model_validate(data)
                    except (json.JSONDecodeError, ValueError):
                        continue
        if hasattr(event, "actions") and event.actions:
            for action in getattr(event.actions, "state_delta", {}) or {}:
                pass
    return None


def _decisions_from_schema(response: TradingDecisionsResponse, valid_tickers: List[str]) -> List[TradingDecision]:
    valid = set(valid_tickers)
    decisions = []
    for item in response.decisions:
        decision = TradingDecision.from_dict(item.model_dump())
        if decision and decision.ticker in valid:
            decisions.append(decision)
    return decisions


async def run_signal_agent(
    system: str,
    user_payload: Dict[str, Any],
    session_id: str,
    valid_tickers: List[str],
) -> List[TradingDecision]:
    """Run ADK LlmAgent signal step and return parsed TradingDecision list."""
    configure_genai_env()

    if system == "baseline":
        agent: LlmAgent = build_baseline_signal_agent()
        app_name = "twin_ledger_baseline"
    elif system == "internal":
        agent = build_internal_signal_agent()
        app_name = "twin_ledger_internal"
    else:
        raise ValueError(f"Unknown system: {system}")

    app = App(name=app_name, root_agent=agent)
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service, auto_create_session=True)

    message_text = (
        "Generate today's Twin Ledger trading decisions from this context. "
        "Return only structured decisions per output_schema.\n\n"
        f"{json.dumps(user_payload, indent=2, default=str)}"
    )
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=message_text)],
    )

    events = []
    async for event in runner.run_async(
        user_id=f"twin_ledger_{system}",
        session_id=session_id,
        new_message=new_message,
    ):
        events.append(event)

    response = _extract_structured_response(events)
    if response:
        return _decisions_from_schema(response, valid_tickers)

    # Fallback: scan final model text for JSON array
    for event in reversed(events):
        text = ""
        if hasattr(event, "content") and event.content:
            text = "".join(p.text or "" for p in (event.content.parts or []))
        if not text:
            continue
        try:
            from src.agents.ledger_utils import parse_trading_decisions
            return parse_trading_decisions(text, valid_tickers)
        except Exception:
            continue

    logger.warning("ADK signal agent returned no parseable decisions")
    return []
