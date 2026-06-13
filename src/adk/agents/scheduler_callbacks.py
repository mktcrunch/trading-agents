"""Route Cloud Scheduler messages to deterministic tools (both Twin Ledger systems)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from src import config
from src.adk.tools.alpaca_tools import run_daily_trading_workflow
from src.audit.store import load_events
from src.logger import setup_logger

logger = setup_logger(__name__)

_DAILY_PHRASES = ("run daily trading workflow", "daily trading workflow")
_RISK_PHRASES = ("run intraday risk check", "intraday risk check", "intraday risk")
_CHASE_PHRASES = ("run post-open chase", "post-open chase", "post open chase")
_FORCE_PHRASES = ("force", "retry", "re-run", "rerun", "retrigger", "manual")


def wants_force_retry(text: str) -> bool:
    """True when the message text explicitly asks to bypass calendar gating."""
    return any(p in text for p in _FORCE_PHRASES)


def resolve_skip_calendar(text: str = "", **kwargs: Any) -> bool:
    """True only when caller explicitly opts in to bypass the market-day calendar.

    Accepted signals (any one is enough):
    - stream_query input: ``force=true`` or ``skip_calendar=true``
    - message text containing force/retry/re-run/manual
    """
    for key in ("force", "skip_calendar"):
        value = kwargs.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes"):
            return True
    return wants_force_retry(text)


def _user_text(callback_context: CallbackContext) -> str:
    content = callback_context.user_content
    if not content or not content.parts:
        return ""
    return "".join(p.text or "" for p in content.parts).strip().lower()


def _matches(text: str, phrases: tuple[str, ...]) -> bool:
    return any(p in text for p in phrases)


def _format_result(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2, default=str)


def _daily_completed_recently(system: str, within_minutes: int = 5) -> bool:
    """True if a daily job_completed was written for this system very recently."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=within_minutes)
    for ev in load_events(limit=40, system=system):
        if ev.get("event_type") != "job_completed":
            continue
        payload = ev.get("payload") or {}
        if ev.get("job_type") not in ("daily", None) and ev.get("action") not in ("daily", None):
            if "orders_placed" not in payload and "message" not in payload:
                continue
        try:
            ts = datetime.fromisoformat(ev["timestamp"].replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        if ts >= cutoff:
            return True
    return False


def build_scheduler_callbacks(system: str):
    """Factory for before/after agent callbacks on baseline or internal coordinator."""

    async def before_agent_callback(*, callback_context: CallbackContext) -> Optional[types.Content]:
        # Scheduler phrases are handled in SchedulerDirectAdkApp.stream_query so the
        # HTTP stream always yields at least one event (before_agent short-circuit
        # produced an empty generator and crashed ASGI with StopIteration on Py 3.11).
        return None

    async def after_agent_callback(*, callback_context: CallbackContext) -> Optional[types.Content]:
        """Fallback if coordinator ran for a daily message but did not finish execution."""
        if not config.DAILY_COORDINATOR_FALLBACK:
            return None

        text = _user_text(callback_context)
        if not _matches(text, _DAILY_PHRASES):
            return None
        if _daily_completed_recently(system):
            return None

        logger.warning(
            f"[scheduler] Coordinator did not complete daily workflow for {system}; "
            "running deterministic fallback pipeline"
        )
        result = await run_daily_trading_workflow(
            system=system,
            skip_calendar=wants_force_retry(text),
        )
        return types.Content(
            role="model",
            parts=[types.Part(text=_format_result({**result, "fallback": True}))],
        )

    return before_agent_callback, after_agent_callback
