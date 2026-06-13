"""Custom AdkApp: scheduler-direct pipelines with non-empty stream_query."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterator, Optional, Union

from vertexai.preview.reasoning_engines.templates.adk import AdkApp

from src.adk.agents.scheduler_callbacks import (
    _CHASE_PHRASES,
    _DAILY_PHRASES,
    _RISK_PHRASES,
    _format_result,
    _matches,
    resolve_skip_calendar,
)
from src.logger import setup_logger

logger = setup_logger(__name__)


def _message_text(message: Union[str, Dict[str, Any], Any]) -> str:
    if isinstance(message, str):
        return message.strip().lower()
    if isinstance(message, dict):
        parts = message.get("parts") or []
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict):
                texts.append(part.get("text") or "")
            elif hasattr(part, "text"):
                texts.append(getattr(part, "text", "") or "")
        return "".join(texts).strip().lower()
    return ""


def _run_async(coro):
    """Run async coroutine from sync stream_query (Agent Engine uses sync generator)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class SchedulerDirectAdkApp(AdkApp):
    """Run Cloud Scheduler phrases without LLM; always yield at least one stream event."""

    def __init__(self, *, system: str, agent, **kwargs):
        self._twin_system = system
        super().__init__(agent=agent, **kwargs)

    def _yield_result(self, result: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        from google.adk.events.event import Event
        from google.genai import types
        from vertexai.agent_engines import _utils

        event = Event(
            author=f"twin_ledger_{self._twin_system}",
            content=types.Content(
                role="model",
                parts=[types.Part(text=_format_result(result))],
            ),
        )
        yield _utils.dump_event_for_json(event)

    def stream_query(
        self,
        *,
        message: Union[str, Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        text = _message_text(message)

        if _matches(text, _DAILY_PHRASES):
            skip_calendar = resolve_skip_calendar(text, **kwargs)
            logger.info(
                f"[scheduler] Direct daily pipeline for {self._twin_system} "
                f"(stream_query, skip_calendar={skip_calendar})"
            )
            from src.adk.tools.alpaca_tools import run_daily_trading_workflow

            result = _run_async(
                run_daily_trading_workflow(
                    system=self._twin_system,
                    skip_calendar=skip_calendar,
                )
            )
            yield from self._yield_result(result)
            return

        if _matches(text, _RISK_PHRASES):
            logger.info(f"[scheduler] Direct risk check for {self._twin_system} (stream_query)")
            from src.adk.tools.alpaca_tools import run_intraday_risk_check

            result = _run_async(run_intraday_risk_check(system=self._twin_system))
            yield from self._yield_result(result)
            return

        if _matches(text, _CHASE_PHRASES):
            logger.info(f"[scheduler] Direct post-open chase for {self._twin_system} (stream_query)")
            from src.adk.tools.alpaca_tools import run_post_open_chase

            result = _run_async(run_post_open_chase(system=self._twin_system))
            yield from self._yield_result(result)
            return

        yield from super().stream_query(
            message=message,
            user_id=user_id,
            session_id=session_id,
            run_config=run_config,
            **kwargs,
        )


def build_scheduler_direct_app(agent, system: str) -> SchedulerDirectAdkApp:
    return SchedulerDirectAdkApp(agent=agent, system=system)
