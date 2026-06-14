"""Agent Engine stream_query hooks: direct scheduler pipelines + force kwarg handling."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterator, Optional, Type, Union

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

_STREAM_QUERY_RESERVED = frozenset({"force", "skip_calendar"})
_PATCHED_CLASSES: set[Type] = set()


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


def _yield_result(system: str, result: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    from google.adk.events.event import Event
    from google.genai import types
    from vertexai.agent_engines import _utils

    event = Event(
        author=f"twin_ledger_{system}",
        content=types.Content(
            role="model",
            parts=[types.Part(text=_format_result(result))],
        ),
    )
    yield _utils.dump_event_for_json(event)


def _strip_reserved_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in _STREAM_QUERY_RESERVED}


def _system_for_app(self, fallback: str) -> str:
    """Infer baseline/internal from Agent Engine app_name (one deploy per folder)."""
    app_name = str((getattr(self, "_tmpl_attrs", None) or {}).get("app_name") or "").lower()
    if "internal" in app_name:
        return "internal"
    if "baseline" in app_name:
        return "baseline"
    return fallback


def _try_direct_scheduler_run(
    system: str,
    text: str,
    kwargs: Dict[str, Any],
) -> Optional[Iterator[Dict[str, Any]]]:
    if _matches(text, _DAILY_PHRASES):
        skip_calendar = resolve_skip_calendar(text, **kwargs)
        logger.info(
            f"[scheduler] Direct daily pipeline for {system} "
            f"(stream_query, skip_calendar={skip_calendar})"
        )
        from src.adk.tools.alpaca_tools import run_daily_trading_workflow

        result = _run_async(
            run_daily_trading_workflow(system=system, skip_calendar=skip_calendar)
        )
        return _yield_result(system, result)

    if _matches(text, _RISK_PHRASES):
        logger.info(f"[scheduler] Direct risk check for {system} (stream_query)")
        from src.adk.tools.alpaca_tools import run_intraday_risk_check

        result = _run_async(run_intraday_risk_check(system=system))
        return _yield_result(system, result)

    if _matches(text, _CHASE_PHRASES):
        logger.info(f"[scheduler] Direct post-open chase for {system} (stream_query)")
        from src.adk.tools.alpaca_tools import run_post_open_chase

        result = _run_async(run_post_open_chase(system=system))
        return _yield_result(system, result)

    return None


def _adk_app_classes() -> list[Type]:
    """Return AdkApp classes used at runtime (Agent Engine uses agent_engines, not preview)."""
    from vertexai import agent_engines

    classes: list[Type] = [agent_engines.AdkApp]
    try:
        from vertexai.preview.reasoning_engines.templates.adk import AdkApp as PreviewAdkApp

        if PreviewAdkApp not in classes:
            classes.append(PreviewAdkApp)
    except ImportError:
        pass
    return classes


def _make_patched_stream_query(original, fallback_system: str):
    def patched_stream_query(
        self,
        *,
        message: Union[str, Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        system = _system_for_app(self, fallback_system)
        text = _message_text(message)
        direct = _try_direct_scheduler_run(system, text, kwargs)
        if direct is not None:
            yield from direct
            return

        yield from original(
            self,
            message=message,
            user_id=user_id,
            session_id=session_id,
            run_config=run_config,
            **_strip_reserved_kwargs(kwargs),
        )

    return patched_stream_query


def install_agent_engine_stream_query_patch(system: str) -> None:
    """Patch Vertex AdkApp.stream_query on the class Agent Engine api_server actually uses.

    Production wraps root_agent with ``vertexai.agent_engines.AdkApp`` (not preview).
    Intercepts scheduler phrases + consumes ``force`` / ``skip_calendar`` before Runner.run().
    """
    for adk_cls in _adk_app_classes():
        if adk_cls in _PATCHED_CLASSES:
            continue
        original = adk_cls.stream_query
        adk_cls.stream_query = _make_patched_stream_query(original, system)
        _PATCHED_CLASSES.add(adk_cls)
        logger.info(
            f"Installed Agent Engine stream_query patch for {system} "
            f"on {adk_cls.__module__}.{adk_cls.__name__}"
        )


def build_scheduler_direct_app(agent, system: str):
    """Optional explicit AdkApp wrapper (local dev); production uses install_*_patch."""
    from vertexai import agent_engines

    class SchedulerDirectAdkApp(agent_engines.AdkApp):
        def __init__(self, *, system: str, agent, **kwargs):
            self._twin_system = system
            super().__init__(agent=agent, **kwargs)

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
            direct = _try_direct_scheduler_run(self._twin_system, text, kwargs)
            if direct is not None:
                yield from direct
                return

            yield from super().stream_query(
                message=message,
                user_id=user_id,
                session_id=session_id,
                run_config=run_config,
                **_strip_reserved_kwargs(kwargs),
            )

    return SchedulerDirectAdkApp(agent=agent, system=system)
