"""Interactive coordinator chat for the dashboard (local ADK or Vertex Agent Engine)."""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src import config
from src.adk.model import configure_genai_env
from src.logger import setup_logger

logger = setup_logger(__name__)

_session_service = InMemorySessionService()


_EXECUTION_PHRASES = (
    "run daily trading workflow",
    "daily trading workflow",
    "run intraday risk check",
    "intraday risk check",
    "intraday risk",
    "run post-open chase",
    "post-open chase",
    "post open chase",
    "execute_trading_decisions",
    "place order",
    "place orders",
    "run overnight",
)


def _looks_like_execution_request(message: str) -> bool:
    text = message.strip().lower()
    return any(p in text for p in _EXECUTION_PHRASES)


def chat_status() -> Dict[str, Any]:
    """Return whether dashboard chat is available and which backend is active."""
    read_only = config.DASHBOARD_CHAT_READ_ONLY
    backend = "local" if read_only else _resolve_backend()
    warning = (
        "Read-only mode: chat can query portfolio and audit history only. "
        "Trading, risk, and chase workflows cannot be triggered from the web app."
        if read_only
        else (
            "Execution enabled: messages matching scheduler phrases "
            "(e.g. 'Run daily trading workflow.') can place real paper trades."
        )
    )
    return {
        "enabled": config.DASHBOARD_CHAT_ENABLED,
        "read_only": read_only,
        "backend": backend,
        "systems": ["baseline", "internal"],
        "vertex_configured": bool(
            config.AGENT_ENGINE_BASELINE_ID and config.AGENT_ENGINE_INTERNAL_ID
        ),
        "execution_blocked": read_only,
        "warning": warning,
    }


def _is_cloud_run() -> bool:
    return bool(os.getenv("K_SERVICE"))


def _resolve_backend() -> str:
    mode = (config.CHAT_BACKEND or "auto").lower()
    if mode == "local":
        return "local"
    if mode == "vertex":
        return "vertex"
    # auto: Cloud Run → Vertex (if engine IDs set); laptop → local ADK
    if _is_cloud_run() and config.AGENT_ENGINE_BASELINE_ID and config.AGENT_ENGINE_INTERNAL_ID:
        return "vertex"
    return "local"


def _vertex_permission_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "permission denied" in msg or "consumer_invalid" in msg or "403" in msg


def _engine_resource(system: str) -> str:
    engine_id = (
        config.AGENT_ENGINE_BASELINE_ID
        if system == "baseline"
        else config.AGENT_ENGINE_INTERNAL_ID
    )
    if not engine_id:
        raise ValueError(f"Agent Engine ID not configured for {system}")
    project = config.GCP_PROJECT
    region = config.GCP_REGION
    if not project:
        raise ValueError("GCP_PROJECT is required for Vertex chat")
    return f"projects/{project}/locations/{region}/reasoningEngines/{engine_id}"


def _chunk_to_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        content = chunk.get("content") or chunk.get("output") or {}
        if isinstance(content, dict):
            parts = content.get("parts") or []
            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
            if texts:
                return "".join(texts)
        for key in ("text", "message", "response", "output"):
            if chunk.get(key):
                return str(chunk[key])
        return json.dumps(chunk, default=str)
    if hasattr(chunk, "content") and chunk.content:
        return "".join(
            p.text or "" for p in (chunk.content.parts or []) if hasattr(p, "text")
        )
    return str(chunk)


def _collect_event_text(events: list) -> str:
    parts: List[str] = []
    for event in events:
        text = _chunk_to_text(event)
        if text:
            parts.append(text)
    if not parts:
        return ""
    return parts[-1] if len(parts) == 1 else "\n\n".join(parts)


async def _chat_local(system: str, message: str, session_id: str, user_id: str) -> Dict[str, Any]:
    configure_genai_env()

    if config.DASHBOARD_CHAT_READ_ONLY:
        from src.adk.agents.coordinators import build_dashboard_readonly_agent

        agent = build_dashboard_readonly_agent(system)
        app_name = f"twin_ledger_{system}_dashboard"
    elif system == "baseline":
        from src.adk.agents.coordinators import build_baseline_root_agent

        agent = build_baseline_root_agent()
        app_name = "twin_ledger_baseline"
    elif system == "internal":
        from src.adk.agents.coordinators import build_internal_root_agent

        agent = build_internal_root_agent()
        app_name = "twin_ledger_internal"
    else:
        raise ValueError(f"Invalid system: {system}")

    app = App(name=app_name, root_agent=agent)
    runner = Runner(
        app=app,
        session_service=_session_service,
        auto_create_session=True,
    )
    new_message = types.Content(role="user", parts=[types.Part(text=message)])

    events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        events.append(event)

    reply = _collect_event_text(events)
    return {
        "success": True,
        "system": system,
        "session_id": session_id,
        "backend": "local",
        "read_only": config.DASHBOARD_CHAT_READ_ONLY,
        "reply": reply or "(No text response — check Decisions tab for new audit events.)",
        "event_count": len(events),
    }


def _chat_vertex_sync(system: str, message: str, user_id: str) -> Dict[str, Any]:
    import vertexai
    from vertexai import agent_engines

    configure_genai_env()
    if config.GCP_PROJECT:
        os.environ["GOOGLE_CLOUD_PROJECT"] = config.GCP_PROJECT
    if config.GCP_REGION:
        os.environ["GOOGLE_CLOUD_LOCATION"] = config.GCP_REGION
    vertexai.init(project=config.GCP_PROJECT, location=config.GCP_REGION)

    resource = _engine_resource(system)
    engine = agent_engines.get(resource)

    chunks: List[str] = []
    for chunk in engine.stream_query(message=message, user_id=user_id):
        text = _chunk_to_text(chunk)
        if text:
            chunks.append(text)

    reply = chunks[-1] if chunks else ""
    return {
        "success": True,
        "system": system,
        "backend": "vertex",
        "engine": resource,
        "reply": reply or "(No text response — check Decisions tab for new audit events.)",
        "chunk_count": len(chunks),
    }


async def run_coordinator_chat(
    system: str,
    message: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a message to the baseline or internal coordinator."""
    if not config.DASHBOARD_CHAT_ENABLED:
        return {"success": False, "error": "Dashboard chat is disabled"}

    system = (system or "").strip().lower()
    message = (message or "").strip()
    if system not in ("baseline", "internal"):
        return {"success": False, "error": "system must be baseline or internal"}
    if not message:
        return {"success": False, "error": "message is required"}

    if config.DASHBOARD_CHAT_READ_ONLY and _looks_like_execution_request(message):
        return {
            "success": False,
            "error": (
                "Execution is disabled from the dashboard (DASHBOARD_CHAT_READ_ONLY=true). "
                "Use Cloud Scheduler or operator CLI to run trading workflows."
            ),
            "read_only": True,
            "blocked": True,
        }

    session_id = session_id or str(uuid.uuid4())
    user_id = user_id or f"dashboard_{system}"

    backend = "local" if config.DASHBOARD_CHAT_READ_ONLY else _resolve_backend()
    logger.info(f"Dashboard chat [{backend}] {system}: {message[:80]}...")

    try:
        if backend == "vertex":
            import asyncio

            try:
                result = await asyncio.to_thread(
                    _chat_vertex_sync, system, message, user_id
                )
                result["session_id"] = session_id
                return result
            except Exception as vertex_err:
                if not _is_cloud_run() and _vertex_permission_error(vertex_err):
                    logger.warning(
                        f"Vertex chat unavailable locally ({vertex_err}); using local ADK"
                    )
                    result = await _chat_local(system, message, session_id, user_id)
                    result["fallback_from"] = "vertex"
                    result["fallback_hint"] = (
                        "Vertex Agent Engine call failed locally (often wrong gcloud quota "
                        "project). Using in-process ADK coordinator. For remote engines, run: "
                        "gcloud auth application-default set-quota-project "
                        f"{config.GCP_PROJECT or 'YOUR_PROJECT_ID'}"
                    )
                    return result
                raise
        return await _chat_local(system, message, session_id, user_id)
    except Exception as e:
        logger.exception(f"Dashboard chat failed for {system}")
        return {"success": False, "error": str(e), "system": system, "backend": backend}
