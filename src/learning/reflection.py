"""Generate and persist agent learning from audit outcomes (Tier 2)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from src import config
from src.agents.ledger_utils import GEMINI_FLASH_MODEL
from src.apis.gemini_client import get_genai_client
from src.learning.analyzer import analyze_agent_outcomes
from src.learning.heuristics import heuristic_for_role
from src.learning.roles import LLM_REFLECTION_ROLES, roles_for_system
from src.learning.store import load_learning, save_learning
from src.logger import setup_logger

logger = setup_logger(__name__)

LOOKBACK_DAYS = config.LEARNING_LOOKBACK_DAYS


def _prior_memory_for_prompt(prior: Dict[str, Any]) -> Dict[str, Any]:
    """Compact prior state for the reflection prompt (not full scorecard/events)."""
    if not prior.get("updated_at"):
        return {}
    return {
        "updated_at": prior.get("updated_at"),
        "lessons_learned": prior.get("lessons_learned") or "",
        "bad_patterns": list(prior.get("bad_patterns") or [])[:5],
        "do_more": list(prior.get("do_more") or [])[:5],
        "prior_scorecard": prior.get("scorecard") or {},
    }


def _parse_reflection_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


async def _llm_reflection(
    system: str,
    role: str,
    analysis: Dict[str, Any],
    heuristic: Dict[str, Any],
    prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not config.LEARNING_USE_LLM or role not in LLM_REFLECTION_ROLES:
        return heuristic

    try:
        client = get_genai_client()
        prior_block = _prior_memory_for_prompt(prior or {})
        prior_section = ""
        if prior_block:
            prior_section = f"""
Previous learning memory (from last refresh — carry forward what still holds, drop what new data contradicts):
{json.dumps(prior_block, indent=2, default=str)}

"""
        prompt = f"""You are a trading agent coach reviewing audit outcomes for the {system} trader's {role} agent.

Analysis JSON (latest {analysis.get('lookback_days', LOOKBACK_DAYS)}-day audit window):
{json.dumps(analysis, indent=2, default=str)}

Draft heuristics:
{json.dumps(heuristic, indent=2)}
{prior_section}Rules:
- Use ONLY facts from Analysis JSON — do not invent pipeline failures.
- If decisions_logged > 0 but decisions_scored == 0, say trades are pending next-day scoring.
- If event counts are > 0, the agent was active — never claim complete inactivity.
- Refresh prior memory: keep durable lessons, update counts/patterns when scorecard changed, remove stale bullets contradicted by new outcomes.
- Keep lessons_learned to 2-3 factual sentences.

Return ONLY JSON:
{{
  "lessons_learned": "2-3 sentences",
  "bad_patterns": ["max 5 bullets"],
  "do_more": ["max 5 bullets"]
}}"""
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        parsed = _parse_reflection_json(response.text or "")
        if parsed.get("lessons_learned"):
            return {
                "lessons_learned": str(parsed.get("lessons_learned", "")),
                "bad_patterns": list(parsed.get("bad_patterns") or [])[:5],
                "do_more": list(parsed.get("do_more") or [])[:5],
            }
    except Exception as e:
        logger.warning(f"LLM reflection failed ({role}/{system}): {e}")
    return heuristic


def _merge_state(
    role: str,
    reflection: Dict[str, Any],
    analysis: Dict[str, Any],
    days: int,
) -> Dict[str, Any]:
    state = {
        **reflection,
        "scorecard": analysis.get("scorecard") or {},
        "lookback_days": days,
        "recent_events": analysis.get("recent_events")
        or analysis.get("scored_decisions")
        or analysis.get("recent_exits")
        or analysis.get("pending_decisions")
        or [],
    }
    if role == "signal":
        state["recent_decisions"] = analysis.get("scored_decisions") or []
        state["pending_decisions"] = analysis.get("pending_decisions") or []
        state["no_action_sessions"] = analysis.get("no_action_sessions") or []
        state["ticker_stats"] = analysis.get("ticker_stats") or {}
    if role == "risk":
        state["recent_exits"] = analysis.get("recent_exits") or []
        state["recent_held"] = analysis.get("recent_held") or []
    return state


async def refresh_agent_learning(
    system: str,
    role: str,
    lookback_days: Optional[int] = None,
) -> Dict[str, Any]:
    days = lookback_days or LOOKBACK_DAYS
    store_system = "internal" if role == "discovery" else system
    prior = load_learning(store_system, role)
    analysis = analyze_agent_outcomes(store_system if role == "discovery" else system, role, days)
    heuristic = heuristic_for_role(role, analysis)
    reflection = await _llm_reflection(system, role, analysis, heuristic, prior=prior)
    state = _merge_state(role, reflection, analysis, days)
    save_learning(store_system, role, state)
    return state


async def refresh_system_learning(
    system: str,
    lookback_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze audit trail, reflect, and persist learning for every crew agent."""
    if not config.LEARNING_ENABLED:
        return {"success": False, "error": "Learning disabled"}

    if system not in ("baseline", "internal"):
        return {"success": False, "error": "system must be baseline or internal"}

    days = lookback_days or LOOKBACK_DAYS
    agents: Dict[str, Any] = {}
    for role in roles_for_system(system):
        try:
            agents[role] = await refresh_agent_learning(system, role, days)
        except Exception as e:
            logger.warning(f"[learning] {system}/{role} failed: {e}")
            agents[role] = {"error": str(e)}

    logger.info(
        f"[learning] Refreshed {system} crew: "
        + ", ".join(f"{r} ok" for r, s in agents.items() if not s.get("error"))
    )
    return {"success": True, "system": system, "agents": agents}


async def refresh_all_traders_learning(lookback_days: Optional[int] = None) -> Dict[str, Any]:
    baseline = await refresh_system_learning("baseline", lookback_days)
    internal = await refresh_system_learning("internal", lookback_days)
    return {"baseline": baseline, "internal": internal}
