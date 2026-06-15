"""Shared JSON parsing for risk LLM planners."""
from __future__ import annotations

import json
from typing import Any, Dict

from src.agents.ledger_utils import _clean_json_text


def parse_planner_object(text: str) -> Dict[str, Any]:
    """Parse a single JSON object, one-element array, or decisions wrapper."""
    data = json.loads(_clean_json_text(text))
    if isinstance(data, list):
        item = data[0] if data else {}
    elif isinstance(data, dict) and "decisions" in data:
        decisions = data.get("decisions") or []
        item = decisions[0] if isinstance(decisions, list) and decisions else {}
    elif isinstance(data, dict):
        item = data
    else:
        item = {}
    return item if isinstance(item, dict) else {}
