"""Shared helpers for Twin Ledger competing agents."""
import json
import re
from typing import Any, Dict, List

from src import config
from src.models.trading_decision import TradingDecision

MC_CONFIDENCE_MAP = {"High": 0.75, "Medium": 0.6, "Low": 0.4}
GEMINI_FLASH_MODEL = config.GEMINI_FLASH_MODEL


def parse_ledger_response(text: str) -> List[Dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def parse_trading_decisions(
    text: str,
    valid_tickers: List[str],
) -> List[TradingDecision]:
    raw = parse_ledger_response(text)
    decisions = []
    valid = set(valid_tickers)
    for item in raw:
        decision = TradingDecision.from_dict(item)
        if decision and decision.ticker in valid:
            decisions.append(decision)
    return decisions


def mc_confidence_score(confidence_label: str) -> float:
    return MC_CONFIDENCE_MAP.get(confidence_label, 0.4)
