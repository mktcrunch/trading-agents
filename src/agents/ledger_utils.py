"""Shared helpers for Twin Ledger competing agents."""
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

from src import config
from src.models.trading_decision import TradingDecision

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent

MC_CONFIDENCE_MAP = {"High": 0.75, "Medium": 0.6, "Low": 0.4}
GEMINI_FLASH_MODEL = config.GEMINI_FLASH_MODEL
PORTFOLIO_TICKER = "PORTFOLIO"
SIGNAL_JSON_PARSE_ATTEMPTS = max(
    1, int(os.getenv("SIGNAL_JSON_PARSE_ATTEMPTS", "3"))
)


def is_malformed_json_error(exc: BaseException) -> bool:
    """True when Gemini returned text that cannot be parsed as signal JSON."""
    if isinstance(exc, json.JSONDecodeError):
        return True
    msg = str(exc).lower()
    return "expecting value" in msg or "jsondecodeerror" in msg


@dataclass
class SignalLedgerResult:
    decisions: List[TradingDecision]
    no_action_rationale: str = ""


def _clean_json_text(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def parse_ledger_response(text: str) -> List[Dict[str, Any]]:
    data = json.loads(_clean_json_text(text))
    if isinstance(data, dict) and "decisions" in data:
        raw = data["decisions"]
    elif isinstance(data, list):
        raw = data
    else:
        raw = []
    return raw if isinstance(raw, list) else []


def parse_signal_ledger_response(
    text: str,
    valid_tickers: List[str],
) -> SignalLedgerResult:
    """Parse signal output: object with decisions + no_action_rationale, or legacy array."""
    data = json.loads(_clean_json_text(text))
    no_action_rationale = ""
    if isinstance(data, dict):
        no_action_rationale = str(data.get("no_action_rationale") or "").strip()
        raw = data.get("decisions", [])
    elif isinstance(data, list):
        raw = data
    else:
        raw = []

    decisions: List[TradingDecision] = []
    valid = set(valid_tickers)
    for item in raw if isinstance(raw, list) else []:
        decision = TradingDecision.from_dict(item)
        if decision and decision.ticker in valid:
            decisions.append(decision)
    return SignalLedgerResult(
        decisions=decisions,
        no_action_rationale=no_action_rationale,
    )


def parse_trading_decisions(
    text: str,
    valid_tickers: List[str],
) -> List[TradingDecision]:
    return parse_signal_ledger_response(text, valid_tickers).decisions


def emit_signal_ledger_audit(
    agent: "BaseAgent",
    result: SignalLedgerResult,
    competition: Dict[str, Any],
) -> None:
    """Log actionable decisions and portfolio-level no-action rationale to audit."""
    actionable = [d for d in result.decisions if d.action != "HOLD"]
    leaderboard = competition.get("leaderboard") or {}

    agent.log_action(
        f"Ledger decisions: {len(actionable)} actionable / {len(result.decisions)} total | "
        f"Rank {leaderboard.get('your_rank', '?')}/2 "
        f"({leaderboard.get('status', '?')} by "
        f"${leaderboard.get('value_gap_usd', 0):,.2f})",
        data={
            "leaderboard": leaderboard,
            "decisions": [d.to_dict() for d in actionable],
            "no_action_rationale": result.no_action_rationale or None,
        },
    )

    for d in actionable:
        agent.log_action(
            f"  {d.action} {d.ticker} size={d.size_pct:.1%} "
            f"conf={d.confidence:.2f} — {d.rationale[:80]}",
            data={**d.to_dict(), "leaderboard": leaderboard},
            event_type="ledger_decision",
        )

    if not actionable:
        rationale = (result.no_action_rationale or "").strip()
        if not rationale:
            rationale = (
                "No actionable trades suggested; model returned an empty decision set "
                "without an explicit no_action_rationale."
            )
        agent.log_action(
            f"No overnight action: {rationale[:120]}",
            data={
                "action": "HOLD",
                "ticker": PORTFOLIO_TICKER,
                "size_pct": 0.0,
                "confidence": 1.0,
                "rationale": rationale,
                "invalidation": "",
                "competitive_note": (
                    f"Rank {leaderboard.get('your_rank', '?')}/2 "
                    f"({leaderboard.get('status', '?')} by "
                    f"${leaderboard.get('value_gap_usd', 0):,.2f})"
                ),
                "no_action": True,
                "portfolio_level": True,
                "leaderboard": leaderboard,
            },
            event_type="ledger_decision",
        )


def mc_confidence_score(confidence_label: str) -> float:
    return MC_CONFIDENCE_MAP.get(confidence_label, 0.4)
