"""LLM overnight entry risk planner — baseline pure LLM, internal hybrid over scripted checks."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src import config
from src.apis.gemini_client import get_genai_client
from src.agents.ledger_utils import GEMINI_FLASH_MODEL
from src.logger import setup_logger
from src.risk.planner_parse import parse_planner_object
from src.risk.trailing_planner import position_side_label

logger = setup_logger(__name__)


class OvernightRiskPlanner:
    """Gemini planner for overnight BUY/SHORT entry approval."""

    def __init__(self, system: str = "baseline"):
        if system not in ("baseline", "internal"):
            raise ValueError(f"Invalid system: {system}")
        self.system = system
        self.cfg = (
            config.BASELINE_RISK_CONFIG
            if system == "baseline"
            else config.INTERNAL_RISK_CONFIG
        )
        self.client = get_genai_client()

    def _build_baseline_prompt(
        self,
        ticker: str,
        side: str,
        weight: float,
        portfolio_context: Dict[str, Any],
    ) -> str:
        from src import config

        max_pct = int(config.MAX_POSITION_SIZE_PCT * 100)
        return f"""You are the overnight risk manager for the BASELINE ETF agent (pure LLM — no scripted 1%/70% rules).

Decide whether to APPROVE this entry for tomorrow's session.

Proposal:
- Ticker: {ticker}
- Side: {side.upper()}
- Size: {weight * 100:.1f}% of portfolio

Portfolio context:
{json.dumps(portfolio_context, indent=2)}

Hard limits (never approve above these): {max_pct}% per ticker, 8 positions, 125% gross exposure.

Return ONLY JSON:
{{"approved": true, "rationale": "one sentence"}}"""

    def _build_internal_prompt(
        self,
        ticker: str,
        side: str,
        weight: float,
        portfolio_context: Dict[str, Any],
        scripted_passed: bool,
    ) -> str:
        from src import config

        max_pct = int(config.MAX_POSITION_SIZE_PCT * 100)
        return f"""You are the overnight risk manager for the INTERNAL ETF agent (scripted checks + LLM).

Scripted risk checks already {"PASSED" if scripted_passed else "FAILED"} for this entry.
Scripted rules enforce: {max_pct}% per ticker, max positions, gross exposure, no long+short conflict.

Your job: on PASSED entries, you may REJECT if MC/macro/correlation warrant; you must NOT approve failed scripted entries.

Proposal:
- Ticker: {ticker}
- Side: {side.upper()}
- Size: {weight * 100:.1f}%

Portfolio context:
{json.dumps(portfolio_context, indent=2)}

Return ONLY JSON:
{{"approved": true, "rationale": "one sentence"}}"""

    def plan_entry(
        self,
        ticker: str,
        side: str,
        weight: float,
        portfolio_context: Dict[str, Any],
        *,
        scripted_passed: bool = True,
    ) -> Dict[str, Any]:
        if not self.cfg.get("llm_overnight_risk_planner", True):
            return {
                "approved": scripted_passed if self.system == "internal" else True,
                "rationale": "planner disabled",
            }

        if self.system == "internal" and not scripted_passed:
            return {
                "approved": False,
                "rationale": "scripted overnight risk rejected",
            }

        prompt = (
            self._build_baseline_prompt(ticker, side, weight, portfolio_context)
            if self.system == "baseline"
            else self._build_internal_prompt(
                ticker, side, weight, portfolio_context, scripted_passed
            )
        )

        try:
            response = self.client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=prompt,
            )
            parsed = parse_planner_object(response.text or "")
            approved = bool(parsed.get("approved", False))
            rationale = str(parsed.get("rationale") or "").strip() or (
                "approved" if approved else "rejected"
            )
            if self.system == "internal" and not scripted_passed:
                approved = False
            logger.info(
                f"[{self.system}] LLM overnight risk {ticker} {side}: "
                f"{'APPROVE' if approved else 'REJECT'} — {rationale[:120]}"
            )
            return {"approved": approved, "rationale": rationale[:200]}
        except Exception as e:
            logger.warning(f"LLM overnight risk failed for {ticker} ({self.system}): {e}")
            return {
                "approved": scripted_passed if self.system == "internal" else True,
                "rationale": f"planner fallback ({e})",
            }
