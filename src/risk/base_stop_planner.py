"""LLM base stop-loss planner — baseline pure LLM, internal hybrid over scripted floor."""
from __future__ import annotations

from typing import Any, Dict, Optional

from src import config
from src.apis.gemini_client import get_genai_client
from src.agents.ledger_utils import GEMINI_FLASH_MODEL
from src.learning.context import build_risk_learning_block
from src.logger import setup_logger
from src.risk.planner_parse import parse_planner_object
from src.risk.trailing_planner import (
    RETURN_PNL_HELP,
    format_position_side_line,
    position_side_label,
)

logger = setup_logger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _should_refresh(cached: Dict, current_return: float) -> bool:
    if not str(cached.get("rationale") or "").strip():
        return True
    last_return = cached.get("planned_at_return", 0.0)
    return abs(current_return - last_return) >= 0.003


def merge_hybrid_base_stop(
    scripted_threshold: float,
    llm_plan: Dict,
) -> tuple[float, str]:
    """Tighter (closer-to-zero) negative return threshold wins."""
    llm_thr = float(llm_plan["stop_loss_threshold"])
    effective = max(float(scripted_threshold), llm_thr)
    policy = (
        f"hybrid:scripted({scripted_threshold * 100:.1f}%)"
        f"+llm({llm_thr * 100:.1f}%)→{effective * 100:.1f}%"
    )
    return effective, policy


class BaseStopPlanner:
    """Gemini planner for intraday base stop-loss return thresholds."""

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

    def _bounds(self) -> Dict:
        return self.cfg.get("llm_base_stop_bounds", {})

    def _heuristic_baseline(
        self,
        ticker: str,
        current_return: float,
        atr_pct: Optional[float],
    ) -> Dict:
        threshold = -0.012
        if atr_pct and atr_pct > 0.02:
            threshold = -0.018
        elif current_return < -0.005:
            threshold = -0.008
        return {
            "stop_loss_threshold": threshold,
            "rationale": f"baseline base-stop heuristic for {ticker}",
            "planned_at_return": current_return,
        }

    def _heuristic_internal(
        self,
        ticker: str,
        current_return: float,
    ) -> Dict:
        return {
            "stop_loss_threshold": self.cfg["base_stop_loss_threshold"],
            "rationale": f"internal scripted base-stop fallback for {ticker}",
            "planned_at_return": current_return,
        }

    def _build_baseline_prompt(
        self,
        ticker: str,
        entry: float,
        current_price: float,
        current_return: float,
        qty: float,
        atr_pct: Optional[float],
        momentum_5d: Optional[float],
    ) -> str:
        bounds = self._bounds()
        risk_learning = ""
        if config.LEARNING_ENABLED:
            risk_learning = build_risk_learning_block(self.system)
        learning_section = f"\n\n{risk_learning}\n" if risk_learning else ""
        return f"""You are a risk manager for a baseline ETF trading agent (technicals only).

Set the BASE stop-loss return threshold for an open position. Baseline uses PURE LLM base stops — no fixed -1% rule.{learning_section}
Position:
- Ticker: {ticker}
{format_position_side_line(qty)}
- Entry: ${entry:.2f} | Current: ${current_price:.2f}
- Return: {current_return * 100:.2f}% — {RETURN_PNL_HELP}
- ATR/price: {f"{atr_pct * 100:.2f}%" if atr_pct else "unknown"}
- 5d momentum: {f"{momentum_5d * 100:.2f}%" if momentum_5d is not None else "unknown"}

stop_loss_threshold must be NEGATIVE (loss %). Exit when return falls to or below this value.
Bounds: {bounds.get('threshold_min', -0.05)} to {bounds.get('threshold_max', -0.003)}

Return ONLY JSON:
{{"stop_loss_threshold": -0.012, "rationale": "one sentence"}}"""

    def _build_internal_prompt(
        self,
        ticker: str,
        entry: float,
        current_price: float,
        current_return: float,
        qty: float,
        atr_pct: Optional[float],
        momentum_5d: Optional[float],
        scripted_threshold: float,
        mc_context: Optional[Dict[str, Any]],
    ) -> str:
        bounds = self._bounds()
        mc_summary = "unavailable"
        if mc_context:
            est = mc_context.get("ai_estimate", {})
            mc_summary = (
                f"target={est.get('target_delta_numeric', 0):.2f}%, "
                f"confidence={est.get('confidence', 'Unknown')}"
            )
        risk_learning = ""
        if config.LEARNING_ENABLED:
            risk_learning = build_risk_learning_block(self.system)
        learning_section = f"\n\n{risk_learning}\n" if risk_learning else ""
        return f"""You are a risk manager for the internal ETF agent (MarketCrunch + scripted stops).{learning_section}
Scripted base-stop floor (never looser than this — threshold closer to zero):
- stop_loss_threshold >= {scripted_threshold} (e.g. {scripted_threshold * 100:.1f}%)

You may suggest TIGHTER stops (more negative, e.g. -1.2%) when MC/momentum warrant earlier exit.
Bounds: {bounds.get('threshold_min', -0.05)} to {bounds.get('threshold_max', -0.003)}

Position:
- Ticker: {ticker}
{format_position_side_line(qty)}
- Entry: ${entry:.2f} | Current: ${current_price:.2f}
- Return: {current_return * 100:.2f}% — {RETURN_PNL_HELP}
- ATR/price: {f"{atr_pct * 100:.2f}%" if atr_pct else "unknown"}
- 5d momentum: {f"{momentum_5d * 100:.2f}%" if momentum_5d is not None else "unknown"}
- MarketCrunch: {mc_summary}

Return ONLY JSON:
{{"stop_loss_threshold": -0.009, "rationale": "one sentence"}}"""

    def plan(
        self,
        ticker: str,
        entry: float,
        current_price: float,
        current_return: float,
        qty: float = 0,
        atr_pct: Optional[float] = None,
        momentum_5d: Optional[float] = None,
        mc_context: Optional[Dict[str, Any]] = None,
        scripted_threshold: Optional[float] = None,
        cached: Optional[Dict] = None,
    ) -> Dict:
        if cached and not _should_refresh(cached, current_return):
            return cached

        if not self.cfg.get("llm_base_stop_planner", True):
            if self.system == "baseline":
                return self._heuristic_baseline(ticker, current_return, atr_pct)
            return self._heuristic_internal(ticker, current_return)

        bounds = self._bounds()
        thr_min = bounds.get("threshold_min", -0.05)
        thr_max = bounds.get("threshold_max", -0.003)
        scripted = (
            float(scripted_threshold)
            if scripted_threshold is not None
            else float(self.cfg.get("base_stop_loss_threshold", -0.01))
        )

        prompt = (
            self._build_baseline_prompt(
                ticker, entry, current_price, current_return, qty, atr_pct, momentum_5d
            )
            if self.system == "baseline"
            else self._build_internal_prompt(
                ticker,
                entry,
                current_price,
                current_return,
                qty,
                atr_pct,
                momentum_5d,
                scripted,
                mc_context,
            )
        )

        try:
            response = self.client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=prompt,
            )
            parsed = parse_planner_object(response.text or "")
            threshold = _clamp(
                float(parsed.get("stop_loss_threshold", scripted)),
                thr_min,
                thr_max,
            )
            if threshold > 0:
                threshold = -abs(threshold)
            rationale = str(parsed.get("rationale") or "").strip()
            if not rationale:
                rationale = (
                    f"LLM base stop {ticker} ({position_side_label(qty)}): "
                    f"exit at {threshold * 100:.1f}% return"
                )
            result = {
                "stop_loss_threshold": threshold,
                "rationale": rationale[:200],
                "planned_at_return": current_return,
            }
            logger.info(
                f"[{self.system}] LLM base stop {ticker}: "
                f"threshold={threshold * 100:.2f}% — {result['rationale']}"
            )
            return result
        except Exception as e:
            logger.warning(f"LLM base stop planner failed for {ticker} ({self.system}): {e}")
            if self.system == "baseline":
                return self._heuristic_baseline(ticker, current_return, atr_pct)
            return self._heuristic_internal(ticker, current_return)
