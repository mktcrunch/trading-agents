"""
LLM trailing-lock planner.

- Baseline: pure LLM trailing (no scripted 1%/70%).
- Internal: LLM refines scripted floor (1% activation, 70% lock) — merged in risk_monitor.
"""
from typing import Any, Dict, Optional

from src import config
from src.apis.gemini_client import get_genai_client
from src.agents.ledger_utils import GEMINI_FLASH_MODEL, _clean_json_text
from src.learning.context import build_risk_learning_block
from src.logger import setup_logger

logger = setup_logger(__name__)



def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def position_side_label(qty: float) -> str:
    if qty > 0:
        return "LONG"
    if qty < 0:
        return "SHORT"
    return "FLAT"


def format_position_side_line(qty: float) -> str:
    side = position_side_label(qty)
    if side == "FLAT":
        return "- Side: FLAT"
    return f"- Side: {side} (qty {qty:g})"


RETURN_PNL_HELP = (
    "Return is position P&L % for this side (positive = winning, negative = losing)."
)


def _should_refresh(cached: Dict, current_return: float) -> bool:
    last_return = cached.get("planned_at_return", 0.0)
    if not str(cached.get("rationale") or "").strip():
        return True
    return current_return >= last_return + 0.005


def parse_trailing_plan_response(text: str) -> Dict[str, Any]:
    """
    Parse Gemini trailing-plan JSON.

    Accepts a single object (prompt contract), a one-element array, or
    ``{"decisions": [{...}]}`` — but NOT ``parse_ledger_response``, which drops
    bare objects and silently defaults activation/lock/rationale.
    """
    import json

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


class TrailingPlanner:
    """Gemini planner for trailing profit locks."""

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
        return self.cfg.get("llm_trailing_bounds", {})

    def _heuristic_baseline(
        self,
        ticker: str,
        current_return: float,
        atr_pct: Optional[float],
    ) -> Dict:
        activation = 0.008
        lock = 0.55
        if atr_pct and atr_pct > 0.02:
            activation = 0.015
            lock = 0.50
        elif current_return > 0.03:
            activation = 0.01
            lock = 0.65
        return {
            "activation_threshold": activation,
            "profit_lock_fraction": lock,
            "rationale": f"baseline heuristic for {ticker}",
            "planned_at_return": current_return,
        }

    def _heuristic_internal(
        self,
        ticker: str,
        current_return: float,
    ) -> Dict:
        """Internal LLM unavailable — return scripted defaults for hybrid merge."""
        return {
            "activation_threshold": self.cfg["trailing_activation_threshold"],
            "profit_lock_fraction": self.cfg["profit_lock_fraction"],
            "rationale": f"internal scripted fallback for {ticker}",
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

Set trailing profit-lock policy for an open position. Baseline uses PURE LLM trailing — no fixed rules.{learning_section}
Position:
- Ticker: {ticker}
{format_position_side_line(qty)}
- Entry: ${entry:.2f} | Current: ${current_price:.2f}
- Return: {current_return*100:.2f}% — {RETURN_PNL_HELP}
- ATR/price: {f"{atr_pct*100:.2f}%" if atr_pct else "unknown"}
- 5d momentum: {f"{momentum_5d*100:.2f}%" if momentum_5d is not None else "unknown"}

Note: You plan TRAILING profit-lock only. Base stop-loss is handled separately (not LLM).

Return:
1. activation_threshold ({bounds.get('activation_min', 0.005)}–{bounds.get('activation_max', 0.04)})
2. profit_lock_fraction ({bounds.get('lock_min', 0.45)}–{bounds.get('lock_max', 0.80)})

Return ONLY JSON:
{{"activation_threshold": 0.012, "profit_lock_fraction": 0.58, "rationale": "one sentence"}}"""

    def _build_internal_prompt(
        self,
        ticker: str,
        entry: float,
        current_price: float,
        current_return: float,
        qty: float,
        atr_pct: Optional[float],
        momentum_5d: Optional[float],
        mc_context: Optional[Dict[str, Any]],
    ) -> str:
        bounds = self._bounds()
        scripted_act = self.cfg["trailing_activation_threshold"]
        scripted_lock = self.cfg["profit_lock_fraction"]
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
        return f"""You are a risk manager for the internal ETF agent (MarketCrunch predictions + scripted stops).{learning_section}
Scripted floor (never go looser than this):
- activation_threshold >= {scripted_act} (1%)
- profit_lock_fraction >= {scripted_lock} (70% lock)

Your job: suggest TIGHTER trailing parameters when MC confidence and momentum support holding more profit.
You may lower activation slightly (down to {bounds.get('activation_min', 0.005)}) to activate earlier,
or raise profit_lock_fraction (up to {bounds.get('lock_max', 0.85)}) to lock more gains.

Position:
- Ticker: {ticker}
{format_position_side_line(qty)}
- Entry: ${entry:.2f} | Current: ${current_price:.2f}
- Return: {current_return*100:.2f}% — {RETURN_PNL_HELP}
- ATR/price: {f"{atr_pct*100:.2f}%" if atr_pct else "unknown"}
- 5d momentum: {f"{momentum_5d*100:.2f}%" if momentum_5d is not None else "unknown"}
- MarketCrunch: {mc_summary}

Note: You plan TRAILING profit-lock only. Base stop-loss uses scripted ATR/fixed rules (not LLM).

Return ONLY JSON:
{{"activation_threshold": 0.008, "profit_lock_fraction": 0.75, "rationale": "one sentence"}}"""

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
        cached: Optional[Dict] = None,
    ) -> Dict:
        if cached and not _should_refresh(cached, current_return):
            return cached

        if not self.cfg.get("llm_trailing_planner", True):
            if self.system == "baseline":
                return self._heuristic_baseline(ticker, current_return, atr_pct)
            return self._heuristic_internal(ticker, current_return)

        bounds = self._bounds()
        prompt = (
            self._build_baseline_prompt(
                ticker, entry, current_price, current_return, qty, atr_pct, momentum_5d
            )
            if self.system == "baseline"
            else self._build_internal_prompt(
                ticker, entry, current_price, current_return, qty,
                atr_pct, momentum_5d, mc_context,
            )
        )

        try:
            response = self.client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=prompt,
            )
            parsed = parse_trailing_plan_response(response.text or "")

            activation = _clamp(
                float(parsed.get("activation_threshold", 0.01)),
                bounds.get("activation_min", 0.005),
                bounds.get("activation_max", 0.04),
            )
            lock = _clamp(
                float(parsed.get("profit_lock_fraction", 0.70)),
                bounds.get("lock_min", 0.45),
                bounds.get("lock_max", 0.85),
            )
            rationale = str(parsed.get("rationale") or "").strip()
            if not rationale:
                rationale = (
                    f"LLM trailing {ticker}: activate {activation*100:.1f}%, "
                    f"lock {lock*100:.0f}% of profit"
                )
            result = {
                "activation_threshold": activation,
                "profit_lock_fraction": lock,
                "rationale": rationale[:200],
                "planned_at_return": current_return,
            }
            logger.info(
                f"[{self.system}] LLM trailing {ticker}: "
                f"activate={activation*100:.2f}% lock={lock*100:.0f}% — {result['rationale']}"
            )
            return result
        except Exception as e:
            logger.warning(f"LLM trailing planner failed for {ticker} ({self.system}): {e}")
            if self.system == "baseline":
                return self._heuristic_baseline(ticker, current_return, atr_pct)
            return self._heuristic_internal(ticker, current_return)


def merge_hybrid_trailing(
    scripted_activation: float,
    scripted_lock: float,
    llm_plan: Dict,
) -> tuple[float, float, str]:
    """
    Internal hybrid: scripted floor + LLM refinement.
    - Activate at earlier of the two thresholds
    - Lock at least scripted % of profit, LLM can tighten further
    """
    llm_act = llm_plan["activation_threshold"]
    llm_lock = llm_plan["profit_lock_fraction"]
    activation = min(scripted_activation, llm_act)
    lock_frac = max(scripted_lock, llm_lock)
    policy = (
        f"hybrid:scripted({scripted_activation*100:.1f}%/{scripted_lock*100:.0f}%)"
        f"+llm({llm_act*100:.1f}%/{llm_lock*100:.0f}%)"
        f"→{activation*100:.1f}%/{lock_frac*100:.0f}%"
    )
    return activation, lock_frac, policy
