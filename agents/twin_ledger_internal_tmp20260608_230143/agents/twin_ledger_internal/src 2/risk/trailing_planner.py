"""
LLM trailing-lock planner.

- Baseline: pure LLM trailing (no scripted 1%/70%).
- Internal: LLM refines scripted floor (1% activation, 70% lock) — merged in risk_monitor.
"""
from typing import Any, Dict, Optional

from src import config
from src.apis.gemini_client import get_genai_client
from src.agents.ledger_utils import GEMINI_FLASH_MODEL, parse_ledger_response
from src.logger import setup_logger

logger = setup_logger(__name__)



def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _should_refresh(cached: Dict, current_return: float) -> bool:
    last_return = cached.get("planned_at_return", 0.0)
    return current_return >= last_return + 0.005


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
        atr_pct: Optional[float],
        momentum_5d: Optional[float],
    ) -> str:
        bounds = self._bounds()
        return f"""You are a risk manager for a baseline ETF trading agent (technicals only).

Set trailing profit-lock policy for an open position. Baseline uses PURE LLM trailing — no fixed rules.

Position:
- Ticker: {ticker}
- Entry: ${entry:.2f} | Current: ${current_price:.2f}
- Return: {current_return*100:.2f}%
- ATR/price: {f"{atr_pct*100:.2f}%" if atr_pct else "unknown"}
- 5d momentum: {f"{momentum_5d*100:.2f}%" if momentum_5d is not None else "unknown"}

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

        return f"""You are a risk manager for the internal ETF agent (MarketCrunch predictions + scripted stops).

Scripted floor (never go looser than this):
- activation_threshold >= {scripted_act} (1%)
- profit_lock_fraction >= {scripted_lock} (70% lock)

Your job: suggest TIGHTER trailing parameters when MC confidence and momentum support holding more profit.
You may lower activation slightly (down to {bounds.get('activation_min', 0.005)}) to activate earlier,
or raise profit_lock_fraction (up to {bounds.get('lock_max', 0.85)}) to lock more gains.

Position:
- Ticker: {ticker}
- Entry: ${entry:.2f} | Current: ${current_price:.2f}
- Return: {current_return*100:.2f}%
- ATR/price: {f"{atr_pct*100:.2f}%" if atr_pct else "unknown"}
- 5d momentum: {f"{momentum_5d*100:.2f}%" if momentum_5d is not None else "unknown"}
- MarketCrunch: {mc_summary}

Return ONLY JSON:
{{"activation_threshold": 0.008, "profit_lock_fraction": 0.75, "rationale": "one sentence"}}"""

    def plan(
        self,
        ticker: str,
        entry: float,
        current_price: float,
        current_return: float,
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
                ticker, entry, current_price, current_return, atr_pct, momentum_5d
            )
            if self.system == "baseline"
            else self._build_internal_prompt(
                ticker, entry, current_price, current_return,
                atr_pct, momentum_5d, mc_context,
            )
        )

        try:
            response = self.client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=prompt,
            )
            parsed = parse_ledger_response(response.text)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}

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
            result = {
                "activation_threshold": activation,
                "profit_lock_fraction": lock,
                "rationale": str(parsed.get("rationale", ""))[:200],
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
