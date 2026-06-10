"""
Internal Signal Agent (System B) — Twin Ledger style.
Same structured competition prompt as baseline, enriched with MC predictions and Kelly sizing.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src import config
from src.apis.gemini_client import get_genai_client
from src.agents.ledger_utils import GEMINI_FLASH_MODEL, mc_confidence_score, parse_trading_decisions
from src.agents.base_agent import BaseAgent
from src.agents.competition_context import build_competition_context
from src.models.signal import Signal
from src.models.trading_decision import TradingDecision
from src.strategies.allocator import PositionAllocator
from src.logger import setup_logger

logger = setup_logger(__name__)

class InternalSignalAgent(BaseAgent):
    """
    Twin Ledger internal agent.
    - Same competition framework as baseline
    - Additional MarketCrunch predictions + Kelly sizing guidance
    - Kelly allocator used for BUY execution sizing
    """

    def __init__(self):
        super().__init__(system="internal")
        self.client = get_genai_client()
        self.ticker_universe = config.TICKER_UNIVERSE
        self.confidence_threshold = config.INTERNAL_CONFIG.get("confidence_threshold", 0.55)
        self.max_positions = config.INTERNAL_CONFIG.get("max_positions", 8)
        self.kelly_fraction = config.INTERNAL_CONFIG.get("kelly_fraction", 0.25)

    def _kelly_context(
        self,
        mc_predictions: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Pre-compute Kelly sizing guidance per ticker from MC predictions."""
        kelly_ctx = {}
        for ticker in self.ticker_universe:
            analysis = mc_predictions.get(ticker, {})
            ai_est = analysis.get("ai_estimate", {})
            conf_label = ai_est.get("confidence", "Low")
            target_delta = float(ai_est.get("target_delta_numeric", 0) or 0)
            confidence = mc_confidence_score(conf_label)
            predicted_return = target_delta / 100

            kelly_raw = PositionAllocator.kelly_criterion(
                predicted_return=predicted_return,
                confidence=confidence,
                max_kelly=self.kelly_fraction,
            )
            kelly_ctx[ticker] = {
                "mc_confidence": conf_label,
                "mc_confidence_score": confidence,
                "mc_target_return_pct": target_delta,
                "kelly_suggested_weight": round(kelly_raw, 4),
                "kelly_max_fraction": self.kelly_fraction,
            }
        return kelly_ctx

    def _build_ledger_prompt(
        self,
        competition: Dict,
        technical_data: Dict[str, Dict],
        mc_predictions: Dict[str, Dict],
        kelly_context: Dict[str, Dict],
        databento_sources: Optional[Dict[str, Dict]] = None,
    ) -> str:
        market_lines = []
        for ticker in self.ticker_universe:
            tech = technical_data.get(ticker, {})
            kelly = kelly_context.get(ticker, {})
            mc = mc_predictions.get(ticker, {})
            ai_est = mc.get("ai_estimate", {}) if mc else {}

            if not tech and not mc:
                market_lines.append(f"- {ticker}: no data")
                continue

            db_note = ""
            if databento_sources and ticker in databento_sources:
                feats = databento_sources[ticker].get("databento_features", {})
                if feats:
                    feat_str = ", ".join(f"{k}={v:.4f}" for k, v in feats.items())
                    db_note = f" | DB[{feat_str}]"

            market_lines.append(
                f"- {ticker}: close=${tech.get('close', 0):.2f} | "
                f"RSI={tech.get('rsi_14', 0):.1f} | "
                f"MACD_hist={tech.get('macd_histogram', 0):.4f} | "
                f"BB_z={tech.get('bollinger_zscore', 0):.2f} | "
                f"MC_target={ai_est.get('target_delta_numeric', 'N/A')}% | "
                f"MC_conf={ai_est.get('confidence', 'N/A')} | "
                f"Kelly_weight={kelly.get('kelly_suggested_weight', 0):.4f}{db_note}"
            )

        return f"""You are an autonomous trading agent in the Twin Ledger — a live head-to-head paper trading competition.

Your goal is to maximize final rank and BEAT the competing Baseline Trader.
You have MarketCrunch predictions, Kelly Criterion sizing guidance, technical indicators,
and optional DataBento enrichment — use these as your edge.

You are shown:
1. Your current portfolio, cash, positions, and P&L (Internal / System B).
2. MarketCrunch predictions and Kelly-suggested weights for each ticker.
3. Current market data and technical indicators for the tradable universe.
4. The leaderboard: Baseline Trader's account value, positions, and P&L.

Use this information to decide whether to:
- preserve capital,
- take asymmetric opportunities backed by high-confidence MC signals,
- reduce risk when ahead on the leaderboard,
- increase risk intelligently when behind,
- size positions using Kelly guidance (conservative fraction: {self.kelly_fraction}),
- avoid unnecessary churn and fees,
- avoid liquidation or catastrophic drawdown.

Trading constraints:
- Long-only ETF paper trading on Alpaca
- Universe: {', '.join(self.ticker_universe)}
- Max {self.max_positions} open positions
- Max 10% of portfolio per new BUY (size_pct <= 0.10)
- Valid actions: BUY, SELL, HOLD, CLOSE (no SHORT/COVER)
- For BUY: size_pct = fraction of total portfolio (0.01–0.10); align with Kelly_suggested_weight when MC confidence is High/Medium
- For SELL: size_pct = fraction of existing position to sell (0.01–1.0)
- For CLOSE: exit the full existing position (size_pct ignored)
- For HOLD: no trade
- Minimum confidence for BUY: {self.confidence_threshold}

Competition context:
{json.dumps(competition, indent=2)}

Kelly sizing context (pre-computed from MC predictions):
{json.dumps(kelly_context, indent=2)}

DataBento discovered features (approved by discovery agent):
{json.dumps(databento_sources or {}, indent=2)}

Market data, MC predictions & indicators:
{chr(10).join(market_lines)}

Return ONLY a JSON array of decisions. Include entries where action is not HOLD.
Each object must have:
- action: BUY | SELL | HOLD | CLOSE
- ticker: symbol from universe
- size_pct: number (for BUY, consider Kelly_suggested_weight; capped at 0.10)
- confidence: 0.0–1.0
- rationale: why this trade helps you beat Baseline Trader
- invalidation: what would make you reverse this decision
- competitive_note: how this relates to your rank and competitor behavior

Example:
[
  {{
    "action": "BUY",
    "ticker": "QQQ",
    "size_pct": 0.08,
    "confidence": 0.78,
    "rationale": "High MC confidence with positive target; Kelly supports 8% allocation.",
    "invalidation": "MC confidence drops below Medium or target turns negative.",
    "competitive_note": "Behind on leaderboard; deploy prediction edge vs technicals-only rival."
  }}
]"""

    async def make_trading_decisions(
        self,
        technical_data: Dict[str, Dict],
        mc_predictions: Dict[str, Dict],
        competition: Optional[Dict] = None,
        databento_sources: Optional[Dict[str, Dict]] = None,
    ) -> List[TradingDecision]:
        competition = competition or build_competition_context("internal")

        try:
            if config.USE_ADK:
                from src.adk.runner import run_signal_agent
                kelly_context = self._kelly_context(mc_predictions)
                decisions = await run_signal_agent(
                    system="internal",
                    user_payload={
                        "competition": competition,
                        "technical_data": technical_data,
                        "mc_predictions": mc_predictions,
                        "kelly_context": kelly_context,
                        "databento_sources": databento_sources or {},
                    },
                    session_id="internal_signal",
                    valid_tickers=self.ticker_universe,
                )
            else:
                kelly_context = self._kelly_context(mc_predictions)
                prompt = self._build_ledger_prompt(
                    competition, technical_data, mc_predictions, kelly_context, databento_sources
                )
                response = self.client.models.generate_content(
                    model=GEMINI_FLASH_MODEL,
                    contents=prompt,
                )
                decisions = parse_trading_decisions(response.text, self.ticker_universe)
            actionable = [d for d in decisions if d.action != "HOLD"]
            self.log_action(
                f"Ledger decisions: {len(actionable)} actionable / {len(decisions)} total | "
                f"Rank {competition['leaderboard']['your_rank']}/2 "
                f"({competition['leaderboard']['status']} by "
                f"${competition['leaderboard']['value_gap_usd']:,.2f})",
                data={
                    "leaderboard": competition.get("leaderboard"),
                    "decisions": [d.to_dict() for d in actionable],
                },
            )
            for d in actionable:
                self.log_action(
                    f"  {d.action} {d.ticker} size={d.size_pct:.1%} "
                    f"conf={d.confidence:.2f} — {d.rationale[:80]}",
                    data={**d.to_dict(), "leaderboard": competition.get("leaderboard")},
                    event_type="ledger_decision",
                )
            return decisions

        except Exception as e:
            self.log_error(f"Ledger decision failed: {e}")
            return []

    def decisions_to_signals(
        self,
        decisions: List[TradingDecision],
        technical_data: Dict[str, Dict],
        mc_predictions: Dict[str, Dict],
    ) -> Dict[str, Signal]:
        """Convert ledger decisions into Signals enriched with MC data (for Kelly sizing)."""
        signals: Dict[str, Signal] = {}

        for decision in decisions:
            if decision.action not in ("BUY", "SELL", "CLOSE"):
                continue
            if decision.action == "BUY" and decision.confidence < self.confidence_threshold:
                self.log_action(
                    f"Skipping BUY {decision.ticker}: "
                    f"confidence {decision.confidence:.2f} < {self.confidence_threshold:.2f}"
                )
                continue

            mc = mc_predictions.get(decision.ticker, {})
            ai_est = mc.get("ai_estimate", {})
            conf_label = ai_est.get("confidence", "Low")
            target_delta = float(ai_est.get("target_delta_numeric", 0) or 0)
            mc_confidence = mc_confidence_score(conf_label)

            tech = technical_data.get(decision.ticker, {})
            signals[decision.ticker] = Signal(
                ticker=decision.ticker,
                timestamp=datetime.now(),
                predicted_return=target_delta / 100,
                confidence=max(decision.confidence, mc_confidence),
                bollinger_zscore=tech.get("bollinger_zscore"),
                macd_histogram=tech.get("macd_histogram"),
                rsi_14=tech.get("rsi_14"),
                llm_reasoning=decision.rationale,
                system="internal",
            )

        return signals

    async def run_ledger_cycle(
        self,
        technical_data: Dict[str, Dict],
        mc_predictions: Dict[str, Dict],
        competition: Optional[Dict] = None,
        databento_sources: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[List[TradingDecision], Dict[str, Signal]]:
        decisions = await self.make_trading_decisions(
            technical_data, mc_predictions, competition, databento_sources
        )
        signals = self.decisions_to_signals(decisions, technical_data, mc_predictions)
        return decisions, signals

    async def execute(self) -> bool:
        self.log_action("Internal Twin Ledger agent ready")
        return True
