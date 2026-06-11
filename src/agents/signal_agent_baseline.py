"""
Baseline Signal Agent (System A) — Twin Ledger style.
Makes structured trading decisions to beat the Internal Trader.
Uses technical indicators + portfolio/competition context only (no MC predictions).
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src import config
from src.apis.gemini_client import get_genai_client
from src.apis.grounding import google_search_grounding_config
from src.agents.ledger_utils import (
    GEMINI_FLASH_MODEL,
    SignalLedgerResult,
    emit_signal_ledger_audit,
    parse_signal_ledger_response,
)
from src.agents.signal_context import fetch_signal_news, format_news_block
from src.agents.base_agent import BaseAgent
from src.agents.competition_context import build_competition_context
from src.learning.context import build_signal_learning_block
from src.models.signal import Signal
from src.models.trading_decision import TradingDecision
from src.logger import setup_logger

logger = setup_logger(__name__)


class BaselineSignalAgent(BaseAgent):
    """
    Twin Ledger baseline agent.
    - Sees own portfolio + competitor (internal) leaderboard snapshot
    - Uses Alpaca-derived technical indicators only
    - Returns structured BUY/SELL/HOLD/CLOSE decisions
    """

    def __init__(self):
        super().__init__(system="baseline")
        self.client = get_genai_client()
        self.ticker_universe = config.TICKER_UNIVERSE
        self.confidence_threshold = config.BASELINE_CONFIG.get("confidence_threshold", 0.5)
        self.max_positions = config.BASELINE_CONFIG.get("max_positions", 8)

    def _build_ledger_prompt(
        self,
        competition: Dict,
        technical_data: Dict[str, Dict],
        learning_block: str = "",
        news_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        market_lines = []
        for ticker in self.ticker_universe:
            tech = technical_data.get(ticker, {})
            if not tech:
                market_lines.append(f"- {ticker}: no data")
                continue
            market_lines.append(
                f"- {ticker}: close=${tech.get('close', 0):.2f} | "
                f"RSI={tech.get('rsi_14', 0):.1f} | "
                f"MACD_hist={tech.get('macd_histogram', 0):.4f} | "
                f"BB_z={tech.get('bollinger_zscore', 0):.2f}"
            )

        return f"""You are an autonomous trading agent in the Twin Ledger — a live head-to-head paper trading competition.

Your goal is to maximize final rank and BEAT the competing Internal Trader.
You do NOT have access to MarketCrunch predictions or proprietary signals.
You must win using technical analysis, portfolio discipline, and relative-performance strategy.

You are shown:
1. Your current portfolio, cash, positions, and P&L (Baseline / System A).
2. Current market data, technical indicators, and recent news for the tradable universe.
3. Google Search grounding for macro/sector drivers when it would improve ETF decisions.
4. The leaderboard: Internal Trader's account value, positions, and P&L.

Use this information to decide whether to:
- preserve capital,
- take asymmetric opportunities,
- reduce risk when ahead,
- increase risk intelligently when behind,
- avoid unnecessary churn and fees,
- avoid liquidation or catastrophic drawdown.

Trading constraints:
- Long-only ETF paper trading on Alpaca
- Universe: {', '.join(self.ticker_universe)}
- Max {self.max_positions} open positions
- Max 10% of portfolio per new BUY (size_pct <= 0.10)
- Valid actions: BUY, SELL, HOLD, CLOSE (no SHORT/COVER)
- For BUY: size_pct = fraction of total portfolio to allocate (0.01–0.10)
- For SELL: size_pct = fraction of existing position to sell (0.01–1.0)
- For CLOSE: exit the full existing position (size_pct ignored)
- For HOLD: no trade
{f'''
{learning_block}

''' if learning_block else ''}
Competition context:
{json.dumps(competition, indent=2)}

Market data & indicators:
{chr(10).join(market_lines)}

Recent news (Alpaca / fallback):
{format_news_block(news_data or {})}

Return ONLY a JSON object with this shape:
{{
  "decisions": [ ...trade objects... ],
  "no_action_rationale": "2-4 sentences — REQUIRED when decisions is empty"
}}

Put trade objects in "decisions" only when action is not HOLD. Each trade object must have:
- action: BUY | SELL | CLOSE
- ticker: symbol from universe
- size_pct: number
- confidence: 0.0–1.0
- rationale: why this trade helps you beat Internal Trader
- invalidation: what would make you reverse this decision
- competitive_note: how this relates to your rank and competitor behavior

When you recommend no trades (decisions = []), you MUST fill no_action_rationale with a clear explanation:
leaderboard posture, technical/macro read, risk discipline, learning lessons applied, and what would change your mind.

Example (trades):
{{
  "decisions": [
    {{
      "action": "BUY",
      "ticker": "QQQ",
      "size_pct": 0.08,
      "confidence": 0.72,
      "rationale": "Momentum improving while competitor is likely overweight low-confidence names.",
      "invalidation": "RSI > 70 and MACD histogram turns negative.",
      "competitive_note": "Behind on leaderboard; need controlled risk-on exposure."
    }}
  ],
  "no_action_rationale": ""
}}

Example (no trades):
{{
  "decisions": [],
  "no_action_rationale": "Ahead on the leaderboard with flat exposure; macro/news skew risk-off and no ticker clears confidence threshold. Staying in cash preserves rank until a high-conviction asymmetric setup appears."
}}"""

    async def make_trading_decisions(
        self,
        technical_data: Dict[str, Dict],
        competition: Optional[Dict] = None,
        prefer_direct: bool = False,
        news_data: Optional[Dict[str, Any]] = None,
    ) -> SignalLedgerResult:
        """Twin Ledger decision step: ADK LlmAgent or direct Gemini call."""
        competition = competition or build_competition_context("baseline")
        learning_block = build_signal_learning_block("baseline") if config.LEARNING_ENABLED else ""

        try:
            if config.USE_ADK and not prefer_direct:
                from src.adk.runner import run_signal_agent
                ledger = await run_signal_agent(
                    system="baseline",
                    user_payload={
                        "competition": competition,
                        "technical_data": technical_data,
                        "signal_learning": learning_block,
                    },
                    session_id="baseline_signal",
                    valid_tickers=self.ticker_universe,
                )
                return ledger
            else:
                if news_data is None:
                    news_data = fetch_signal_news(self.ticker_universe).get("news") or {}
                grounding_on = config.SIGNAL_GOOGLE_SEARCH_GROUNDING
                article_count = sum(
                    len(v) for v in news_data.values() if isinstance(v, list)
                )
                self.log_action(
                    f"Signal context: {article_count} news articles"
                    + (" | Google Search grounding ON" if grounding_on else ""),
                    data={
                        "news_article_count": article_count,
                        "google_search_grounding": grounding_on,
                    },
                )
                prompt = self._build_ledger_prompt(
                    competition,
                    technical_data,
                    learning_block=learning_block,
                    news_data=news_data,
                )
                gen_config = (
                    google_search_grounding_config()
                    if grounding_on
                    else None
                )
                response = self.client.models.generate_content(
                    model=GEMINI_FLASH_MODEL,
                    contents=prompt,
                    config=gen_config,
                )
                ledger = parse_signal_ledger_response(
                    response.text, self.ticker_universe
                )

            emit_signal_ledger_audit(self, ledger, competition)
            return ledger

        except Exception as e:
            self.log_error(f"Ledger decision failed: {e}")
            return SignalLedgerResult(decisions=[], no_action_rationale="")

    def decisions_to_signals(
        self,
        decisions: List[TradingDecision],
        technical_data: Dict[str, Dict],
    ) -> Dict[str, Signal]:
        """Convert ledger decisions into Signal objects for downstream allocation."""
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

            tech = technical_data.get(decision.ticker, {})
            predicted_return = decision.size_pct if decision.action == "BUY" else -decision.size_pct

            signals[decision.ticker] = Signal(
                ticker=decision.ticker,
                timestamp=datetime.now(),
                predicted_return=predicted_return,
                confidence=decision.confidence,
                bollinger_zscore=tech.get("bollinger_zscore"),
                macd_histogram=tech.get("macd_histogram"),
                rsi_14=tech.get("rsi_14"),
                llm_reasoning=decision.rationale,
                system="baseline",
            )

        return signals

    async def run_ledger_cycle(
        self,
        technical_data: Dict[str, Dict],
        competition: Optional[Dict] = None,
    ) -> Tuple[List[TradingDecision], Dict[str, Signal]]:
        """Full Twin Ledger step: decisions + signals."""
        ledger = await self.make_trading_decisions(technical_data, competition)
        signals = self.decisions_to_signals(ledger.decisions, technical_data)
        return ledger.decisions, signals

    async def execute(self) -> bool:
        self.log_action("Baseline Twin Ledger agent ready")
        return True
