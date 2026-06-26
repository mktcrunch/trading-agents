"""
Internal Signal Agent (System B) — Twin Ledger style.
Same structured competition prompt as baseline, enriched with MC predictions and Kelly sizing.
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src import config
from src.apis.gemini_client import get_genai_client
from src.apis.grounding import google_search_grounding_config
from src.agents.ledger_utils import (
    GEMINI_FLASH_MODEL,
    SIGNAL_JSON_PARSE_ATTEMPTS,
    SignalLedgerResult,
    emit_signal_ledger_audit,
    mc_confidence_score,
    parse_signal_ledger_response,
    record_signal_gemini_query,
)
from src.agents.signal_context import fetch_signal_news
from src.agents.base_agent import BaseAgent
from src.agents.competition_context import build_competition_context
from src.adk.prompts.internal import INTERNAL_SIGNAL_INSTRUCTION
from src.adk.prompts.signal_context import build_runtime_signal_prompt
from src.learning.context import build_signal_learning_block
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
        self.confidence_threshold = config.INTERNAL_CONFIG.get("confidence_threshold", 0.5)
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

            kelly_raw_long = PositionAllocator.kelly_criterion(
                predicted_return=predicted_return,
                confidence=confidence,
                max_kelly=self.kelly_fraction,
                side="long",
            )
            kelly_raw_short = PositionAllocator.kelly_criterion(
                predicted_return=predicted_return,
                confidence=confidence,
                max_kelly=self.kelly_fraction,
                side="short",
            )
            kelly_ctx[ticker] = {
                "mc_confidence": conf_label,
                "mc_confidence_score": confidence,
                "mc_target_return_pct": target_delta,
                "kelly_suggested_weight_long": round(kelly_raw_long, 4),
                "kelly_suggested_weight_short": round(kelly_raw_short, 4),
                "kelly_suggested_weight": round(kelly_raw_long, 4),
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
        learning_block: str = "",
        news_data: Optional[Dict[str, Any]] = None,
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
                f"Kelly_long={kelly.get('kelly_suggested_weight_long', 0):.4f} | "
                f"Kelly_short={kelly.get('kelly_suggested_weight_short', 0):.4f}{db_note}"
            )

        extra = f"""Kelly sizing context (pre-computed from MC predictions):
{json.dumps(kelly_context, indent=2)}

DataBento discovered features (approved by discovery agent):
{json.dumps(databento_sources or {}, indent=2)}"""

        return build_runtime_signal_prompt(
            INTERNAL_SIGNAL_INSTRUCTION,
            competition=competition,
            market_lines=market_lines,
            news_data=news_data,
            learning_block=learning_block,
            extra_sections=extra,
        )

    async def make_trading_decisions(
        self,
        technical_data: Dict[str, Dict],
        mc_predictions: Dict[str, Dict],
        competition: Optional[Dict] = None,
        databento_sources: Optional[Dict[str, Dict]] = None,
        prefer_direct: bool = False,
        news_data: Optional[Dict[str, Any]] = None,
    ) -> SignalLedgerResult:
        competition = competition or build_competition_context("internal")
        learning_block = build_signal_learning_block("internal") if config.LEARNING_ENABLED else ""

        try:
            if config.USE_ADK and not prefer_direct:
                from src.adk.runner import run_signal_agent
                kelly_context = self._kelly_context(mc_predictions)
                ledger = await run_signal_agent(
                    system="internal",
                    user_payload={
                        "competition": competition,
                        "technical_data": technical_data,
                        "mc_predictions": mc_predictions,
                        "kelly_context": kelly_context,
                        "databento_sources": databento_sources or {},
                        "signal_learning": learning_block,
                    },
                    session_id="internal_signal",
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
                kelly_context = self._kelly_context(mc_predictions)
                prompt = self._build_ledger_prompt(
                    competition,
                    technical_data,
                    mc_predictions,
                    kelly_context,
                    databento_sources,
                    learning_block=learning_block,
                    news_data=news_data,
                )
                record_signal_gemini_query(
                    system="internal",
                    path="direct_prompt",
                    query_text=prompt,
                    payload={
                        "valid_tickers": list(self.ticker_universe),
                        "competition": competition,
                        "technical_data": technical_data,
                        "mc_predictions": mc_predictions,
                        "kelly_context": kelly_context,
                        "databento_sources": databento_sources or {},
                        "news_data": news_data or {},
                        "signal_learning": learning_block,
                    },
                    agent=self.__class__.__name__,
                )
                gen_config = (
                    google_search_grounding_config()
                    if grounding_on
                    else None
                )
                ledger = None
                for attempt in range(1, SIGNAL_JSON_PARSE_ATTEMPTS + 1):
                    try:
                        response = self.client.models.generate_content(
                            model=GEMINI_FLASH_MODEL,
                            contents=prompt,
                            config=gen_config,
                        )
                        ledger = parse_signal_ledger_response(
                            response.text, self.ticker_universe
                        )
                        break
                    except json.JSONDecodeError as exc:
                        if attempt < SIGNAL_JSON_PARSE_ATTEMPTS:
                            self.log_action(
                                f"Ledger JSON parse failed (attempt {attempt}/"
                                f"{SIGNAL_JSON_PARSE_ATTEMPTS}), retrying Gemini",
                                data={"error": str(exc)},
                            )
                            continue
                        raise
                if ledger is None:
                    raise RuntimeError("Signal ledger generation produced no result")

            emit_signal_ledger_audit(self, ledger, competition)
            return ledger

        except Exception as e:
            self.log_error(f"Ledger decision failed: {e}")
            return SignalLedgerResult(decisions=[], no_action_rationale="")

    def decisions_to_signals(
        self,
        decisions: List[TradingDecision],
        technical_data: Dict[str, Dict],
        mc_predictions: Dict[str, Dict],
    ) -> Dict[str, Signal]:
        """Convert ledger decisions into Signals enriched with MC data (for Kelly sizing)."""
        signals: Dict[str, Signal] = {}

        for decision in decisions:
            if decision.action not in ("BUY", "SELL", "CLOSE", "SHORT", "COVER"):
                continue
            if decision.action in ("BUY", "SHORT") and decision.confidence < self.confidence_threshold:
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
        ledger = await self.make_trading_decisions(
            technical_data, mc_predictions, competition, databento_sources
        )
        signals = self.decisions_to_signals(
            ledger.decisions, technical_data, mc_predictions
        )
        return ledger.decisions, signals

    async def execute(self) -> bool:
        self.log_action("Internal Twin Ledger agent ready")
        return True
