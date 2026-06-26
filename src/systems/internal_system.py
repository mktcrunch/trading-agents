"""
Internal System (System B) Orchestrator — Twin Ledger style.
Competes against Baseline using MC predictions, Kelly sizing, and structured LLM decisions.
"""
import asyncio
from src.logger import setup_logger
from src.agents.data_agent_internal import InternalDataAgent
from src.agents.signal_agent_internal import InternalSignalAgent
from src.agents.competition_context import build_competition_context
from src.agents.execution_agent import ExecutionAgent
from src.agents.risk_agent import RiskAgent, entry_sides_from_decisions
from src.agents.monitor_agent import MonitorAgent
from src.strategies.signal_generator import SignalGenerator
from src.strategies.allocator import PositionAllocator
from src.strategies.order_manager import OrderManager
from src import config

logger = setup_logger(__name__)


class InternalSystem:
    """
    System B (Internal) — Twin Ledger competitor
    - MarketCrunch predictions + Kelly Criterion sizing
    - Same structured BUY/SELL/HOLD/CLOSE decisions as baseline
    - Goal: compound returns and hold #1 on the Twin Ledger (not defensive cash when ahead)
    - Paper trades on Alpaca Account #2
    """

    def __init__(self):
        self.system_name = "internal"
        self.data_agent = InternalDataAgent()
        self.signal_agent = InternalSignalAgent()
        self.execution_agent = ExecutionAgent(system="internal")
        self.risk_agent = RiskAgent(system="internal")
        self.monitor_agent = MonitorAgent(system="internal")
        self.order_manager = OrderManager()
        self.logger = logger

    async def run_daily_workflow(self):
        """Execute daily Twin Ledger trading workflow."""
        from src.market.calendar import check_overnight_trading_session

        session_ok, session_reason = check_overnight_trading_session(system="internal")
        if not session_ok:
            self.logger.info(f"[INTERNAL] Overnight skipped: {session_reason}")
            return True

        if config.USE_ADK:
            from src.adk.workflows.daily_pipeline import run_daily_trading_pipeline

            result = await run_daily_trading_pipeline("internal")
            if not result.get("success"):
                self.logger.error(result.get("error") or "Daily pipeline failed")
                return False
            dry = config.is_dry_run()
            self.logger.info(
                f"[INTERNAL] Pipeline complete — "
                f"{result.get('actionable_count', 0)} actionable, "
                f"orders={'simulated (dry run)' if dry else result.get('orders_placed', 0)}"
            )
            return True

        self.logger.info("=" * 60)
        self.logger.info("[INTERNAL SYSTEM] Twin Ledger daily workflow started")
        self.logger.info("=" * 60)

        # Step 1: Portfolio + competition context
        self.logger.info("\n[Step 1] Fetching account info & competition context...")
        account_info = await self.data_agent.get_account_info()
        if not account_info:
            self.logger.error("Failed to get account info")
            return False

        competition = build_competition_context("internal")
        lb = competition["leaderboard"]
        self.logger.info(
            f"Leaderboard: rank {lb['your_rank']}/2 | "
            f"{lb['status']} Baseline Trader by ${lb['value_gap_usd']:,.2f}"
        )
        self.logger.info(
            f"  You: ${competition['your_portfolio']['portfolio_value']:,.2f} "
            f"({competition['your_portfolio']['pnl_pct']:+.2f}%)"
        )
        self.logger.info(
            f"  Baseline: ${competition['competitor']['portfolio_value']:,.2f} "
            f"({competition['competitor']['pnl_pct']:+.2f}%)"
        )

        # Step 2: MarketCrunch predictions
        self.logger.info("\n[Step 2] Fetching MarketCrunch predictions...")
        mc_predictions = await self.data_agent.fetch_mc_predictions(config.TICKER_UNIVERSE)
        self.logger.info(f"Fetched MC predictions for {len(mc_predictions)} tickers")

        # Step 3: Alpaca market data for full universe
        self.logger.info("\n[Step 3] Fetching Alpaca market data...")
        price_data = await self.data_agent.fetch_price_data(config.TICKER_UNIVERSE)
        technical_data = SignalGenerator.build_technical_data(price_data)
        if not technical_data:
            self.logger.error("No technical data available")
            return False

        databento_sources = await self.data_agent.enrich_with_databento(config.TICKER_UNIVERSE)
        if databento_sources:
            self.logger.info(f"Enriched with DataBento data for {len(databento_sources)} tickers")

        # Step 4: Twin Ledger decisions (MC + Kelly + technicals)
        self.logger.info("\n[Step 4] Twin Ledger trading decisions...")
        decisions, signals = await self.signal_agent.run_ledger_cycle(
            technical_data,
            mc_predictions,
            competition,
            databento_sources,
        )
        if not decisions:
            self.logger.warning("No ledger decisions returned")
            return True

        actionable = [d for d in decisions if d.action != "HOLD"]
        self.logger.info(f"Actionable decisions: {len(actionable)}")
        for d in actionable:
            sig = signals.get(d.ticker)
            mc_note = (
                f"MC_ret={sig.predicted_return*100:.2f}% conf={sig.confidence:.2f}"
                if sig else "no MC signal"
            )
            self.logger.info(
                f"  {d.action} {d.ticker} | size={d.size_pct:.1%} | "
                f"conf={d.confidence:.2f} | {mc_note} | {d.rationale[:80]}"
            )

        # Step 5: Risk validation (Kelly weights for BUY and SHORT)
        self.logger.info("\n[Step 5] Risk validation...")
        current_positions = await self.data_agent.get_current_positions()
        entry_decisions = [d for d in decisions if d.action in ("BUY", "SHORT")]
        entry_sides = entry_sides_from_decisions(entry_decisions)
        entry_signals = {t: s for t, s in signals.items() if t in entry_sides}
        proposed_weights = PositionAllocator.internal_entry_target_weights(
            entry_signals, entry_sides
        )
        validation_results = await self.risk_agent.validate_positions(
            proposed_weights,
            float(account_info.get("portfolio_value", 0)),
            current_positions,
            entry_sides=entry_sides,
            open_orders_raw=self.execution_agent.alpaca_client.get_orders(status="open"),
        )

        valid_entries = {t for t, ok in validation_results.items() if ok}
        filtered_decisions = [
            d for d in decisions
            if d.action not in ("BUY", "SHORT") or d.ticker in valid_entries
        ]
        filtered_entry_signals = {
            t: s for t, s in entry_signals.items() if t in valid_entries
        }
        filtered_entry_sides = {
            t: entry_sides[t] for t in valid_entries if t in entry_sides
        }
        self.logger.info(
            f"Valid entry decisions after risk check: "
            f"{len([d for d in filtered_decisions if d.action in ('BUY', 'SHORT')])}"
        )

        # Step 6: Kelly allocation + sell/close sizing
        self.logger.info("\n[Step 6] Building position changes (Kelly + decisions)...")
        latest_prices = {
            t: technical_data[t].get("close", 0)
            for t in technical_data
            if technical_data[t].get("close")
        }
        position_changes = PositionAllocator.allocate_internal_from_decisions(
            filtered_decisions,
            filtered_entry_signals,
            float(account_info.get("portfolio_value", 0)),
            current_positions,
            latest_prices,
            entry_sides=filtered_entry_sides,
        )
        self.logger.info(f"Orders to place: {len(position_changes)}")

        # Step 7: Place orders
        self.logger.info("\n[Step 7] Placing overnight orders...")
        if position_changes:
            self.order_manager.build_overnight_orders(
                position_changes, latest_prices, spread_pct=0.5
            )
            order_ids = await self.execution_agent.place_overnight_orders(
                position_changes, latest_prices, current_positions
            )
            placed = len([o for o in order_ids.values() if o])
            self.logger.info(f"Placed {placed} orders")
        else:
            self.logger.info("No orders to place (all HOLD or filtered out)")

        # Step 8: Monitor
        self.logger.info("\n[Step 8] Portfolio monitoring...")
        metrics = await self.monitor_agent.get_portfolio_metrics()
        await self.monitor_agent.log_daily_performance(metrics)

        self.logger.info("=" * 60)
        self.logger.info("[INTERNAL SYSTEM] Twin Ledger workflow complete")
        self.logger.info("=" * 60)
        return True

    async def post_market_open_chase(self):
        """Chase unfilled orders post-market-open (9:35 AM EST)"""
        self.logger.info("\n[Post-Open] Chasing unfilled orders...")
        new_orders = await self.execution_agent.chase_unfilled_orders(fill_threshold=0.70)
        self.logger.info(f"Chased {len(new_orders)} unfilled orders")

    async def eod_position_check(self):
        """EOD check: trim losers if next-day confidence < 0.55 (4:10 PM EST)"""
        self.logger.info("\n[Step EOD] Position check at market close...")
        from src.apis.marketcrunch_client import MarketCrunchClient
        from src.agents.ledger_utils import mc_confidence_score

        mc_client = MarketCrunchClient()
        current_positions = await self.data_agent.get_current_positions()
        trim_candidates = []

        for ticker, position in current_positions.items():
            analysis = mc_client.get_ai_estimates(ticker)
            if analysis:
                ai_est = analysis.get("ai_estimate", {})
                conf_score = mc_confidence_score(ai_est.get("confidence", "Low"))
                if conf_score < 0.55:
                    trim_candidates.append(ticker)

        if trim_candidates:
            self.logger.info(f"EOD trim: {len(trim_candidates)} positions (confidence < 0.55)")
        else:
            self.logger.info("EOD check: No positions to trim")

    def start(self):
        """Start the internal trading system"""
        self.logger.info("Starting Internal Twin Ledger System")
        asyncio.run(self.run_daily_workflow())
