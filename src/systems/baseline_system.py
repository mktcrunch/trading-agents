"""
Baseline System (System A) Orchestrator — Twin Ledger style.
Competes against Internal System (System B) using technicals + LLM strategy only.
"""
import asyncio
from src.logger import setup_logger
from src.agents.data_agent_baseline import BaselineDataAgent
from src.agents.signal_agent_baseline import BaselineSignalAgent
from src.agents.competition_context import build_competition_context
from src.agents.execution_agent import ExecutionAgent
from src.agents.risk_agent import RiskAgent, entry_sides_from_decisions
from src.agents.monitor_agent import MonitorAgent
from src.strategies.signal_generator import SignalGenerator
from src.strategies.allocator import PositionAllocator
from src.strategies.order_manager import OrderManager
from src import config

logger = setup_logger(__name__)


class BaselineSystem:
    """
    System A (Baseline) — Twin Ledger competitor
    - Technical indicators from Alpaca only
    - LLM makes structured BUY/SELL/HOLD/CLOSE decisions
    - Goal: compound returns and hold #1 on the Twin Ledger (not defensive cash when ahead)
    - Paper trades on Alpaca Account #1
    """

    def __init__(self):
        self.system_name = "baseline"
        self.data_agent = BaselineDataAgent()
        self.signal_agent = BaselineSignalAgent()
        self.execution_agent = ExecutionAgent(system="baseline")
        self.risk_agent = RiskAgent(system="baseline")
        self.monitor_agent = MonitorAgent(system="baseline")
        self.order_manager = OrderManager()
        self.logger = logger

    async def run_daily_workflow(self):
        """Execute daily Twin Ledger trading workflow."""
        from src.market.calendar import check_overnight_trading_session

        session_ok, session_reason = check_overnight_trading_session(system="baseline")
        if not session_ok:
            self.logger.info(f"[BASELINE] Overnight skipped: {session_reason}")
            return True

        if config.USE_ADK:
            from src.adk.workflows.daily_pipeline import run_daily_trading_pipeline

            result = await run_daily_trading_pipeline("baseline")
            if not result.get("success"):
                self.logger.error(result.get("error") or "Daily pipeline failed")
                return False
            dry = config.is_dry_run()
            self.logger.info(
                f"[BASELINE] Pipeline complete — "
                f"{result.get('actionable_count', 0)} actionable, "
                f"orders={'simulated (dry run)' if dry else result.get('orders_placed', 0)}"
            )
            return True

        self.logger.info("=" * 60)
        self.logger.info("[BASELINE SYSTEM] Twin Ledger daily workflow started")
        self.logger.info("=" * 60)

        # Step 1: Portfolio + competition context
        self.logger.info("\n[Step 1] Fetching account info & competition context...")
        account_info = await self.data_agent.get_account_info()
        if not account_info:
            self.logger.error("Failed to get account info")
            return False

        competition = build_competition_context("baseline")
        lb = competition["leaderboard"]
        self.logger.info(
            f"Leaderboard: rank {lb['your_rank']}/2 | "
            f"{lb['status']} Internal Trader by ${lb['value_gap_usd']:,.2f}"
        )
        self.logger.info(
            f"  You: ${competition['your_portfolio']['portfolio_value']:,.2f} "
            f"({competition['your_portfolio']['pnl_pct']:+.2f}%)"
        )
        self.logger.info(
            f"  Internal: ${competition['competitor']['portfolio_value']:,.2f} "
            f"({competition['competitor']['pnl_pct']:+.2f}%)"
        )

        # Step 2: Market data for full universe
        self.logger.info("\n[Step 2] Fetching Alpaca market data...")
        price_data = await self.data_agent.fetch_price_data(config.TICKER_UNIVERSE)
        technical_data = SignalGenerator.build_technical_data(price_data)
        if not technical_data:
            self.logger.error("No technical data available")
            return False

        # Step 3: Twin Ledger decisions
        self.logger.info("\n[Step 3] Twin Ledger trading decisions...")
        decisions, signals = await self.signal_agent.run_ledger_cycle(
            technical_data, competition
        )
        if not decisions:
            self.logger.warning("No ledger decisions returned")
            return True

        actionable = [d for d in decisions if d.action != "HOLD"]
        self.logger.info(f"Actionable decisions: {len(actionable)}")
        for d in actionable:
            self.logger.info(
                f"  {d.action} {d.ticker} | size={d.size_pct:.1%} | "
                f"conf={d.confidence:.2f} | {d.rationale[:100]}"
            )

        # Step 4: Risk validation (BUY/SHORT entry weights)
        self.logger.info("\n[Step 4] Risk validation...")
        current_positions = await self.data_agent.get_current_positions()
        entry_decisions = [d for d in decisions if d.action in ("BUY", "SHORT")]
        proposed_weights = PositionAllocator.decision_target_weights(entry_decisions)
        validation_results = await self.risk_agent.validate_positions(
            proposed_weights,
            float(account_info.get("portfolio_value", 0)),
            current_positions,
            entry_sides=entry_sides_from_decisions(entry_decisions),
            open_orders_raw=self.execution_agent.alpaca_client.get_orders(status="open"),
        )

        valid_entries = {t for t, ok in validation_results.items() if ok}
        filtered_decisions = [
            d for d in decisions
            if d.action not in ("BUY", "SHORT") or d.ticker in valid_entries
        ]
        self.logger.info(
            f"Valid entry decisions after risk check: "
            f"{len([d for d in filtered_decisions if d.action in ('BUY', 'SHORT')])}"
        )

        # Step 5: Convert decisions to orders
        self.logger.info("\n[Step 5] Building position changes...")
        latest_prices = {
            t: technical_data[t].get("close", 0)
            for t in technical_data
            if technical_data[t].get("close")
        }
        position_changes = PositionAllocator.allocate_from_decisions(
            filtered_decisions,
            float(account_info.get("portfolio_value", 0)),
            current_positions,
            latest_prices,
        )
        self.logger.info(f"Orders to place: {len(position_changes)}")

        # Step 6: Place orders
        self.logger.info("\n[Step 6] Placing overnight orders...")
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

        # Step 7: Monitor
        self.logger.info("\n[Step 7] Portfolio monitoring...")
        metrics = await self.monitor_agent.get_portfolio_metrics()
        await self.monitor_agent.log_daily_performance(metrics)

        self.logger.info("=" * 60)
        self.logger.info("[BASELINE SYSTEM] Twin Ledger workflow complete")
        self.logger.info("=" * 60)
        return True

    async def post_market_open_chase(self):
        """Chase unfilled orders post-market-open (9:35 AM EST)"""
        self.logger.info("\n[Post-Open] Chasing unfilled orders...")
        new_orders = await self.execution_agent.chase_unfilled_orders(fill_threshold=0.70)
        self.logger.info(f"Chased {len(new_orders)} unfilled orders")

    async def eod_position_check(self):
        """EOD check at market close."""
        self.logger.info("\n[EOD] Position check at market close...")
        self.logger.info("EOD check complete")

    def start(self):
        """Start the baseline trading system"""
        self.logger.info("Starting Baseline Twin Ledger System")
        asyncio.run(self.run_daily_workflow())
