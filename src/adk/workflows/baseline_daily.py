"""ADK Workflow for Baseline Twin Ledger daily pipeline."""
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import Workflow, node
from google.adk.workflow._base_node import START
from google.genai import types

from src import config
from src.adk.model import configure_genai_env
from src.agents.competition_context import build_competition_context
from src.agents.data_agent_baseline import BaselineDataAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.monitor_agent import MonitorAgent
from src.agents.risk_agent import RiskAgent, entry_sides_from_decisions
from src.agents.signal_agent_baseline import BaselineSignalAgent
from src.models.trading_decision import TradingDecision
from src.strategies.allocator import PositionAllocator
from src.strategies.order_manager import OrderManager
from src.logger import setup_logger

logger = setup_logger(__name__)


@node(name="baseline_fetch_context")
async def baseline_fetch_context(ctx):
    """Fetch account, competition, and technical data."""
    data_agent = BaselineDataAgent()
    account_info = await data_agent.get_account_info()
    if not account_info:
        ctx.state["error"] = "Failed to get account info"
        return {"success": False}

    competition = build_competition_context("baseline")
    price_data = await data_agent.fetch_price_data(config.TICKER_UNIVERSE)
    from src.strategies.signal_generator import SignalGenerator
    technical_data = SignalGenerator.build_technical_data(price_data)

    ctx.state["account_info"] = account_info
    ctx.state["competition"] = competition
    ctx.state["technical_data"] = technical_data
    ctx.state["current_positions"] = await data_agent.get_current_positions()
    return {"tickers": len(technical_data)}


@node(name="baseline_signal_decisions")
async def baseline_signal_decisions(ctx):
    """Generate Twin Ledger decisions via ADK LlmAgent."""
    technical_data = ctx.state.get("technical_data") or {}
    competition = ctx.state.get("competition") or {}

    signal_agent = BaselineSignalAgent()
    decisions, _ = await signal_agent.run_ledger_cycle(technical_data, competition)

    ctx.state["decisions"] = [d.to_dict() for d in decisions]
    return {"decisions": len(decisions)}


@node(name="baseline_risk_and_execute")
async def baseline_risk_and_execute(ctx):
    """Validate risk, allocate positions, place overnight orders."""
    decisions_raw = ctx.state.get("decisions") or []
    decisions = [
        TradingDecision.from_dict(d) for d in decisions_raw
        if TradingDecision.from_dict(d)
    ]
    account_info = ctx.state.get("account_info") or {}
    technical_data = ctx.state.get("technical_data") or {}
    current_positions = ctx.state.get("current_positions") or {}

    risk_agent = RiskAgent(system="baseline")
    entry_decisions = [d for d in decisions if d.action in ("BUY", "SHORT")]
    proposed_weights = PositionAllocator.decision_target_weights(entry_decisions)
    validation = await risk_agent.validate_positions(
        proposed_weights,
        float(account_info.get("portfolio_value", 0)),
        current_positions,
        entry_sides=entry_sides_from_decisions(entry_decisions),
    )
    valid_entries = {t for t, ok in validation.items() if ok}
    filtered = [
        d for d in decisions
        if d.action not in ("BUY", "SHORT") or d.ticker in valid_entries
    ]

    latest_prices = {
        t: technical_data[t].get("close", 0)
        for t in technical_data
        if technical_data[t].get("close")
    }
    position_changes = PositionAllocator.allocate_from_decisions(
        filtered,
        float(account_info.get("portfolio_value", 0)),
        current_positions,
        latest_prices,
    )

    execution_agent = ExecutionAgent(system="baseline")
    order_manager = OrderManager()
    placed = 0
    if position_changes:
        order_manager.build_overnight_orders(position_changes, latest_prices, spread_pct=0.5)
        order_ids = await execution_agent.place_overnight_orders(
            position_changes, latest_prices, current_positions
        )
        placed = len([o for o in order_ids.values() if o])

    ctx.state["orders_placed"] = placed
    ctx.state["success"] = True
    return {"orders_placed": placed}


@node(name="baseline_monitor")
async def baseline_monitor(ctx):
    """Log portfolio metrics."""
    monitor = MonitorAgent(system="baseline")
    metrics = await monitor.get_portfolio_metrics()
    await monitor.log_daily_performance(metrics)
    return metrics


def build_baseline_daily_workflow() -> Workflow:
    return Workflow(
        name="baseline_daily_workflow",
        state_schema=None,
        edges=[
            (START, baseline_fetch_context, baseline_signal_decisions, baseline_risk_and_execute, baseline_monitor),
        ],
    )


async def run_baseline_daily_adk() -> bool:
    """Run full baseline daily pipeline through ADK Workflow + Runner."""
    configure_genai_env()
    workflow = build_baseline_daily_workflow()
    app = App(name="twin_ledger_baseline_workflow", root_agent=workflow)
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service, auto_create_session=True)

    logger.info("[ADK Workflow] Starting baseline_daily_workflow")
    async for _event in runner.run_async(
        user_id="twin_ledger",
        session_id="baseline_daily",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Run baseline Twin Ledger daily workflow.")],
        ),
    ):
        pass

    logger.info("[ADK Workflow] baseline_daily_workflow complete")
    return True
