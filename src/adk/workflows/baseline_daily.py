"""ADK Workflow for Baseline Twin Ledger daily pipeline."""
from __future__ import annotations

from typing import Any, Dict

from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import Workflow, node
from google.adk.workflow._base_node import START
from google.genai import types

from src import config
from src.adk.model import configure_genai_env
from src.adk.workflows.common import (
    fetch_news_for_universe,
    invoke_signal_agent,
    ledger_to_state,
    workflow_daily_setup,
)
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


@node(name="baseline_fetch_context", rerun_on_resume=True)
async def baseline_fetch_context(ctx):
    """Fetch account, competition, technical data, and news."""
    data_agent = BaselineDataAgent()
    account_info = await data_agent.get_account_info()
    if not account_info:
        ctx.state["error"] = "Failed to get account info"
        ctx.state["workflow_result"] = {"success": False, "error": "Failed to get account info"}
        return {"success": False}

    competition = build_competition_context("baseline")
    price_data = await data_agent.fetch_price_data(config.TICKER_UNIVERSE)
    from src.strategies.signal_generator import SignalGenerator

    technical_data = SignalGenerator.build_technical_data(price_data)
    news_data = fetch_news_for_universe()

    ctx.state["account_info"] = account_info
    ctx.state["competition"] = competition
    ctx.state["technical_data"] = technical_data
    ctx.state["news_data"] = news_data
    ctx.state["current_positions"] = await data_agent.get_current_positions()
    return {"tickers": len(technical_data)}


@node(name="baseline_signal_decisions", rerun_on_resume=True)
async def baseline_signal_decisions(ctx):
    """Generate Twin Ledger decisions via ADK signal LlmAgent (ctx.run_node)."""
    technical_data = ctx.state.get("technical_data") or {}
    competition = ctx.state.get("competition") or {}
    news_data = ctx.state.get("news_data") or {}

    learning_block = ""
    if config.LEARNING_ENABLED:
        from src.learning.context import build_signal_learning_block

        learning_block = build_signal_learning_block("baseline")

    ledger = await invoke_signal_agent(
        ctx,
        "baseline",
        {
            "system": "baseline",
            "competition": competition,
            "technical_data": technical_data,
            "news_data": news_data,
            "signal_learning": learning_block,
            "valid_tickers": list(config.TICKER_UNIVERSE),
        },
    )
    signal_agent = BaselineSignalAgent()
    from src.agents.ledger_utils import emit_signal_ledger_audit

    emit_signal_ledger_audit(signal_agent, ledger, competition)
    ctx.state.update(ledger_to_state(ledger))
    return {
        "decisions": len(ledger.decisions),
        "no_action_rationale": ledger.no_action_rationale or "",
    }


@node(name="baseline_risk_and_execute", rerun_on_resume=True)
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
    no_action_rationale = ctx.state.get("no_action_rationale") or ""

    actionable = [d for d in decisions if d.action != "HOLD"]
    if not actionable:
        result = {
            "success": True,
            "pipeline": "adk_workflow",
            "system": "baseline",
            "orders_placed": 0,
            "dry_run": config.is_dry_run(),
            "message": "No actionable decisions from signal step",
            "no_action_rationale": no_action_rationale,
        }
        ctx.state["workflow_result"] = result
        ctx.state["orders_placed"] = 0
        ctx.state["success"] = True
        return result

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
    placed = 0
    if position_changes:
        OrderManager().build_overnight_orders(position_changes, latest_prices, spread_pct=0.5)
        order_ids = await execution_agent.place_overnight_orders(
            position_changes, latest_prices, current_positions
        )
        placed = len([o for o in order_ids.values() if o])

    result = {
        "success": True,
        "pipeline": "adk_workflow",
        "system": "baseline",
        "orders_placed": placed,
        "dry_run": config.is_dry_run(),
        "decisions_count": len(decisions),
        "actionable_count": len([d for d in decisions if d.action != "HOLD"]),
        "validation_results": validation,
    }
    ctx.state["workflow_result"] = result
    ctx.state["orders_placed"] = placed
    ctx.state["success"] = True
    return result


@node(name="baseline_monitor", rerun_on_resume=True)
async def baseline_monitor(ctx):
    """Log portfolio metrics and finalize workflow result."""
    monitor = MonitorAgent(system="baseline")
    metrics = await monitor.get_portfolio_metrics()
    if metrics:
        await monitor.log_daily_performance(metrics)

    result = dict(ctx.state.get("workflow_result") or {})
    result.setdefault("success", ctx.state.get("success", True))
    result.setdefault("system", "baseline")
    result.setdefault("pipeline", "adk_workflow")
    result["metrics"] = metrics
    ctx.state["workflow_result"] = result

    from src.audit import end_trace

    end_trace(
        "daily",
        system="baseline",
        success=bool(result.get("success")),
        summary={
            "orders_placed": result.get("orders_placed", 0),
            "dry_run": config.is_dry_run(),
            "pipeline": "adk_workflow",
        },
    )
    return result


def build_baseline_daily_workflow() -> Workflow:
    return Workflow(
        name="baseline_daily_workflow",
        state_schema=None,
        edges=[
            (START, baseline_fetch_context, baseline_signal_decisions, baseline_risk_and_execute, baseline_monitor),
        ],
    )


async def run_baseline_daily_adk() -> Dict[str, Any]:
    """Run full baseline daily pipeline through ADK Workflow + Runner."""
    skipped = await workflow_daily_setup("baseline")
    if skipped:
        return skipped

    configure_genai_env()
    workflow = build_baseline_daily_workflow()
    app = App(name="twin_ledger_baseline_workflow", root_agent=workflow)
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service, auto_create_session=True)
    session_id = "baseline_daily_adk"

    logger.info("[ADK Workflow] Starting baseline_daily_workflow")
    try:
        async for _event in runner.run_async(
            user_id="twin_ledger",
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="Run baseline Twin Ledger daily workflow.")],
            ),
        ):
            pass
    except Exception as e:
        logger.exception("[ADK Workflow] baseline_daily_workflow failed")
        from src.audit import end_trace

        end_trace("daily", system="baseline", success=False, summary={"error": str(e)})
        return {"success": False, "pipeline": "adk_workflow", "system": "baseline", "error": str(e)}

    session = await session_service.get_session(
        app_name=app.name,
        user_id="twin_ledger",
        session_id=session_id,
    )
    state = (session.state if session else {}) or {}
    result = state.get("workflow_result") or {
        "success": state.get("success", True),
        "orders_placed": state.get("orders_placed", 0),
        "pipeline": "adk_workflow",
        "system": "baseline",
        "dry_run": config.is_dry_run(),
    }
    logger.info(f"[ADK Workflow] baseline_daily_workflow complete: {result}")
    return result
