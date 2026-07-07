#!/usr/bin/env python3
"""
Master entry point: Run both Baseline and Internal trading systems simultaneously.

Usage:
    python main.py              # Run both systems in parallel (discovery before internal)
    python main.py --baseline   # Run only baseline system
    python main.py --baseline --dry-run   # Signal + audit only, no Alpaca orders
    python main.py --internal   # Run discovery (if stale) + internal system
    python main.py --internal --dry-run
    python main.py --discovery  # Run DataBento discovery only
    python main.py --overnight  # EOD workflow for both systems (2:00 PM PT job)
    python main.py --risk       # Intraday risk check for both accounts
    python main.py --serve      # Cloud Run HTTP server (scheduler triggers)
    python main.py --reconcile-orders  # Cancel duplicate OPG orders (both accounts)
"""

import asyncio
import sys

from src.adk.model import configure_genai_env

configure_genai_env()

from src import config
from src.systems.baseline_system import BaselineSystem
from src.systems.internal_system import InternalSystem
from src.agents.discovery_agent import DiscoveryAgent
from src.risk.risk_monitor import run_risk_all, run_risk_for_system
from src.audit import end_trace, start_trace
from src.logger import setup_logger

logger = setup_logger(__name__)


async def run_discovery(force: bool = False, traced: bool = True) -> bool:
    """Run DataBento discovery pipeline."""
    if not config.DATABENTO_DISCOVERY_ENABLED:
        logger.info(
            "DataBento discovery disabled (DATABENTO_DISCOVERY_ENABLED=false) — "
            "using cached approved sources only"
        )
        if traced:
            start_trace("discovery", system="discovery")
            end_trace(
                "discovery",
                system="discovery",
                success=True,
                summary={"skipped": True, "reason": "discovery_disabled"},
            )
        return True

    if traced:
        start_trace("discovery", system="discovery")
    logger.info("=" * 80)
    logger.info("STARTING DATABENTO DISCOVERY AGENT")
    logger.info("=" * 80)

    agent = DiscoveryAgent()
    try:
        result = await agent.ensure_fresh_sources(force=force)
        approved = result.get("summary", {}).get("approved_count", 0)
        tickers = result.get("summary", {}).get("tickers_with_features", 0)
        logger.info(f"Discovery result: {approved} approved sources, {tickers} tickers")
        if traced:
            end_trace("discovery", system="discovery", success=True, summary={"approved": approved, "tickers": tickers})
        return True
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        if traced:
            end_trace("discovery", system="discovery", success=False, summary={"error": str(e)})
        return False


async def run_both_systems():
    """Run baseline and internal systems in parallel."""
    start_trace("daily", system="both")
    logger.info("=" * 80)
    logger.info("STARTING DUAL TRADING SYSTEM")
    logger.info("=" * 80)
    logger.info("")
    logger.info("System A (Baseline): Technical indicators only")
    logger.info("System B (Internal): MarketCrunch predictions + DataBento")
    logger.info("")

    if not await run_discovery(traced=False):
        logger.warning("Discovery did not complete — internal will run without DataBento")

    baseline = BaselineSystem()
    internal = InternalSystem()

    try:
        await asyncio.gather(
            baseline.run_daily_workflow(),
            internal.run_daily_workflow(),
        )
    except Exception as e:
        logger.error(f"Error running systems: {e}")
        end_trace("daily", system="both", success=False, summary={"error": str(e)})
        return False

    logger.info("")
    logger.info("=" * 80)
    logger.info("DUAL SYSTEM EXECUTION COMPLETE")
    logger.info("=" * 80)
    end_trace("daily", system="both", success=True)
    return True


async def run_baseline_only(dry_run: bool = False):
    """Run only the baseline system."""
    start_trace("daily", system="baseline", meta={"dry_run": dry_run})
    logger.info("=" * 80)
    logger.info("STARTING BASELINE SYSTEM (System A)" + (" — DRY RUN" if dry_run else ""))
    logger.info("=" * 80)

    baseline = BaselineSystem()

    try:
        with config.dry_run_mode(dry_run):
            await baseline.run_daily_workflow()
    except Exception as e:
        logger.error(f"Error running baseline system: {e}")
        end_trace("daily", system="baseline", success=False, summary={"error": str(e)})
        return False

    logger.info("=" * 80)
    logger.info("BASELINE SYSTEM COMPLETE")
    logger.info("=" * 80)
    end_trace("daily", system="baseline", success=True, summary={"dry_run": dry_run})
    return True


async def run_internal_only(dry_run: bool = False):
    """Run discovery (if stale) then internal system."""
    start_trace("daily", system="internal", meta={"dry_run": dry_run})
    logger.info("=" * 80)
    logger.info("STARTING INTERNAL SYSTEM (System B)" + (" — DRY RUN" if dry_run else ""))
    logger.info("=" * 80)

    if not await run_discovery(traced=False):
        logger.warning("Discovery did not complete — continuing without DataBento enrichment")

    internal = InternalSystem()

    try:
        with config.dry_run_mode(dry_run):
            await internal.run_daily_workflow()
    except Exception as e:
        logger.error(f"Error running internal system: {e}")
        end_trace("daily", system="internal", success=False, summary={"error": str(e)})
        return False

    logger.info("=" * 80)
    logger.info("INTERNAL SYSTEM COMPLETE")
    logger.info("=" * 80)
    end_trace("daily", system="internal", success=True, summary={"dry_run": dry_run})
    return True


async def run_overnight_job(dry_run: bool = False):
    """
    Scheduled EOD job (~2:00 PM PT): discovery + both systems place overnight OPG orders.
    """
    start_trace("overnight", system="both", meta={"dry_run": dry_run})
    logger.info("=" * 80)
    logger.info(
        "OVERNIGHT JOB — discovery + baseline + internal"
        + (" — DRY RUN" if dry_run else "")
    )
    logger.info("=" * 80)

    from src.market.calendar import check_overnight_trading_session

    session_ok, session_reason = check_overnight_trading_session()
    if not session_ok:
        logger.info(f"OVERNIGHT JOB SKIPPED: {session_reason}")
        end_trace(
            "overnight",
            system="both",
            success=True,
            summary={"skipped": True, "skip_reason": session_reason},
        )
        return True

    if not await run_discovery(traced=False):
        logger.warning("Discovery incomplete — internal runs without fresh DataBento")

    baseline = BaselineSystem()
    internal = InternalSystem()
    try:
        with config.dry_run_mode(dry_run):
            await asyncio.gather(
                baseline.run_daily_workflow(),
                internal.run_daily_workflow(),
            )
    except Exception as e:
        logger.error(f"Overnight job failed: {e}")
        end_trace("overnight", system="both", success=False, summary={"error": str(e)})
        return False

    logger.info("OVERNIGHT JOB COMPLETE")
    end_trace("overnight", system="both", success=True, summary={"dry_run": dry_run})
    return True


async def run_reconcile_orders():
    """Cancel duplicate OPG open orders on both accounts."""
    from src.apis.alpaca_client import AlpacaClient
    from src.audit import record_event
    from src.strategies.order_dedup import reconcile_duplicate_open_orders

    start_trace("reconcile", system="both")
    logger.info("=" * 80)
    logger.info("RECONCILE OPEN ORDERS — baseline + internal")
    logger.info("=" * 80)

    total_cancelled = 0
    for system in ("baseline", "internal"):
        client = AlpacaClient(system=system)
        before = len(client.get_orders(status="open"))
        cancelled = reconcile_duplicate_open_orders(client)
        after = len(client.get_orders(status="open"))
        total_cancelled += len(cancelled)
        logger.info(
            f"[{system}] open orders {before} → {after}, cancelled {len(cancelled)} duplicates"
        )
        for c in cancelled:
            logger.info(
                f"  cancelled {c['side']} {c['symbol']} {c['cancelled_qty']:.0f} sh "
                f"@ ${c['limit_price']:.2f}"
            )
            record_event(
                event_type="order_cancelled_duplicate",
                action=f"Reconcile cancelled {c['side']} {c['symbol']}",
                system=system,
                agent="reconcile",
                payload=c,
            )

    logger.info(f"Reconcile complete: {total_cancelled} duplicate orders cancelled")
    end_trace("reconcile", system="both", success=True, summary={"cancelled": total_cancelled})
    return True


async def run_risk_job(system: str = "both", dry_run: bool = False):
    """Intraday risk monitor — runs during market hours via scheduler."""
    start_trace("risk", system=system, meta={"dry_run": dry_run})
    logger.info("=" * 80)
    logger.info(f"RISK JOB — system={system} dry_run={dry_run}")
    logger.info("=" * 80)

    try:
        if system == "baseline":
            result = run_risk_for_system("baseline", dry_run=dry_run)
        elif system == "internal":
            result = run_risk_for_system("internal", dry_run=dry_run)
        else:
            result = run_risk_all(dry_run=dry_run)
        logger.info(f"Risk result: {result}")
        end_trace("risk", system=system, success=True, summary=result)
        return True
    except Exception as e:
        logger.error(f"Risk job failed: {e}")
        end_trace("risk", system=system, success=False, summary={"error": str(e)})
        return False


async def run_chase_job(system: str = "both"):
    """Post-market-open chase unfilled limit orders with market orders."""
    from src.market.calendar import check_chase_trading_session

    session_ok, session_reason = check_chase_trading_session(
        system="baseline" if system == "both" else system,
    )
    if not session_ok:
        logger.info(f"CHASE JOB SKIPPED: {session_reason}")
        start_trace("chase", system=system, meta={"skipped": True, "reason": session_reason})
        end_trace(
            "chase",
            system=system,
            success=True,
            summary={"skipped": True, "skip_reason": session_reason},
        )
        return True

    start_trace("chase", system=system)
    logger.info("=" * 80)
    logger.info(f"CHASE JOB — system={system}")
    logger.info("=" * 80)

    try:
        from src.agents.execution_agent import ExecutionAgent
        results = {}
        if system in ("baseline", "both"):
            agent = ExecutionAgent(system="baseline")
            results["baseline"] = await agent.chase_unfilled_orders(fill_threshold=0.70)
        if system in ("internal", "both"):
            agent = ExecutionAgent(system="internal")
            results["internal"] = await agent.chase_unfilled_orders(fill_threshold=0.70)
            
        logger.info(f"Chase result: {results}")
        end_trace("chase", system=system, success=True, summary=results)
        return True
    except Exception as e:
        logger.error(f"Chase job failed: {e}")
        end_trace("chase", system=system, success=False, summary={"error": str(e)})
        return False


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        dry_run = "--dry-run" in sys.argv
        positional = [
            a for a in sys.argv[1:]
            if a not in ("--dry-run", "--force")
        ]
        if not positional:
            print("Usage: python main.py <command> [--dry-run] [--force]")
            sys.exit(1)
        arg = positional[0]
        if arg == "--baseline":
            success = asyncio.run(run_baseline_only(dry_run=dry_run))
        elif arg == "--internal":
            success = asyncio.run(run_internal_only(dry_run=dry_run))
        elif arg == "--discovery":
            force = "--force" in sys.argv
            success = asyncio.run(run_discovery(force=force))
        elif arg == "--overnight":
            success = asyncio.run(run_overnight_job(dry_run=dry_run))
        elif arg == "--risk":
            system = "both"
            if "--baseline" in sys.argv:
                system = "baseline"
            elif "--internal" in sys.argv:
                system = "internal"
            success = asyncio.run(run_risk_job(system=system, dry_run=dry_run))
        elif arg == "--chase":
            system = "both"
            if "--baseline" in sys.argv:
                system = "baseline"
            elif "--internal" in sys.argv:
                system = "internal"
            success = asyncio.run(run_chase_job(system=system))
        elif arg == "--serve":
            from server import run_server
            run_server()
            return
        elif arg == "--reconcile-orders":
            success = asyncio.run(run_reconcile_orders())
        else:
            print(f"Unknown argument: {arg}")
            print(
                "Usage: python main.py "
                "[--baseline|--internal|--discovery|--overnight|--risk|--chase|--serve|--reconcile-orders] "
                "[--force] [--dry-run]"
            )
            sys.exit(1)
    else:
        success = asyncio.run(run_both_systems())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
