"""ADK FunctionTools wrapping Alpaca market data and account APIs."""
import json
from typing import Any, Dict, List, TYPE_CHECKING

from src import config
from src.apis.alpaca_client import AlpacaClient
from src.apis.price_fetcher import fetch_ohlcv_for_tickers
from src.audit.serialize import account_snapshot, positions_snapshot
from src.logger import setup_logger
from src.strategies.signal_generator import SignalGenerator

if TYPE_CHECKING:
    from src.models.trading_decision import TradingDecision

logger = setup_logger(__name__)


def _audit_ledger_decisions(system: str, decisions: List["TradingDecision"]) -> None:
    """Record LLM trading decisions in the active audit trace (overnight daily job)."""
    if not config.AUDIT_ENABLED:
        return
    from src.audit import record_event

    actionable = [d for d in decisions if d.action != "HOLD"]
    record_event(
        event_type="agent_action",
        action=(
            f"Ledger decisions: {len(actionable)} actionable / {len(decisions)} total"
        ),
        system=system,
        agent="SignalAgent",
        payload={
            "decisions": [d.to_dict() for d in actionable],
            "total": len(decisions),
            "actionable": len(actionable),
        },
    )
    for d in actionable:
        record_event(
            event_type="ledger_decision",
            action=(
                f"{d.action} {d.ticker} size={d.size_pct:.1%} "
                f"conf={d.confidence:.2f} — {d.rationale[:80]}"
            ),
            system=system,
            agent="SignalAgent",
            payload=d.to_dict(),
        )


def _client(system: str) -> AlpacaClient:
    return AlpacaClient(system=system)


def get_account_info(system: str = "baseline") -> Dict[str, Any]:
    """Return Alpaca account snapshot (cash, portfolio value, buying power).

    Args:
        system: 'baseline' or 'internal' — selects the Twin Ledger paper account.
    """
    account = _client(system).get_account()
    return account_snapshot(account)


def get_open_positions(system: str = "baseline") -> Dict[str, Any]:
    """Return open positions for the given Twin Ledger system account."""
    positions = _client(system).get_positions()
    return {"positions": positions_snapshot(positions), "count": len(positions)}


def get_technical_indicators(
    system: str = "baseline",
    tickers: str | None = None,
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Fetch OHLCV from Alpaca and compute RSI, MACD, Bollinger indicators.

    Args:
        system: 'baseline' or 'internal'.
        tickers: Comma-separated ticker list; defaults to full trading universe.
        lookback_days: History window for indicator calculation.
    """
    universe = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else list(config.TICKER_UNIVERSE)
    )
    client = _client(system)
    price_data = fetch_ohlcv_for_tickers(client, universe, lookback_days=lookback_days)
    technical = SignalGenerator.build_technical_data(price_data)
    return {
        "tickers": list(technical.keys()),
        "technical_data": technical,
    }


def _parse_mc_predictions_json(raw: str | None) -> Dict[str, Dict]:
    """Parse MC payload from coordinator or get_marketcrunch_predictions tool output."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse mc_predictions_json")
        return {}
    if not isinstance(data, dict):
        return {}
    if "predictions" in data and isinstance(data["predictions"], dict):
        return data["predictions"]
    if data and all(isinstance(v, dict) for v in data.values()):
        return data
    return {}


def _parse_technical_data_json(raw: str | None) -> Dict[str, Dict]:
    """Parse technical indicators payload from get_technical_indicators tool output."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse technical_data_json")
        return {}
    if not isinstance(data, dict):
        return {}
    if "technical_data" in data and isinstance(data["technical_data"], dict):
        return data["technical_data"]
    if data and all(isinstance(v, dict) for v in data.values()):
        return data
    return {}


def _mc_has_estimate(entry: Dict) -> bool:
    ai_est = entry.get("ai_estimate", {}) if entry else {}
    delta = float(ai_est.get("target_delta_numeric", 0) or 0)
    return delta != 0 or bool(ai_est.get("target_delta_pct"))


def _audit_mc_context(
    system: str,
    mc_predictions: Dict[str, Dict],
    source: str,
    fetched_tickers: List[str],
) -> None:
    if not config.AUDIT_ENABLED:
        return
    from src.audit import record_event

    summary = {}
    for ticker, entry in mc_predictions.items():
        ai_est = entry.get("ai_estimate", {})
        summary[ticker] = {
            "target_delta_numeric": ai_est.get("target_delta_numeric"),
            "confidence": ai_est.get("confidence"),
        }
    record_event(
        event_type="mc_context_loaded",
        action=f"MC context: {len(mc_predictions)} tickers via {source}",
        system=system,
        agent="InternalDataAgent",
        payload={
            "source": source,
            "fetched_fallback": fetched_tickers,
            "predictions": summary,
        },
    )


def get_latest_prices(
    system: str = "baseline",
    tickers: str | None = None,
) -> Dict[str, float]:
    """Return latest close prices for tickers from Alpaca OHLCV."""
    result = get_technical_indicators(system=system, tickers=tickers, lookback_days=30)
    technical = result.get("technical_data", {})
    return {
        t: float(data.get("close", 0) or 0)
        for t, data in technical.items()
        if data.get("close")
    }


async def execute_trading_decisions(
    system: str,
    decisions_json: str,
    mc_predictions_json: str | None = None,
    technical_data_json: str | None = None,
) -> Dict[str, Any]:
    """Validate risk, allocate positions, and place overnight limit orders on Alpaca.

    Args:
        system: 'baseline' or 'internal'.
        decisions_json: JSON string representing a list of TradingDecision objects.
        mc_predictions_json: Optional JSON from internal_data step (get_marketcrunch_predictions).
            Reused for Kelly sizing so execution does not re-fetch the full universe.
        technical_data_json: Optional JSON from internal_data step (get_technical_indicators).
    """
    import json
    from src.audit import start_trace, end_trace
    from src.gcs.store import get_gcs_store
    from src.models.trading_decision import TradingDecision
    from src.agents.risk_agent import RiskAgent
    from src.strategies.allocator import PositionAllocator
    from src.agents.execution_agent import ExecutionAgent
    from src.strategies.order_manager import OrderManager

    # 1. Hydrate audit log from GCS to avoid overwriting history
    try:
        get_gcs_store().hydrate_audit_log()
    except Exception as e:
        logger.warning(f"GCS audit hydrate failed before executing decisions: {e}")

    # 2. Start trace
    start_trace("daily", system=system)

    try:
        raw_list = json.loads(decisions_json)
        if isinstance(raw_list, dict) and "decisions" in raw_list:
            raw_list = raw_list["decisions"]
        decisions = []
        for item in raw_list:
            d = TradingDecision.from_dict(item)
            if d:
                decisions.append(d)
    except Exception as e:
        end_trace("daily", system=system, success=False, summary={"error": f"Failed to parse decisions: {e}"})
        return {"success": False, "error": f"Failed to parse decisions JSON: {e}"}

    if not decisions:
        end_trace("daily", system=system, success=True, summary={"message": "No decisions to execute"})
        return {"success": True, "orders_placed": 0, "message": "No decisions to execute"}

    _audit_ledger_decisions(system, decisions)

    try:
        # Fetch current state
        client = _client(system)
        account_info = client.get_account()
        if not account_info:
            end_trace("daily", system=system, success=False, summary={"error": "Failed to fetch account info"})
            return {"success": False, "error": "Failed to fetch account info"}
        portfolio_value = float(account_info.get("portfolio_value", 0))

        current_positions = client.get_positions()

        # Fetch latest prices for allocation
        latest_prices = get_latest_prices(system=system)

        risk_agent = RiskAgent(system=system)
        buy_decisions = [d for d in decisions if d.action == "BUY"]

        if system == "internal":
            from src.agents.signal_agent_internal import InternalSignalAgent
            from src.agents.data_agent_internal import InternalDataAgent

            actionable_tickers = sorted(
                {d.ticker for d in decisions if d.action in ("BUY", "SELL", "CLOSE")}
            )

            # Prefer MC + technicals passed from coordinator (same snapshot signal agent used)
            mc_predictions = _parse_mc_predictions_json(mc_predictions_json)
            technical_data = _parse_technical_data_json(technical_data_json)
            mc_source = "coordinator" if mc_predictions else "none"
            fetched_fallback: List[str] = []

            buy_tickers = [d.ticker for d in decisions if d.action == "BUY"]
            missing_mc = [
                t for t in buy_tickers
                if not _mc_has_estimate(mc_predictions.get(t, {}))
            ]
            if missing_mc:
                data_agent = InternalDataAgent()
                fetched = await data_agent.fetch_mc_predictions(missing_mc)
                mc_predictions.update(fetched)
                fetched_fallback = missing_mc
                mc_source = "coordinator+fetch" if mc_predictions_json else "fetch"

            if not technical_data and actionable_tickers:
                tech_result = get_technical_indicators(
                    system=system,
                    tickers=",".join(actionable_tickers),
                    lookback_days=90,
                )
                technical_data = tech_result.get("technical_data", {})

            latest_prices = {
                t: float(technical_data[t].get("close", 0) or 0)
                for t in technical_data
                if technical_data[t].get("close")
            }
            for ticker in actionable_tickers:
                if ticker not in latest_prices:
                    latest_prices.update(
                        get_latest_prices(system=system, tickers=ticker)
                    )

            _audit_mc_context(system, mc_predictions, mc_source, fetched_fallback)

            signal_agent = InternalSignalAgent()
            signals = signal_agent.decisions_to_signals(
                decisions, technical_data, mc_predictions
            )

            buy_signals = {
                t: s for t, s in signals.items()
                if any(d.action == "BUY" and d.ticker == t for d in decisions)
            }
            proposed_weights = PositionAllocator.internal_target_weights(buy_signals)
            validation = await risk_agent.validate_positions(
                proposed_weights,
                portfolio_value,
                current_positions,
            )
            valid_buys = {t for t, ok in validation.items() if ok}
            filtered = [d for d in decisions if d.action != "BUY" or d.ticker in valid_buys]
            filtered_buy_signals = {t: s for t, s in buy_signals.items() if t in valid_buys}

            position_changes = PositionAllocator.allocate_internal_from_decisions(
                filtered,
                filtered_buy_signals,
                portfolio_value,
                current_positions,
                latest_prices,
            )
        else:
            # Baseline uses equal weights
            technical_data = _parse_technical_data_json(technical_data_json)
            if technical_data:
                latest_prices = {
                    t: float(technical_data[t].get("close", 0) or 0)
                    for t in technical_data
                    if technical_data[t].get("close")
                }
                actionable_tickers = sorted(
                    {d.ticker for d in decisions if d.action in ("BUY", "SELL", "CLOSE")}
                )
                for ticker in actionable_tickers:
                    if ticker not in latest_prices:
                        latest_prices.update(
                            get_latest_prices(system=system, tickers=ticker)
                        )

            proposed_weights = PositionAllocator.decision_target_weights(buy_decisions)
            validation = await risk_agent.validate_positions(
                proposed_weights,
                portfolio_value,
                current_positions,
            )
            valid_buys = {t for t, ok in validation.items() if ok}
            filtered = [d for d in decisions if d.action != "BUY" or d.ticker in valid_buys]

            position_changes = PositionAllocator.allocate_from_decisions(
                filtered,
                portfolio_value,
                current_positions,
                latest_prices,
            )

        execution_agent = ExecutionAgent(system=system)
        order_manager = OrderManager()
        placed = 0

        from src import config as app_config

        dry = app_config.is_dry_run()
        if position_changes:
            order_manager.build_overnight_orders(position_changes, latest_prices, spread_pct=0.5)
            order_ids = await execution_agent.place_overnight_orders(
                position_changes, latest_prices, current_positions
            )
            if dry:
                placed = len([t for t, q in position_changes.items() if q])
            else:
                placed = len([o for o in order_ids.values() if o])

        # 3. End trace with success
        end_trace(
            "daily",
            system=system,
            success=True,
            summary={
                "orders_placed": placed,
                "dry_run": dry,
                "position_changes": position_changes,
                "validation_results": validation,
            }
        )

        return {
            "success": True,
            "orders_placed": placed,
            "dry_run": dry,
            "position_changes": position_changes,
            "validation_results": validation,
        }

    except Exception as e:
        end_trace("daily", system=system, success=False, summary={"error": str(e)})
        return {"success": False, "error": f"Execution failed: {e}"}


async def run_daily_trading_workflow(system: str) -> Dict[str, Any]:
    """Run the full overnight pipeline: data → signal → risk → execute.

    Deterministic path used by Cloud Scheduler and as coordinator fallback.
    Works for both ``baseline`` and ``internal`` (Internal passes MC snapshot to Kelly).

    Args:
        system: ``baseline`` or ``internal``.
    """
    from src.adk.workflows.daily_pipeline import run_daily_trading_pipeline

    return await run_daily_trading_pipeline(system)


async def run_intraday_risk_check(system: str, dry_run: bool = False) -> Dict[str, Any]:
    """Execute intraday risk monitor checks and place exit market orders if triggered.

    Args:
        system: 'baseline' or 'internal'.
        dry_run: If True, simulate exits without placing real orders.
    """
    from src.audit import start_trace, end_trace
    from src.gcs.store import get_gcs_store
    from src.risk.risk_monitor import run_risk_for_system

    # 1. Hydrate audit log from GCS to avoid overwriting history
    try:
        get_gcs_store().hydrate_audit_log()
    except Exception as e:
        logger.warning(f"GCS audit hydrate failed before risk check: {e}")

    # 2. Start trace
    start_trace("risk", system=system, meta={"dry_run": dry_run})

    try:
        result = run_risk_for_system(system, dry_run=dry_run)
        
        # 3. End trace with success
        end_trace("risk", system=system, success=True, summary=result)
        
        return {
            "success": True,
            "system": system,
            "dry_run": dry_run,
            "result": result,
        }
    except Exception as e:
        # 3. End trace with failure
        end_trace("risk", system=system, success=False, summary={"error": str(e)})
        return {"success": False, "error": f"Risk check failed: {e}"}


async def run_post_open_chase(system: str) -> Dict[str, Any]:
    """Execute post-market-open chase to cancel unfilled overnight limit orders and place market orders.

    Args:
        system: 'baseline' or 'internal'.
    """
    from src.audit import start_trace, end_trace
    from src.gcs.store import get_gcs_store
    from src.agents.execution_agent import ExecutionAgent

    # 1. Hydrate audit log from GCS to avoid overwriting history
    try:
        get_gcs_store().hydrate_audit_log()
    except Exception as e:
        logger.warning(f"GCS audit hydrate failed before chase: {e}")

    # 2. Start trace
    start_trace("chase", system=system)

    try:
        agent = ExecutionAgent(system=system)
        new_orders = await agent.chase_unfilled_orders(fill_threshold=0.70)
        
        # 3. End trace with success
        end_trace("chase", system=system, success=True, summary={"chased_orders": new_orders})
        
        return {
            "success": True,
            "system": system,
            "chased_orders": new_orders,
            "count": len(new_orders),
        }
    except Exception as e:
        # 3. End trace with failure
        end_trace("chase", system=system, success=False, summary={"error": str(e)})
        return {"success": False, "error": f"Post-open chase failed: {e}"}


def get_recent_news(tickers: str | None = None, max_articles_per_ticker: int = 3) -> Dict[str, Any]:
    """Fetch the latest news articles for specified tickers from Alpaca News API (primary) or Yahoo Finance (fallback).

    Args:
        tickers: Comma-separated list of symbols (e.g., 'AAPL,SPY'). Defaults to full universe.
        max_articles_per_ticker: Maximum number of news stories to return per ticker.
    """
    from datetime import datetime
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    symbols = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else list(config.TICKER_UNIVERSE)
    )

    news_by_ticker = {symbol: [] for symbol in symbols}

    # 1. Try Alpaca News API
    try:
        client = NewsClient(
            api_key=config.ALPACA_API_KEY_BASELINE,
            secret_key=config.ALPACA_SECRET_KEY_BASELINE,
        )
        symbols_str = ",".join(symbols)
        limit = len(symbols) * max_articles_per_ticker * 2
        req = NewsRequest(symbols=symbols_str, limit=limit)
        res = client.get_news(req)

        raw_articles = res.data.get("news", []) if hasattr(res, "data") else []
        for article in raw_articles:
            art_symbols = article.get("symbols", [])
            for sym in art_symbols:
                if sym in news_by_ticker and len(news_by_ticker[sym]) < max_articles_per_ticker:
                    created_at = article.get("created_at")
                    time_str = "Unknown"
                    if created_at:
                        if isinstance(created_at, datetime):
                            time_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            time_str = str(created_at)

                    news_by_ticker[sym].append({
                        "title": article.get("headline", "No Title"),
                        "publisher": article.get("source", "Unknown Source"),
                        "summary": article.get("summary", ""),
                        "link": article.get("url", ""),
                        "published_at": time_str,
                    })
    except Exception:
        pass

    # 2. Fallback to yfinance for any symbols that still have no news
    try:
        import yfinance as yf
    except ModuleNotFoundError:
        yf = None

    for symbol in symbols:
        if not news_by_ticker[symbol]:
            if yf is None:
                continue
            try:
                ticker_obj = yf.Ticker(symbol)
                raw_news = ticker_obj.news or []
                for item in raw_news[:max_articles_per_ticker]:
                    content = item.get("content", {})
                    title = content.get("title") or item.get("title", "No Title")
                    
                    provider = content.get("provider", {})
                    publisher = provider.get("displayName") or item.get("publisher", "Unknown Publisher")
                    
                    summary = content.get("summary") or content.get("description") or item.get("summary", "")
                    
                    canonical_url = content.get("canonicalUrl", {})
                    link = canonical_url.get("url") or item.get("link", "")
                    
                    pub_date = content.get("pubDate")
                    time_str = "Unknown"
                    if pub_date:
                        time_str = str(pub_date)
                    else:
                        pub_time = item.get("providerPublishTime")
                        if pub_time:
                            try:
                                time_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M:%S")
                            except Exception:
                                pass

                    news_by_ticker[symbol].append({
                        "title": title,
                        "publisher": publisher,
                        "summary": summary,
                        "link": link,
                        "published_at": time_str,
                    })
            except Exception as e:
                news_by_ticker[symbol] = [{"error": f"Failed to fetch news: {e}"}]

    return {
        "success": True,
        "news": news_by_ticker,
    }
