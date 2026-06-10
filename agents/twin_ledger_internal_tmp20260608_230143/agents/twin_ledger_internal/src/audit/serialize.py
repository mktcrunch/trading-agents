"""Compact snapshots for audit payloads (JSON-safe, size-bounded)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def account_snapshot(account: Optional[Dict]) -> Dict[str, Any]:
    if not account:
        return {}
    keys = (
        "portfolio_value", "cash", "equity", "buying_power",
        "last_equity", "status", "daytrade_count",
    )
    return {k: account[k] for k in keys if account.get(k) is not None}


def positions_snapshot(positions: Dict) -> List[Dict[str, Any]]:
    rows = []
    for ticker, pos in (positions or {}).items():
        if not isinstance(pos, dict):
            continue
        rows.append({
            "ticker": ticker,
            "qty": pos.get("qty"),
            "market_value": pos.get("market_value"),
            "avg_fill_price": pos.get("avg_fill_price"),
            "current_price": pos.get("current_price"),
            "unrealized_pl": pos.get("unrealized_pl"),
            "unrealized_plpc": pos.get("unrealized_plpc"),
        })
    return rows


def mc_analysis_snapshot(analysis: Optional[Dict], ticker: Optional[str] = None) -> Dict[str, Any]:
    if not analysis:
        return {"ticker": ticker}
    ai = analysis.get("ai_estimate") or {}
    return {
        "ticker": ticker or analysis.get("ticker"),
        "confidence": ai.get("confidence"),
        "target_delta_numeric": ai.get("target_delta_numeric"),
        "target_price": ai.get("target_price"),
        "current_price": ai.get("current_price"),
    }


def discovery_snapshot(data: Optional[Dict]) -> Dict[str, Any]:
    if not data:
        return {}
    summary = data.get("summary") or {}
    return {
        "generated_at": data.get("generated_at"),
        "mode": data.get("mode"),
        "strategy_note": data.get("strategy_note"),
        "approved_count": summary.get("approved_count"),
        "tickers_with_features": summary.get("tickers_with_features"),
        "probes_run": summary.get("probes_run"),
        "error": summary.get("error"),
    }
