#!/usr/bin/env python3
"""MCP stdio server exposing Twin Ledger tools (Alpaca, MarketCrunch, DataBento).

Run: python -m src.adk.mcp.server

Used by ADK McpToolset for competition Track 1 (MCP external tool connections).
"""
from mcp.server.fastmcp import FastMCP

from src.adk.tools import alpaca_tools, competition_tools, databento_tools, marketcrunch_tools

mcp = FastMCP("twin-ledger-tools")


@mcp.tool()
def get_account_info(system: str = "baseline") -> dict:
    """Alpaca account snapshot for baseline or internal paper account."""
    return alpaca_tools.get_account_info(system=system)


@mcp.tool()
def get_open_positions(system: str = "baseline") -> dict:
    """Open positions for baseline or internal account."""
    return alpaca_tools.get_open_positions(system=system)


@mcp.tool()
def get_technical_indicators(system: str = "baseline", tickers: str = "", lookback_days: int = 90) -> dict:
    """OHLCV + RSI/MACD/Bollinger for comma-separated tickers (empty = full universe)."""
    return alpaca_tools.get_technical_indicators(
        system=system,
        tickers=tickers or None,
        lookback_days=lookback_days,
    )


@mcp.tool()
def get_leaderboard(system: str = "baseline") -> dict:
    """Twin Ledger competition context for the given system."""
    return competition_tools.get_leaderboard(system=system)


@mcp.tool()
def get_competition_status() -> dict:
    """Portfolio values and ranks for both competing agents."""
    return competition_tools.get_competition_status()


@mcp.tool()
def get_performance_metrics(hours: int = 720) -> dict:
    """Quant head-to-head metrics: excess return, Sharpe, drawdown, significance."""
    return competition_tools.get_performance_metrics(hours=hours)


@mcp.tool()
def get_marketcrunch_predictions(tickers: str = "") -> dict:
    """MarketCrunch AI estimates + Kelly weights (comma-separated tickers, empty = universe)."""
    return marketcrunch_tools.get_marketcrunch_predictions(tickers=tickers or None)


@mcp.tool()
def get_databento_features(tickers: str = "") -> dict:
    """Approved DataBento feature sources per ticker."""
    return databento_tools.get_databento_features(tickers=tickers or None)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
