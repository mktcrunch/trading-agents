"""ADK tool registry for Twin Ledger."""
from google.adk.tools.function_tool import FunctionTool

from . import alpaca_tools, competition_tools, databento_tools, marketcrunch_tools


def _data_read_tools() -> list[FunctionTool]:
    """Read-only market/account tools for data and signal sub-agents."""
    return [
        FunctionTool(alpaca_tools.get_account_info),
        FunctionTool(alpaca_tools.get_open_positions),
        FunctionTool(alpaca_tools.get_technical_indicators),
        FunctionTool(alpaca_tools.get_latest_prices),
        FunctionTool(alpaca_tools.get_recent_news),
        FunctionTool(competition_tools.get_leaderboard),
        FunctionTool(competition_tools.get_competition_status),
    ]


def baseline_data_tools() -> list[FunctionTool]:
    """Tools for baseline data sub-agent (fetch only)."""
    return _data_read_tools()


def baseline_signal_tools() -> list[FunctionTool]:
    """Tools for baseline signal sub-agent (context fetch, no execution)."""
    return _data_read_tools()


def internal_data_tools() -> list[FunctionTool]:
    """Tools for internal data sub-agent."""
    return _data_read_tools() + [
        FunctionTool(marketcrunch_tools.get_marketcrunch_predictions),
        FunctionTool(databento_tools.get_databento_features),
    ]


def internal_signal_tools() -> list[FunctionTool]:
    """Read-only market tools; MC/DataBento are preloaded by the ADK workflow."""
    return _data_read_tools()


def coordinator_execution_tools() -> list[FunctionTool]:
    """Trading workflow tools for coordinators only (not sub-agents)."""
    return [
        FunctionTool(alpaca_tools.run_daily_trading_workflow),
        FunctionTool(alpaca_tools.execute_trading_decisions),
        FunctionTool(alpaca_tools.run_intraday_risk_check),
        FunctionTool(alpaca_tools.run_post_open_chase),
    ]


def all_function_tools() -> list[FunctionTool]:
    """All Twin Ledger tools (for MCP parity)."""
    return internal_data_tools() + coordinator_execution_tools()
