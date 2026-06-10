"""ADK tool registry for Twin Ledger."""
from google.adk.tools.function_tool import FunctionTool

from . import alpaca_tools, competition_tools, databento_tools, marketcrunch_tools


def baseline_data_tools() -> list[FunctionTool]:
    """Tools available to baseline data / signal agents."""
    return [
        FunctionTool(alpaca_tools.get_account_info),
        FunctionTool(alpaca_tools.get_open_positions),
        FunctionTool(alpaca_tools.get_technical_indicators),
        FunctionTool(alpaca_tools.get_latest_prices),
        FunctionTool(alpaca_tools.get_recent_news),
        FunctionTool(alpaca_tools.execute_trading_decisions),
        FunctionTool(alpaca_tools.run_intraday_risk_check),
        FunctionTool(alpaca_tools.run_post_open_chase),
        FunctionTool(competition_tools.get_leaderboard),
        FunctionTool(competition_tools.get_competition_status),
    ]


def internal_data_tools() -> list[FunctionTool]:
    """Tools for internal system (baseline tools + MC + DataBento)."""
    return baseline_data_tools() + [
        FunctionTool(marketcrunch_tools.get_marketcrunch_predictions),
        FunctionTool(databento_tools.get_databento_features),
    ]


def all_function_tools() -> list[FunctionTool]:
    """All Twin Ledger tools (for MCP parity)."""
    return internal_data_tools()
