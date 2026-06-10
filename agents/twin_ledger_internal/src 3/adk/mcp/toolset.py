"""Optional MCP toolset for ADK agents (stdio connection to twin-ledger-tools)."""
import sys
from typing import Optional

from mcp import StdioServerParameters

from src import config


def build_mcp_toolset():
    """Return McpToolset connected to local Twin Ledger MCP server."""
    from google.adk.tools.mcp_tool import McpToolset

    return McpToolset(
        connection_params=StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.adk.mcp.server"],
            env=None,
        ),
        tool_filter=[
            "get_account_info",
            "get_open_positions",
            "get_technical_indicators",
            "get_leaderboard",
            "get_competition_status",
            "get_marketcrunch_predictions",
            "get_databento_features",
        ],
    )


def optional_mcp_tools() -> list:
    """Attach MCP toolset when USE_ADK_MCP is enabled."""
    if not config.USE_ADK_MCP:
        return []
    return [build_mcp_toolset()]
