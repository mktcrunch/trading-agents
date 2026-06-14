"""ADK web / Agent Engine entry — Internal Twin Ledger."""
import sys
from pathlib import Path

_agent_dir = Path(__file__).resolve().parent
_repo_root = _agent_dir.parent.parent
for _p in (_repo_root, _agent_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.adk.agent_engine_app import build_scheduler_direct_app, install_agent_engine_stream_query_patch
from src.adk.agents.coordinators import build_internal_root_agent

root_agent = build_internal_root_agent()
install_agent_engine_stream_query_patch("internal")
app = build_scheduler_direct_app(root_agent, system="internal")
