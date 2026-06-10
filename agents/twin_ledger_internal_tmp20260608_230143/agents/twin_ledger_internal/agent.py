"""ADK web / Agent Engine entry — Internal Twin Ledger."""
import sys
from pathlib import Path

_agent_dir = Path(__file__).resolve().parent
_repo_root = _agent_dir.parent.parent
for _p in (_repo_root, _agent_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.adk.agents.coordinators import build_internal_root_agent

root_agent = build_internal_root_agent()
