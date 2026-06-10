"""Learning roles for Twin Ledger specialist crew."""
from __future__ import annotations

from typing import List

TRADER_LEARNING_ROLES: List[str] = [
    "coordinator",
    "data",
    "signal",
    "risk",
    "execution",
    "monitor",
]

INTERNAL_ONLY_LEARNING_ROLES: List[str] = ["discovery"]

LLM_REFLECTION_ROLES = frozenset({"signal", "risk"})


def roles_for_system(system: str) -> List[str]:
    roles = list(TRADER_LEARNING_ROLES)
    if system == "internal":
        roles.extend(INTERNAL_ONLY_LEARNING_ROLES)
    return roles
