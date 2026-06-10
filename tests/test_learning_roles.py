"""Learning role coverage."""
from src.learning.roles import roles_for_system, LLM_REFLECTION_ROLES


def test_baseline_has_six_learning_roles():
    roles = roles_for_system("baseline")
    assert roles == [
        "coordinator", "data", "signal", "risk", "execution", "monitor",
    ]


def test_internal_adds_discovery():
    roles = roles_for_system("internal")
    assert "discovery" in roles
    assert len(roles) == 7


def test_llm_reflection_limited_to_signal_and_risk():
    assert LLM_REFLECTION_ROLES == frozenset({"signal", "risk"})
