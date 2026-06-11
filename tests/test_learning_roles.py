"""Learning role coverage."""
from src.learning.reflection import _prior_memory_for_prompt
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


def test_prior_memory_omitted_when_never_saved():
    assert _prior_memory_for_prompt({}) == {}


def test_prior_memory_includes_lessons_for_prompt():
    prior = {
        "updated_at": "2026-06-10T12:00:00+00:00",
        "lessons_learned": "SLV underperformed.",
        "bad_patterns": ["duplicate signals"],
        "do_more": ["favor EFA"],
        "scorecard": {"wins": 3, "losses": 2},
        "recent_decisions": [{"ticker": "SPY"}],
    }
    compact = _prior_memory_for_prompt(prior)
    assert compact["lessons_learned"] == "SLV underperformed."
    assert "recent_decisions" not in compact
