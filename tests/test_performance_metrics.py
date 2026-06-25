"""Tests for Twin Ledger quant performance metrics."""
from src.analytics.performance_metrics import compute_head_to_head_metrics


def _series(values):
    return [
        {"timestamp": f"2026-06-{10 + i:02d}T21:00:00+00:00", "portfolio_value": v}
        for i, v in enumerate(values)
    ]


def test_head_to_head_metrics_aligns_and_computes_sharpe():
    baseline = _series([100_000, 100_100, 100_050, 100_200, 100_150, 100_300])
    internal = _series([100_000, 100_200, 100_180, 100_400, 100_350, 100_500])
    out = compute_head_to_head_metrics(baseline, internal)

    assert out["observation_days"] == 5
    assert out["baseline"]["sharpe"] is not None
    assert out["internal"]["sharpe"] is not None
    assert out["comparison"]["daily_delta_pct"] is not None
    assert out["comparison"]["sign_convention"] == "internal_minus_baseline"
    assert "internal_minus_baseline" in out["comparison"]
    assert out["comparison"]["mean_daily_alpha_pct"] is not None
    assert out["baseline"]["max_drawdown_pct"] is not None
    assert out["internal"]["max_drawdown_pct"] is not None
    assert "significance" in out["comparison"]


def test_insufficient_data_marks_significance_na():
    baseline = _series([100_000, 100_100])
    internal = _series([100_000, 100_150])
    out = compute_head_to_head_metrics(baseline, internal)

    assert out["observation_days"] == 1
    sig = out["comparison"]["significance"]["daily_alpha"]
    assert sig["insufficient_data"] is True
    assert sig["p_value"] is None


def test_total_return_significance_bootstrap():
    baseline = _series([100_000, 99_900, 99_800, 99_700, 99_600, 99_500])
    internal = _series([100_000, 100_050, 100_100, 100_150, 100_200, 100_250])
    out = compute_head_to_head_metrics(baseline, internal)

    assert out["comparison"]["total_return_diff_pct"] is not None
    sig = out["comparison"]["significance"]["total_return_diff"]
    assert sig["test"] == "bootstrap_paired"
    assert sig["insufficient_data"] is False
    assert sig["p_value"] is not None
    assert sig.get("days_required_95") is not None


def test_days_to_significance_projected():
    baseline = _series([100_000, 99_900, 99_800, 99_700, 99_600, 99_500, 99_400, 99_300])
    internal = _series([100_000, 100_050, 100_100, 100_150, 100_200, 100_250, 100_300, 100_350])
    out = compute_head_to_head_metrics(baseline, internal)
    sig = out["comparison"]["significance"]["daily_alpha"]
    assert sig["days_required_95"] >= 5
    if not sig["significant_95"]:
        assert sig.get("days_remaining_95", 0) >= 0


def test_zero_effect_marks_significance_na():
    values = [100_000, 100_100, 100_200, 100_150, 100_300, 100_250]
    baseline = _series(values)
    internal = _series(values)
    out = compute_head_to_head_metrics(baseline, internal)
    sig = out["comparison"]["significance"]["daily_alpha"]
    assert sig.get("zero_effect") is True
    assert sig.get("days_remaining_95") is None


def test_internal_outperforms_has_positive_alpha():
    baseline = _series([100_000, 99_900, 99_800, 99_700, 99_600, 99_500])
    internal = _series([100_000, 100_050, 100_100, 100_150, 100_200, 100_250])
    out = compute_head_to_head_metrics(baseline, internal)

    assert out["comparison"]["mean_daily_alpha_pct"] > 0
    assert out["comparison"]["sharpe_diff"] is not None
