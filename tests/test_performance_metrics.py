"""Tests for Twin Ledger quant performance metrics."""
from src.analytics.performance_metrics import (
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    _annualized_cumulative_return_pct,
    _sharpe,
    compute_head_to_head_metrics,
)


def _series(values):
    return [
        {"timestamp": f"2026-06-{10 + i:02d}T21:00:00+00:00", "portfolio_value": v}
        for i, v in enumerate(values)
    ]


def test_sharpe_uses_risk_free_rate():
    # Constant positive daily returns still have zero excess if equal to rf.
    rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
    assert _sharpe([rf_daily, rf_daily, rf_daily, rf_daily]) is None  # zero std
    # Mean above rf → positive Sharpe; below rf → negative.
    above = [rf_daily + 0.001] * 5 + [rf_daily + 0.002] * 5
    below = [rf_daily - 0.001] * 5 + [rf_daily - 0.002] * 5
    assert _sharpe(above) > 0
    assert _sharpe(below) < 0
    assert RISK_FREE_RATE_ANNUAL == 0.0425


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
    assert out["comparison"]["annualized_alpha_pct"] is not None
    assert out["comparison"]["risk_free_rate_annual"] == 0.0425
    assert out["baseline"]["mean_daily_return_pct"] is not None
    assert out["baseline"]["annualized_return_pct"] is not None
    assert out["baseline"]["annualized_cumulative_return_pct"] is not None
    assert out["internal"]["mean_daily_return_pct"] is not None
    assert out["internal"]["annualized_return_pct"] is not None
    assert out["internal"]["annualized_cumulative_return_pct"] is not None
    # Annualized alpha = mean daily alpha × 252 (rounded to 3 dp)
    assert out["comparison"]["annualized_alpha_pct"] == round(
        out["comparison"]["mean_daily_alpha_pct"] * 252, 3
    )
    # Compound-annualized excess ≠ mean-daily × 252 in general
    n = out["observation_days"]
    assert out["comparison"]["annualized_excess_return_pct"] == (
        _annualized_cumulative_return_pct(
            out["comparison"]["total_return_diff_pct"], n
        )
    )
    assert (
        out["comparison"]["annualized_excess_return_pct"]
        != out["comparison"]["annualized_alpha_pct"]
    )
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


def test_path_comparison_win_rates():
    baseline = _series([100_000, 100_100, 100_050, 100_200, 100_150, 100_300])
    internal = _series([100_000, 100_200, 100_180, 100_400, 100_350, 100_500])
    out = compute_head_to_head_metrics(baseline, internal)
    path = out["path_comparison"]

    assert path["daily_win_rate"]["internal_wins"] > 0
    assert path["days_equity_ahead"]["internal_wins"] > 0
    assert path["daily_win_rate"]["significance"]["test"] == "binomial"
    assert path["days_equity_ahead"]["rate_pct"] is not None


def test_live_daily_return_uses_account_equity():
    baseline = _series([100_000, 100_100, 100_200])
    internal = _series([100_000, 100_150, 100_300])
    out = compute_head_to_head_metrics(
        baseline,
        internal,
        live_daily_returns={"baseline": 0.5, "internal": 1.2},
    )
    assert out["comparison"]["daily_delta_basis"] == "live"
    assert out["comparison"]["daily_delta_pct"] == 0.7
    assert out["baseline"]["daily_return_pct"] == 0.5
    assert out["internal"]["daily_return_pct"] == 1.2


def test_live_daily_return_from_account():
    from src.analytics.performance_metrics import live_daily_return_pct

    history = _series([100_000, 100_100])
    ret = live_daily_return_pct(
        history,
        {"equity": 100_250, "last_equity": 100_200},
    )
    assert ret == round((100_250 - 100_200) / 100_200 * 100, 3)
