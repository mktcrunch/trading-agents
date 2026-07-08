"""Tests for Twin Ledger quant performance metrics."""
from src.analytics.performance_metrics import (
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    _annualized_cumulative_return_pct,
    _beta_vs_market,
    _sharpe,
    _spy_benchmark_from_closes,
    append_live_equity_points,
    attach_spy_benchmark,
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
        live_accounts={
            "baseline": {"equity": 100_700, "daily_return_pct": 0.5},
            "internal": {"equity": 101_800, "daily_return_pct": 1.2},
        },
    )
    assert out["comparison"]["daily_delta_basis"] == "live"
    assert out["comparison"]["daily_delta_pct"] == 0.7
    assert out["baseline"]["daily_return_pct"] == 0.5
    assert out["internal"]["daily_return_pct"] == 1.2
    assert out["comparison"]["series_basis"] == "paired_plus_live_latest"
    assert out["comparison"]["total_return_diff_pct"] == 1.1
    assert out["baseline"]["total_return_pct"] == 0.7
    assert out["internal"]["total_return_pct"] == 1.8
    assert out["comparison"]["sharpe_diff"] is not None
    assert out["comparison"]["max_drawdown_diff_pct"] is not None
    assert out["comparison"]["paired_close"]["total_return_diff_pct"] == 0.1
    assert out["paired_observation_days"] == 2
    assert out["observation_days"] == 3
    assert out["comparison"]["significance"]["daily_alpha"]["n"] == 3
    assert out["comparison"]["significance"]["total_return_diff"]["n"] == 3
    assert out["comparison"]["significance"]["sharpe_diff"]["n"] == 3
    assert out["comparison"]["significance"]["max_drawdown_diff"]["n"] == 3


def test_live_daily_return_from_account():
    from src.analytics.performance_metrics import live_daily_return_pct

    history = _series([100_000, 100_100])
    ret = live_daily_return_pct(
        history,
        {"equity": 100_250, "last_equity": 100_200},
    )
    assert ret == round((100_250 - 100_200) / 100_200 * 100, 3)


def test_spy_benchmark_from_closes_compound_annualizes():
    closes = [
        ("2026-06-09", 100.0),
        ("2026-06-10", 101.0),
        ("2026-06-11", 102.0),
        ("2026-06-12", 103.0),
    ]
    spy = _spy_benchmark_from_closes(
        closes,
        start_date="2026-06-09",
        start_label="June 9, 2026",
    )
    assert spy is not None
    assert spy["source"] == "alpaca"
    assert spy["total_return_pct"] == 3.0
    assert spy["observation_days"] == 3
    assert spy["annualized_return_pct"] == _annualized_cumulative_return_pct(3.0, 3)
    assert spy["mean_daily_return_pct"] is not None
    assert spy["annualized_mean_return_pct"] == round(spy["mean_daily_return_pct"] * 252, 3)
    assert spy["sharpe"] is not None
    assert spy["max_drawdown_pct"] is not None
    assert spy["start_date"] == "2026-06-09"


def test_spy_benchmark_includes_live_latest_point():
    closes = [
        ("2026-06-09", 100.0),
        ("2026-06-10", 101.0),
        ("2026-06-11", 102.0),
    ]
    spy = _spy_benchmark_from_closes(
        closes,
        start_date="2026-06-09",
        start_label="June 9, 2026",
        live_price=103.5,
    )
    assert spy is not None
    assert spy["live_included"] is True
    assert spy["display_end_date"] == "LIVE"
    assert spy["live_price"] == 103.5
    assert spy["live_daily_return_pct"] == round((103.5 - 102.0) / 102.0 * 100, 3)
    assert spy["observation_days"] == 3
    assert spy["annualized_return_pct"] == _annualized_cumulative_return_pct(
        spy["total_return_pct"], 3
    )


def test_beta_vs_market_unit_when_identical():
    rets = {"2026-06-10": 0.01, "2026-06-11": -0.005, "2026-06-12": 0.02}
    assert _beta_vs_market(rets, rets) == 1.0


def test_attach_spy_benchmark_sets_beta(monkeypatch):
    baseline = _series([100_000, 100_100, 100_050, 100_200, 100_150, 100_300])
    internal = _series([100_000, 100_200, 100_180, 100_400, 100_350, 100_500])
    metrics = compute_head_to_head_metrics(baseline, internal)

    closes = [
        ("2026-06-10", 100.0),
        ("2026-06-11", 100.5),
        ("2026-06-12", 100.2),
        ("2026-06-13", 101.0),
        ("2026-06-14", 100.8),
        ("2026-06-15", 101.5),
    ]

    def _fake_closes(start_date=None):
        return "2026-06-10", "June 10, 2026", closes

    monkeypatch.setattr(
        "src.analytics.performance_metrics._fetch_spy_closes",
        _fake_closes,
    )
    monkeypatch.setattr(
        "src.analytics.performance_metrics._fetch_spy_live_price",
        lambda: 102.0,
    )
    out = attach_spy_benchmark(
        metrics,
        baseline_history=baseline,
        internal_history=internal,
    )
    assert out["benchmark"]["spy"]["source"] == "alpaca"
    assert out["benchmark"]["spy"]["live_included"] is True
    assert out["baseline"]["beta_spy"] is not None
    assert out["internal"]["beta_spy"] is not None


def test_append_live_equity_points_extends_close_history():
    history = {
        "baseline": _series([100_000, 100_500]),
        "internal": _series([100_000, 101_000]),
    }
    out, basis = append_live_equity_points(
        history,
        {"baseline": {"equity": 100_750}, "internal": {"equity": 101_200}},
        starting_equity=100_000,
    )
    assert basis == "closes_plus_live_latest"
    assert len(out["baseline"]) == 3
    assert len(out["internal"]) == 3
    assert out["baseline"][-1]["source"] == "live"
    assert out["baseline"][-1]["portfolio_value"] == 100_750
    assert out["internal"][-1]["portfolio_value"] == 101_200
    assert history["baseline"][-1]["portfolio_value"] == 100_500


def test_append_live_equity_points_replaces_live_tail():
    history = {
        "baseline": _series([100_000, 100_500])
        + [{"timestamp": "2026-07-08T01:00:00+00:00", "portfolio_value": 100_600, "source": "live"}],
        "internal": _series([100_000, 101_000]),
    }
    out, basis = append_live_equity_points(
        history,
        {"baseline": {"equity": 100_800}, "internal": {"equity": 101_200}},
        starting_equity=100_000,
    )
    assert basis == "closes_plus_live_latest"
    assert len(out["baseline"]) == 3
    assert out["baseline"][-1]["portfolio_value"] == 100_800
    assert out["baseline"][-1]["source"] == "live"


def test_append_live_equity_points_without_live_accounts():
    history = {"baseline": _series([100_000]), "internal": _series([100_000])}
    out, basis = append_live_equity_points(history, None)
    assert basis == "closes_only"
    assert out == history
