"""
Quant performance metrics for Twin Ledger head-to-head comparison.

Aligns daily Alpaca equity snapshots, computes risk-adjusted stats, and
estimates statistical significance for daily alpha, Sharpe spread, and drawdown spread.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

TRADING_DAYS_PER_YEAR = 252
MIN_OBS_FOR_STATS = 5
BOOTSTRAP_SAMPLES = 4000


def _date_key(ts: str) -> str:
    return (ts or "")[:10]


def _equity_by_date(points: List[Dict[str, Any]]) -> Dict[str, float]:
    by_date: Dict[str, float] = {}
    for pt in points or []:
        ts = pt.get("timestamp")
        pv = pt.get("portfolio_value")
        if not ts or pv is None:
            continue
        try:
            pv_f = float(pv)
        except (TypeError, ValueError):
            continue
        if pv_f <= 0:
            continue
        dk = _date_key(str(ts))
        if dk:
            by_date[dk] = pv_f
    return by_date


def _align_returns(
    baseline_pts: List[Dict[str, Any]],
    internal_pts: List[Dict[str, Any]],
) -> Tuple[List[str], List[float], List[float]]:
    b_eq = _equity_by_date(baseline_pts)
    i_eq = _equity_by_date(internal_pts)
    common = sorted(set(b_eq) & set(i_eq))
    if len(common) < 2:
        return [], [], []

    dates: List[str] = []
    b_rets: List[float] = []
    i_rets: List[float] = []
    for j in range(1, len(common)):
        d0, d1 = common[j - 1], common[j]
        pb, cb = b_eq[d0], b_eq[d1]
        pi, ci = i_eq[d0], i_eq[d1]
        if pb <= 0 or pi <= 0:
            continue
        dates.append(d1)
        b_rets.append((cb - pb) / pb)
        i_rets.append((ci - pi) / pi)
    return dates, b_rets, i_rets


def _sharpe(returns: List[float], annualize: bool = True) -> Optional[float]:
    if len(returns) < 2:
        return None
    arr = np.asarray(returns, dtype=float)
    std = float(arr.std(ddof=1))
    if std == 0 or math.isnan(std):
        return None
    sr = float(arr.mean() / std)
    if annualize:
        sr *= math.sqrt(TRADING_DAYS_PER_YEAR)
    return round(sr, 3)


def _max_drawdown(equity_by_date: Dict[str, float]) -> Dict[str, Optional[float]]:
    dates = sorted(equity_by_date.keys())
    if not dates:
        return {"max_drawdown_pct": None, "current_drawdown_pct": None}

    peak = equity_by_date[dates[0]]
    max_dd = 0.0
    cur_dd = 0.0
    for d in dates:
        v = equity_by_date[d]
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        cur_dd = dd
    return {
        "max_drawdown_pct": round(max_dd * 100, 3),
        "current_drawdown_pct": round(cur_dd * 100, 3),
    }


def _volatility_ann_pct(returns: List[float]) -> Optional[float]:
    if len(returns) < 2:
        return None
    std = float(np.std(returns, ddof=1))
    return round(std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100, 3)


def _total_return_pct(equity_by_date: Dict[str, float], starting_equity: float) -> Optional[float]:
    if not equity_by_date or starting_equity <= 0:
        return None
    dates = sorted(equity_by_date.keys())
    last = equity_by_date[dates[-1]]
    return round((last - starting_equity) / starting_equity * 100, 3)


def _significance_ttest(diff: List[float]) -> Dict[str, Any]:
    n = len(diff)
    if n < MIN_OBS_FOR_STATS:
        result = {
            "p_value": None,
            "significant_95": None,
            "test": "paired_t",
            "n": n,
            "insufficient_data": True,
        }
        return _enrich_significance(result, diff=diff)

    _, p = stats.ttest_1samp(diff, popmean=0.0, alternative="two-sided")
    p_f = float(p)
    result = {
        "p_value": round(p_f, 4),
        "significant_95": p_f < 0.05,
        "test": "paired_t",
        "n": n,
        "insufficient_data": False,
    }
    return _enrich_significance(result, diff=diff)


def _days_for_paired_mean(
    diff: List[float],
    alpha: float = 0.05,
    power: float = 0.80,
) -> Tuple[Optional[int], Optional[int]]:
    """Paired-sample days required to detect the observed mean excess (80% power)."""
    n = len(diff)
    if n < 2:
        return None, None
    arr = np.asarray(diff, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std == 0 or abs(mean) < 1e-15:
        return None, None
    effect = abs(mean) / std
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_beta = float(stats.norm.ppf(power))
    n_req = int(np.ceil(((z_alpha + z_beta) / effect) ** 2))
    n_req = max(n_req, MIN_OBS_FOR_STATS)
    return n_req, max(0, n_req - n)


def _days_from_pvalue(
    n: int,
    p_value: Optional[float],
    alpha: float = 0.05,
) -> Tuple[Optional[int], Optional[int]]:
    """Project paired days from current p-value (sqrt-n evidence scaling)."""
    if p_value is None or p_value <= alpha or n < MIN_OBS_FOR_STATS:
        return None, None
    if p_value >= 1.0:
        return None, None
    t_current = abs(float(stats.norm.ppf(p_value / 2)))
    t_need = float(stats.norm.ppf(1 - alpha / 2))
    if t_current < 1e-6:
        return None, None
    n_req = int(np.ceil(n * (t_need / t_current) ** 2))
    n_req = max(n_req, MIN_OBS_FOR_STATS)
    return n_req, max(0, n_req - n)


def _enrich_significance(
    sig: Dict[str, Any],
    *,
    diff: Optional[List[float]] = None,
) -> Dict[str, Any]:
    out = dict(sig)
    n = int(out.get("n") or 0)
    p = out.get("p_value")

    if out.get("significant_95"):
        out["days_required_95"] = n
        out["days_remaining_95"] = 0
        return out

    n_req: Optional[int] = None
    remaining: Optional[int] = None

    if out.get("test") == "paired_t" and diff is not None and len(diff) >= 2:
        n_req, remaining = _days_for_paired_mean(diff)
    elif not out.get("insufficient_data") and p is not None:
        n_req, remaining = _days_from_pvalue(n, float(p))
    elif diff is not None and len(diff) >= 2:
        n_req, remaining = _days_for_paired_mean(diff)

    if n_req is not None:
        out["days_required_95"] = n_req
        out["days_remaining_95"] = remaining
    return out


def _bootstrap_pvalue(observed: float, samples: List[float]) -> float:
    if not samples:
        return 1.0
    arr = np.abs(np.asarray(samples, dtype=float))
    return float(np.mean(arr >= abs(observed)))


def _bootstrap_sharpe_diff(b_rets: List[float], i_rets: List[float], seed: int = 42) -> Dict[str, Any]:
    n = len(b_rets)
    if n < MIN_OBS_FOR_STATS:
        return _enrich_significance({
            "p_value": None,
            "significant_95": None,
            "test": "bootstrap_paired",
            "n": n,
            "insufficient_data": True,
        }, diff=[i - b for b, i in zip(b_rets, i_rets)] if n else None)

    obs_b = _sharpe(b_rets)
    obs_i = _sharpe(i_rets)
    if obs_b is None or obs_i is None:
        return _enrich_significance({
            "p_value": None,
            "significant_95": None,
            "test": "bootstrap_paired",
            "n": n,
            "insufficient_data": True,
        }, diff=[i - b for b, i in zip(b_rets, i_rets)])

    observed = obs_i - obs_b
    rng = np.random.default_rng(seed)
    b_arr = np.asarray(b_rets, dtype=float)
    i_arr = np.asarray(i_rets, dtype=float)
    boot_diffs: List[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, size=n)
        sb = _sharpe(b_arr[idx].tolist())
        si = _sharpe(i_arr[idx].tolist())
        if sb is not None and si is not None:
            boot_diffs.append(si - sb)

    if len(boot_diffs) < 100:
        return _enrich_significance({
            "p_value": None,
            "significant_95": None,
            "test": "bootstrap_paired",
            "n": n,
            "insufficient_data": True,
        }, diff=(i_arr - b_arr).tolist())

    p = _bootstrap_pvalue(observed, boot_diffs)
    result = {
        "p_value": round(p, 4),
        "significant_95": p < 0.05,
        "test": "bootstrap_paired",
        "n": n,
        "insufficient_data": False,
    }
    excess = (i_arr - b_arr).tolist()
    return _enrich_significance(result, diff=excess)


def _max_dd_from_path(path: List[float]) -> float:
    peak = path[0]
    max_dd = 0.0
    for v in path:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _bootstrap_max_dd_diff(b_rets: List[float], i_rets: List[float], seed: int = 42) -> Dict[str, Any]:
    n = len(b_rets)
    excess = [i - b for b, i in zip(b_rets, i_rets)]
    if n < MIN_OBS_FOR_STATS:
        return _enrich_significance({
            "p_value": None,
            "significant_95": None,
            "test": "bootstrap_paired",
            "n": n,
            "insufficient_data": True,
        }, diff=excess)

    def path_from_rets(rets: List[float], indices: np.ndarray) -> List[float]:
        eq = 1.0
        path = [eq]
        for idx in indices:
            eq *= 1.0 + rets[int(idx)]
            path.append(eq)
        return path

    b_path = path_from_rets(b_rets, np.arange(n))
    i_path = path_from_rets(i_rets, np.arange(n))
    observed = (_max_dd_from_path(i_path) - _max_dd_from_path(b_path)) * 100

    rng = np.random.default_rng(seed)
    boot_obs: List[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, size=n)
        b_boot = path_from_rets(b_rets, idx)
        i_boot = path_from_rets(i_rets, idx)
        boot_obs.append((_max_dd_from_path(i_boot) - _max_dd_from_path(b_boot)) * 100)

    p = _bootstrap_pvalue(observed, boot_obs)
    return _enrich_significance({
        "p_value": round(p, 4),
        "significant_95": p < 0.05,
        "test": "bootstrap_paired",
        "n": n,
        "insufficient_data": False,
    }, diff=excess)


def _bootstrap_total_return_diff(b_rets: List[float], i_rets: List[float], seed: int = 42) -> Dict[str, Any]:
    """Bootstrap paired cumulative return spread (Internal − Baseline) over aligned days."""
    n = len(b_rets)
    excess = [i - b for b, i in zip(b_rets, i_rets)]
    if n < MIN_OBS_FOR_STATS:
        return _enrich_significance({
            "p_value": None,
            "significant_95": None,
            "test": "bootstrap_paired",
            "n": n,
            "insufficient_data": True,
        }, diff=excess)

    observed = _terminal_return_pct(i_rets) - _terminal_return_pct(b_rets)
    rng = np.random.default_rng(seed)
    b_arr = np.asarray(b_rets, dtype=float)
    i_arr = np.asarray(i_rets, dtype=float)
    boot_obs: List[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, size=n)
        boot_obs.append(
            _terminal_return_pct(i_arr[idx].tolist()) - _terminal_return_pct(b_arr[idx].tolist())
        )

    p = _bootstrap_pvalue(observed, boot_obs)
    return _enrich_significance({
        "p_value": round(p, 4),
        "significant_95": p < 0.05,
        "test": "bootstrap_paired",
        "n": n,
        "insufficient_data": False,
    }, diff=excess)


def _terminal_return_pct(rets: List[float]) -> float:
    wealth = 1.0
    for r in rets:
        wealth *= 1.0 + r
    return (wealth - 1.0) * 100


def _agent_metrics(
    returns: List[float],
    equity_by_date: Dict[str, float],
    starting_equity: float,
    latest_daily_return_pct: Optional[float],
) -> Dict[str, Any]:
    dd = _max_drawdown(equity_by_date)
    return {
        "daily_return_pct": latest_daily_return_pct,
        "sharpe": _sharpe(returns),
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "current_drawdown_pct": dd["current_drawdown_pct"],
        "volatility_ann_pct": _volatility_ann_pct(returns),
        "total_return_pct": _total_return_pct(equity_by_date, starting_equity),
        "observation_days": len(returns),
    }


def compute_head_to_head_metrics(
    baseline_history: List[Dict[str, Any]],
    internal_history: List[Dict[str, Any]],
    starting_equity: float = 100_000.0,
) -> Dict[str, Any]:
    """Compute aligned quant metrics and head-to-head significance tests."""
    dates, b_rets, i_rets = _align_returns(baseline_history, internal_history)
    b_eq = _equity_by_date(baseline_history)
    i_eq = _equity_by_date(internal_history)

    latest_b = round(b_rets[-1] * 100, 3) if b_rets else None
    latest_i = round(i_rets[-1] * 100, 3) if i_rets else None
    daily_delta = (
        round(latest_i - latest_b, 3)
        if latest_b is not None and latest_i is not None
        else None
    )

    excess = [i - b for b, i in zip(b_rets, i_rets)]
    mean_alpha = round(float(np.mean(excess)) * 100, 4) if excess else None
    cum_alpha = (
        round((float(np.prod([1.0 + e for e in excess])) - 1.0) * 100, 3)
        if excess
        else None
    )

    b_sharpe = _sharpe(b_rets)
    i_sharpe = _sharpe(i_rets)
    sharpe_diff = (
        round(i_sharpe - b_sharpe, 3)
        if b_sharpe is not None and i_sharpe is not None
        else None
    )

    b_dd = _max_drawdown(b_eq)
    i_dd = _max_drawdown(i_eq)
    dd_diff = None
    if b_dd["max_drawdown_pct"] is not None and i_dd["max_drawdown_pct"] is not None:
        dd_diff = round(i_dd["max_drawdown_pct"] - b_dd["max_drawdown_pct"], 3)

    b_total = _agent_metrics(b_rets, b_eq, starting_equity, latest_b)
    i_total = _agent_metrics(i_rets, i_eq, starting_equity, latest_i)
    total_return_diff = None
    if b_total["total_return_pct"] is not None and i_total["total_return_pct"] is not None:
        total_return_diff = round(i_total["total_return_pct"] - b_total["total_return_pct"], 3)

    return {
        "observation_days": len(b_rets),
        "min_days_for_stats": MIN_OBS_FOR_STATS,
        "latest_date": dates[-1] if dates else None,
        "baseline": b_total,
        "internal": i_total,
        "comparison": {
            "daily_delta_pct": daily_delta,
            "mean_daily_alpha_pct": mean_alpha,
            "cumulative_alpha_pct": cum_alpha,
            "total_return_diff_pct": total_return_diff,
            "sharpe_diff": sharpe_diff,
            "max_drawdown_diff_pct": dd_diff,
            "significance": {
                "total_return_diff": _bootstrap_total_return_diff(b_rets, i_rets),
                "daily_alpha": _significance_ttest(excess),
                "sharpe_diff": _bootstrap_sharpe_diff(b_rets, i_rets),
                "max_drawdown_diff": _bootstrap_max_dd_diff(b_rets, i_rets),
            },
        },
    }
