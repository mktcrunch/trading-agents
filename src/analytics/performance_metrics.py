"""
Quant performance metrics for Twin Ledger head-to-head comparison.

Aligns daily Alpaca equity snapshots, computes risk-adjusted stats, and
estimates statistical significance for daily alpha, Sharpe spread, and drawdown spread.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
# Annual risk-free rate used in Sharpe (excess return over cash).
RISK_FREE_RATE_ANNUAL = 0.0425
MIN_OBS_FOR_STATS = 5
BOOTSTRAP_SAMPLES = 4000
SIGN_CONVENTION = "internal_minus_baseline"
COMPARISON_FORMULA = "internal_value - baseline_value"


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
    """Annualized Sharpe using excess returns over RISK_FREE_RATE_ANNUAL."""
    if len(returns) < 2:
        return None
    arr = np.asarray(returns, dtype=float)
    std = float(arr.std(ddof=1))
    if std == 0 or math.isnan(std):
        return None
    rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
    sr = float((arr.mean() - rf_daily) / std)
    if annualize:
        sr *= math.sqrt(TRADING_DAYS_PER_YEAR)
    return round(sr, 3)


def _mean_daily_return_pct(returns: List[float]) -> Optional[float]:
    if not returns:
        return None
    return round(float(np.mean(returns)) * 100, 4)


def _annualized_return_pct(mean_daily_return_pct: Optional[float]) -> Optional[float]:
    """Simple annualization: mean daily % × 252 trading days."""
    if mean_daily_return_pct is None:
        return None
    return round(mean_daily_return_pct * TRADING_DAYS_PER_YEAR, 3)


def _annualized_cumulative_return_pct(
    total_return_pct: Optional[float],
    observation_days: int,
) -> Optional[float]:
    """Compound-annualize a realized cumulative return over observation_days.

    (1 + r)^(252 / n) − 1, where r is total return as a decimal and n is paired days.
    Distinct from mean-daily × 252 (run-rate annualization on the Daily delta card).
    """
    if total_return_pct is None or observation_days < 1:
        return None
    r = total_return_pct / 100.0
    if r <= -1.0:
        return None
    ann = (1.0 + r) ** (TRADING_DAYS_PER_YEAR / observation_days) - 1.0
    return round(ann * 100, 3)


def _current_equity_from_account(
    account: Optional[Dict[str, Any]],
) -> Optional[float]:
    if not account:
        return None
    equity = account.get("equity")
    if equity is None:
        equity = account.get("portfolio_value")
    try:
        equity_f = float(equity)
    except (TypeError, ValueError):
        return None
    return equity_f if equity_f > 0 else None


def _daily_returns_by_date(closes: List[Tuple[str, float]]) -> Dict[str, float]:
    """Map end-date → simple close-to-close return from ordered (date, close) pairs."""
    out: Dict[str, float] = {}
    for i in range(1, len(closes)):
        _d0, c0 = closes[i - 1]
        d1, c1 = closes[i]
        if c0 > 0:
            out[d1] = (c1 - c0) / c0
    return out


def _portfolio_returns_by_date(history: List[Dict[str, Any]]) -> Dict[str, float]:
    eq = _equity_by_date(history)
    dates = sorted(eq.keys())
    out: Dict[str, float] = {}
    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]
        p0, p1 = eq[d0], eq[d1]
        if p0 > 0:
            out[d1] = (p1 - p0) / p0
    return out


def _beta_vs_market(
    portfolio_returns_by_date: Dict[str, float],
    market_returns_by_date: Dict[str, float],
) -> Optional[float]:
    """OLS beta: Cov(r_p, r_m) / Var(r_m) on aligned daily returns."""
    common = sorted(set(portfolio_returns_by_date) & set(market_returns_by_date))
    if len(common) < 2:
        return None
    p = np.asarray([portfolio_returns_by_date[d] for d in common], dtype=float)
    m = np.asarray([market_returns_by_date[d] for d in common], dtype=float)
    var_m = float(m.var(ddof=1))
    if var_m == 0 or math.isnan(var_m):
        return None
    cov = float(np.cov(p, m, ddof=1)[0, 1])
    return round(cov / var_m, 3)


def _spy_benchmark_from_closes(
    closes: List[Tuple[str, float]],
    *,
    start_date: str,
    start_label: str,
    live_price: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Build SPY benchmark stats from (date, close) pairs on/after start_date."""
    points = [(d, c) for d, c in closes if d >= start_date and c > 0]
    if len(points) < 2:
        return None
    live_daily_return_pct: Optional[float] = None
    live_included = False
    display_points = list(points)
    if live_price is not None and live_price > 0:
        last_close = points[-1][1]
        if last_close > 0:
            live_daily_return_pct = round((live_price - last_close) / last_close * 100, 3)
            display_points = points + [("LIVE", live_price)]
            live_included = True
    start_close = points[0][1]
    end_close = display_points[-1][1]
    if start_close <= 0:
        return None
    total_return_pct = round((end_close / start_close - 1.0) * 100, 3)
    observation_days = len(display_points) - 1
    spy_rets: List[float] = []
    for i in range(1, len(display_points)):
        c0 = display_points[i - 1][1]
        c1 = display_points[i][1]
        if c0 > 0:
            spy_rets.append((c1 - c0) / c0)
    spy_eq = {d: c for d, c in display_points}
    mean_daily = _mean_daily_return_pct(spy_rets)
    dd = _max_drawdown(spy_eq)
    return {
        "ticker": "SPY",
        "source": "alpaca",
        "start_date": start_date,
        "start_label": start_label,
        "end_date": points[-1][0],
        "display_end_date": "LIVE" if live_included else points[-1][0],
        "start_close": round(start_close, 4),
        "end_close": round(points[-1][1], 4),
        "display_end_close": round(end_close, 4),
        "live_price": round(live_price, 4) if live_price is not None and live_price > 0 else None,
        "live_daily_return_pct": live_daily_return_pct,
        "live_included": live_included,
        "total_return_pct": total_return_pct,
        "annualized_return_pct": _annualized_cumulative_return_pct(
            total_return_pct, observation_days
        ),
        "mean_daily_return_pct": mean_daily,
        "annualized_mean_return_pct": _annualized_return_pct(mean_daily),
        "sharpe": _sharpe(spy_rets),
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "current_drawdown_pct": dd["current_drawdown_pct"],
        "volatility_ann_pct": _volatility_ann_pct(spy_rets),
        "observation_days": observation_days,
    }


def _fetch_spy_closes(start_date: Optional[str] = None) -> Tuple[str, str, List[Tuple[str, float]]]:
    """Return (start_date, start_label, closes) from Alpaca daily bars."""
    from src.apis.alpaca_client import AlpacaClient
    from src.config import FIRST_TRADE_DATE, FIRST_TRADE_DATE_LABEL

    start = start_date or FIRST_TRADE_DATE
    label = FIRST_TRADE_DATE_LABEL if start == FIRST_TRADE_DATE else start
    try:
        start_d = date.fromisoformat(start)
    except ValueError:
        logger.warning("Invalid SPY benchmark start_date=%s", start)
        return start, label, []

    today = datetime.now(timezone.utc).date()
    lookback_days = max(30, (today - start_d).days + 14)

    try:
        client = AlpacaClient(system="baseline")
        df = client.get_historical_bars("SPY", lookback_days=lookback_days)
    except Exception:
        logger.exception("Failed to fetch SPY bars for benchmark")
        return start, label, []

    if df is None or df.empty:
        return start, label, []

    closes: List[Tuple[str, float]] = []
    for _, row in df.iterrows():
        ts = row.get("date")
        close = row.get("close")
        if ts is None or close is None:
            continue
        try:
            dk = _date_key(ts.isoformat() if hasattr(ts, "isoformat") else str(ts))
            closes.append((dk, float(close)))
        except (TypeError, ValueError):
            continue
    closes.sort(key=lambda x: x[0])
    return start, label, closes


def _fetch_spy_live_price() -> Optional[float]:
    from src.apis.alpaca_client import AlpacaClient

    try:
        client = AlpacaClient(system="baseline")
        return client.get_latest_price("SPY")
    except Exception:
        logger.exception("Failed to fetch live SPY price")
        return None


def fetch_spy_benchmark(start_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """SPY total and compound-annualized return since first trade, from Alpaca daily bars."""
    start, label, closes = _fetch_spy_closes(start_date)
    return _spy_benchmark_from_closes(
        closes,
        start_date=start,
        start_label=label,
        live_price=_fetch_spy_live_price(),
    )


def attach_spy_benchmark(
    metrics: Dict[str, Any],
    baseline_history: Optional[List[Dict[str, Any]]] = None,
    internal_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Attach SPY benchmark and per-desk beta vs SPY (Alpaca daily closes)."""
    out = dict(metrics)
    start, label, closes = _fetch_spy_closes()
    spy_live_price = _fetch_spy_live_price()
    spy = _spy_benchmark_from_closes(
        closes,
        start_date=start,
        start_label=label,
        live_price=spy_live_price,
    )
    if not spy:
        return out

    out["benchmark"] = {"spy": spy}
    points = [(d, c) for d, c in closes if d >= start and c > 0]
    mkt_rets = _daily_returns_by_date(points)
    if spy.get("live_included") and spy.get("live_daily_return_pct") is not None:
        mkt_rets = dict(mkt_rets)
        mkt_rets["LIVE"] = float(spy["live_daily_return_pct"]) / 100.0

    for system, history in (
        ("baseline", baseline_history),
        ("internal", internal_history),
    ):
        agent = out.get(system)
        if not isinstance(agent, dict) or not history:
            continue
        port_rets = _portfolio_returns_by_date(history)
        latest_daily = agent.get("daily_return_pct")
        if latest_daily is not None and "LIVE" in mkt_rets:
            port_rets = dict(port_rets)
            port_rets["LIVE"] = float(latest_daily) / 100.0
        beta = _beta_vs_market(port_rets, mkt_rets)
        updated = dict(agent)
        updated["beta_spy"] = beta
        out[system] = updated

    return out


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


def _max_drawdown_from_points(points: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in points if v is not None and float(v) > 0]
    if not vals:
        return {"max_drawdown_pct": None, "current_drawdown_pct": None}
    peak = vals[0]
    max_dd = 0.0
    cur_dd = 0.0
    for v in vals:
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


def _binomial_significance(wins: int, trials: int) -> Dict[str, Any]:
    """Two-sided binomial test vs 50% (H0: neither desk leads more often)."""
    if trials < MIN_OBS_FOR_STATS:
        return _enrich_significance({
            "p_value": None,
            "significant_95": None,
            "test": "binomial",
            "n": trials,
            "insufficient_data": True,
        })

    wins = max(0, min(int(wins), int(trials)))
    p_f = float(stats.binomtest(wins, trials, p=0.5, alternative="two-sided").pvalue)
    result = {
        "p_value": round(p_f, 4),
        "significant_95": p_f < 0.05,
        "test": "binomial",
        "n": trials,
        "insufficient_data": False,
    }
    if wins == trials // 2 and trials % 2 == 0:
        out = dict(result)
        out["zero_effect"] = True
        return out
    return _enrich_significance(result)


def _path_comparison_stats(
    baseline_history: List[Dict[str, Any]],
    internal_history: List[Dict[str, Any]],
    b_rets: List[float],
    i_rets: List[float],
) -> Dict[str, Any]:
    """Head-to-head path stats: days equity ahead and daily return win rate."""
    b_eq = _equity_by_date(baseline_history)
    i_eq = _equity_by_date(internal_history)
    common = sorted(set(b_eq) & set(i_eq))

    eq_ahead = sum(1 for d in common if i_eq[d] > b_eq[d])
    eq_behind = sum(1 for d in common if i_eq[d] < b_eq[d])
    eq_ties = len(common) - eq_ahead - eq_behind
    eq_trials = eq_ahead + eq_behind

    ret_wins = sum(1 for b, i in zip(b_rets, i_rets) if i > b)
    ret_losses = sum(1 for b, i in zip(b_rets, i_rets) if i < b)
    ret_ties = len(b_rets) - ret_wins - ret_losses
    ret_trials = ret_wins + ret_losses

    def _block(wins: int, trials: int, ties: int, paired_days: int) -> Dict[str, Any]:
        losses = trials - wins if trials else 0
        rate = round(wins / trials * 100, 1) if trials else None
        return {
            "internal_wins": wins,
            "baseline_wins": losses,
            "paired_days": paired_days,
            "decisive_days": trials,
            "ties": ties,
            "rate_pct": rate,
            "significance": _binomial_significance(wins, trials) if trials else _binomial_significance(0, 0),
        }

    return {
        "days_equity_ahead": _block(eq_ahead, eq_trials, eq_ties, len(common)),
        "daily_win_rate": _block(ret_wins, ret_trials, ret_ties, len(b_rets)),
    }


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
    elif not out.get("insufficient_data") and out.get("significant_95") is False:
        out["zero_effect"] = True
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


def live_daily_return_pct(
    history_pts: List[Dict[str, Any]],
    account: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Today's return vs prior close: (equity − last_equity) / last_equity."""
    if not account:
        return None
    equity = account.get("equity")
    if equity is None:
        return None
    try:
        equity_f = float(equity)
    except (TypeError, ValueError):
        return None
    if equity_f <= 0:
        return None

    prior: Optional[float] = None
    last_equity = account.get("last_equity")
    if last_equity is not None:
        try:
            prior = float(last_equity)
        except (TypeError, ValueError):
            prior = None

    if prior is None or prior <= 0:
        by_date = _equity_by_date(history_pts)
        dates = sorted(by_date.keys())
        if dates:
            prior = by_date[dates[-1]]

    if prior is None or prior <= 0:
        return None
    return round((equity_f - prior) / prior * 100, 3)


def collect_live_daily_returns(
    history: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Optional[float]]]:
    """Today's live return for both desks from current account equity vs prior close."""
    from src.apis.alpaca_client import AlpacaClient

    out: Dict[str, Optional[float]] = {}
    for system in ("baseline", "internal"):
        try:
            client = AlpacaClient(system=system)
            account = client.get_account() or {}
            out[system] = live_daily_return_pct(history.get(system) or [], account)
        except Exception:
            out[system] = None

    if out.get("baseline") is None or out.get("internal") is None:
        return None
    return out


def collect_live_account_snapshots(
    history: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """Current account snapshot for both desks plus today's live return."""
    from src.apis.alpaca_client import AlpacaClient

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for system in ("baseline", "internal"):
        try:
            client = AlpacaClient(system=system)
            account = client.get_account() or {}
            out[system] = {
                "equity": _current_equity_from_account(account),
                "last_equity": (
                    float(account["last_equity"])
                    if account.get("last_equity") is not None
                    else None
                ),
                "daily_return_pct": live_daily_return_pct(history.get(system) or [], account),
            }
        except Exception:
            out[system] = {
                "equity": None,
                "last_equity": None,
                "daily_return_pct": None,
            }

    if not out:
        return None
    return out


def append_live_equity_points(
    history: Dict[str, List[Dict[str, Any]]],
    live_accounts: Optional[Dict[str, Dict[str, Optional[float]]]],
    *,
    starting_equity: float = 100_000.0,
) -> tuple[Dict[str, List[Dict[str, Any]]], str]:
    """Extend close history with the latest live equity point per desk (chart display series)."""
    if not live_accounts:
        return history, "closes_only"

    now = datetime.now(timezone.utc).isoformat()
    out: Dict[str, List[Dict[str, Any]]] = {}
    live_included = False

    for system in ("baseline", "internal"):
        pts = list(history.get(system) or [])
        equity = (live_accounts.get(system) or {}).get("equity")
        if equity is None:
            out[system] = pts
            continue
        try:
            pv = float(equity)
        except (TypeError, ValueError):
            out[system] = pts
            continue
        if pv <= 0:
            out[system] = pts
            continue

        live_included = True
        pnl_usd = pv - starting_equity
        pnl_pct = (pnl_usd / starting_equity * 100) if starting_equity else 0.0
        live_point = {
            "timestamp": now,
            "portfolio_value": round(pv, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pnl_usd, 2),
            "source": "live",
        }
        if pts and pts[-1].get("source") == "live":
            pts[-1] = live_point
        else:
            pts.append(live_point)
        out[system] = pts

    basis = "closes_plus_live_latest" if live_included else "closes_only"
    return out, basis


def _agent_metrics(
    returns: List[float],
    equity_by_date: Dict[str, float],
    starting_equity: float,
    latest_daily_return_pct: Optional[float],
    *,
    current_equity: Optional[float] = None,
) -> Dict[str, Any]:
    eq_points = [equity_by_date[d] for d in sorted(equity_by_date.keys())]
    if current_equity is not None:
        eq_points.append(current_equity)
    dd = _max_drawdown_from_points(eq_points)
    mean_daily = _mean_daily_return_pct(returns)
    n = len(returns)
    total_return = (
        round((current_equity - starting_equity) / starting_equity * 100, 3)
        if current_equity is not None and starting_equity > 0
        else _total_return_pct(equity_by_date, starting_equity)
    )
    return {
        "daily_return_pct": latest_daily_return_pct,
        "mean_daily_return_pct": mean_daily,
        "annualized_return_pct": _annualized_return_pct(mean_daily),
        "annualized_cumulative_return_pct": _annualized_cumulative_return_pct(
            total_return, n
        ),
        "sharpe": _sharpe(returns),
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "current_drawdown_pct": dd["current_drawdown_pct"],
        "volatility_ann_pct": _volatility_ann_pct(returns),
        "total_return_pct": total_return,
        "observation_days": n,
    }


def compute_head_to_head_metrics(
    baseline_history: List[Dict[str, Any]],
    internal_history: List[Dict[str, Any]],
    starting_equity: float = 100_000.0,
    *,
    live_daily_returns: Optional[Dict[str, Optional[float]]] = None,
    live_accounts: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
) -> Dict[str, Any]:
    """Compute aligned quant metrics and head-to-head significance tests."""
    dates, b_rets, i_rets = _align_returns(baseline_history, internal_history)
    b_eq = _equity_by_date(baseline_history)
    i_eq = _equity_by_date(internal_history)
    paired_days = len(b_rets)

    latest_b = round(b_rets[-1] * 100, 3) if b_rets else None
    latest_i = round(i_rets[-1] * 100, 3) if i_rets else None
    daily_delta_basis = "prior_close"
    display_b_rets = list(b_rets)
    display_i_rets = list(i_rets)
    current_b_equity: Optional[float] = None
    current_i_equity: Optional[float] = None

    if live_daily_returns:
        live_b = live_daily_returns.get("baseline")
        live_i = live_daily_returns.get("internal")
        if live_b is not None and live_i is not None:
            latest_b = live_b
            latest_i = live_i
            daily_delta_basis = "live"
            display_b_rets.append(live_b / 100.0)
            display_i_rets.append(live_i / 100.0)

    if live_accounts:
        current_b_equity = live_accounts.get("baseline", {}).get("equity")
        current_i_equity = live_accounts.get("internal", {}).get("equity")

    daily_delta = (
        round(latest_i - latest_b, 3)
        if latest_b is not None and latest_i is not None
        else None
    )

    close_excess = [i - b for b, i in zip(b_rets, i_rets)]
    display_excess = [i - b for b, i in zip(display_b_rets, display_i_rets)]
    mean_alpha = round(float(np.mean(display_excess)) * 100, 4) if display_excess else None
    annualized_alpha = _annualized_return_pct(mean_alpha)
    cum_alpha = (
        round((float(np.prod([1.0 + e for e in display_excess])) - 1.0) * 100, 3)
        if display_excess
        else None
    )

    b_sharpe = _sharpe(display_b_rets)
    i_sharpe = _sharpe(display_i_rets)
    sharpe_diff = (
        round(i_sharpe - b_sharpe, 3)
        if b_sharpe is not None and i_sharpe is not None
        else None
    )

    b_points = [b_eq[d] for d in sorted(b_eq.keys())]
    i_points = [i_eq[d] for d in sorted(i_eq.keys())]
    if current_b_equity is not None:
        b_points.append(current_b_equity)
    if current_i_equity is not None:
        i_points.append(current_i_equity)
    b_dd = _max_drawdown_from_points(b_points)
    i_dd = _max_drawdown_from_points(i_points)
    dd_diff = None
    if b_dd["max_drawdown_pct"] is not None and i_dd["max_drawdown_pct"] is not None:
        dd_diff = round(i_dd["max_drawdown_pct"] - b_dd["max_drawdown_pct"], 3)

    b_total = _agent_metrics(
        display_b_rets,
        b_eq,
        starting_equity,
        latest_b,
        current_equity=current_b_equity,
    )
    i_total = _agent_metrics(
        display_i_rets,
        i_eq,
        starting_equity,
        latest_i,
        current_equity=current_i_equity,
    )
    total_return_diff = None
    if b_total["total_return_pct"] is not None and i_total["total_return_pct"] is not None:
        total_return_diff = round(i_total["total_return_pct"] - b_total["total_return_pct"], 3)
    annualized_return_diff = None
    if (
        b_total["annualized_return_pct"] is not None
        and i_total["annualized_return_pct"] is not None
    ):
        annualized_return_diff = round(
            i_total["annualized_return_pct"] - b_total["annualized_return_pct"], 3
        )
    # Compound-annualize realized cumulative excess over the displayed paired
    # series (paired closes plus today's live move when available).
    n_obs = len(display_b_rets)
    annualized_excess = _annualized_cumulative_return_pct(total_return_diff, n_obs)
    annualized_cum_diff = None
    if (
        b_total["annualized_cumulative_return_pct"] is not None
        and i_total["annualized_cumulative_return_pct"] is not None
    ):
        annualized_cum_diff = round(
            i_total["annualized_cumulative_return_pct"]
            - b_total["annualized_cumulative_return_pct"],
            3,
        )

    close_b_total = _agent_metrics(b_rets, b_eq, starting_equity, round(b_rets[-1] * 100, 3) if b_rets else None)
    close_i_total = _agent_metrics(i_rets, i_eq, starting_equity, round(i_rets[-1] * 100, 3) if i_rets else None)
    close_total_return_diff = None
    if (
        close_b_total["total_return_pct"] is not None
        and close_i_total["total_return_pct"] is not None
    ):
        close_total_return_diff = round(
            close_i_total["total_return_pct"] - close_b_total["total_return_pct"], 3
        )
    close_annualized_excess = _annualized_cumulative_return_pct(
        close_total_return_diff, paired_days
    )
    close_sharpe_diff = (
        round(close_i_total["sharpe"] - close_b_total["sharpe"], 3)
        if close_b_total["sharpe"] is not None and close_i_total["sharpe"] is not None
        else None
    )
    close_dd_diff = (
        round(
            close_i_total["max_drawdown_pct"] - close_b_total["max_drawdown_pct"], 3
        )
        if close_b_total["max_drawdown_pct"] is not None
        and close_i_total["max_drawdown_pct"] is not None
        else None
    )

    internal_minus_baseline = {
        "daily_delta_pct": daily_delta,
        "excess_return_pct": total_return_diff,
        "mean_daily_alpha_pct": mean_alpha,
        "annualized_alpha_pct": annualized_alpha,
        "annualized_return_diff_pct": annualized_return_diff,
        "annualized_excess_return_pct": annualized_excess,
        "annualized_cumulative_return_diff_pct": annualized_cum_diff,
        "cumulative_alpha_pct": cum_alpha,
        "sharpe_diff": sharpe_diff,
        "max_drawdown_diff_pct": dd_diff,
    }

    path_comparison = _path_comparison_stats(
        baseline_history, internal_history, b_rets, i_rets,
    )

    return {
        "observation_days": len(display_b_rets),
        "paired_observation_days": paired_days,
        "min_days_for_stats": MIN_OBS_FOR_STATS,
        "latest_date": dates[-1] if dates else None,
        "path_comparison": path_comparison,
        "baseline": b_total,
        "internal": i_total,
        "comparison": {
            "sign_convention": SIGN_CONVENTION,
            "formula": COMPARISON_FORMULA,
            "internal_minus_baseline": internal_minus_baseline,
            # Flat aliases kept for dashboard backward compatibility.
            "daily_delta_pct": daily_delta,
            "daily_delta_basis": daily_delta_basis,
            "series_basis": (
                "paired_plus_live_latest" if daily_delta_basis == "live" else "paired_closes"
            ),
            "paired_close": {
                "observation_days": paired_days,
                "daily_delta_pct": (
                    round(i_rets[-1] * 100 - b_rets[-1] * 100, 3) if b_rets and i_rets else None
                ),
                "mean_daily_alpha_pct": (
                    round(float(np.mean(close_excess)) * 100, 4) if close_excess else None
                ),
                "total_return_diff_pct": close_total_return_diff,
                "annualized_excess_return_pct": close_annualized_excess,
                "sharpe_diff": close_sharpe_diff,
                "max_drawdown_diff_pct": close_dd_diff,
            },
            "mean_daily_alpha_pct": mean_alpha,
            "annualized_alpha_pct": annualized_alpha,
            "annualized_return_diff_pct": annualized_return_diff,
            "annualized_excess_return_pct": annualized_excess,
            "annualized_cumulative_return_diff_pct": annualized_cum_diff,
            "cumulative_alpha_pct": cum_alpha,
            "total_return_diff_pct": total_return_diff,
            "sharpe_diff": sharpe_diff,
            "risk_free_rate_annual": RISK_FREE_RATE_ANNUAL,
            "max_drawdown_diff_pct": dd_diff,
            "field_glossary": {
                "daily_delta_pct": (
                    "Internal − Baseline latest return in the displayed paired series. "
                    "Historical observations are paired close-to-close; today's live move "
                    "is appended when available."
                ),
                "mean_daily_alpha_pct": (
                    "Mean daily Internal − Baseline return over the displayed paired "
                    "series, including today's live point when available."
                ),
                "annualized_alpha_pct": (
                    "Mean daily alpha × 252 trading days (simple annualization)."
                ),
                "annualized_return_diff_pct": (
                    "Internal − Baseline annualized return "
                    "(each desk: mean daily return × 252)."
                ),
                "annualized_excess_return_pct": (
                    "Compound-annualized excess return: "
                    "(1 + excess)^(252/n) − 1 over the displayed paired series. "
                    "Distinct from mean-daily × 252."
                ),
                "annualized_cumulative_return_diff_pct": (
                    "Internal − Baseline compound-annualized cumulative return "
                    "(each desk: (1 + total_return)^(252/n) − 1)."
                ),
                "total_return_diff_pct": (
                    "Internal − Baseline total return vs starting equity using the "
                    "displayed paired series and current live equity when available."
                ),
                "sharpe_diff": (
                    f"Internal Sharpe − Baseline Sharpe "
                    f"(excess over {RISK_FREE_RATE_ANNUAL * 100:.2f}% annual risk-free). "
                    "Positive favors Internal."
                ),
                "max_drawdown_diff_pct": (
                    "Internal max drawdown − Baseline max drawdown (pp). "
                    "Negative means Internal had a shallower drawdown."
                ),
            },
            "significance": {
                "total_return_diff": _bootstrap_total_return_diff(display_b_rets, display_i_rets),
                "daily_alpha": _significance_ttest(display_excess),
                "sharpe_diff": _bootstrap_sharpe_diff(display_b_rets, display_i_rets),
                "max_drawdown_diff": _bootstrap_max_dd_diff(display_b_rets, display_i_rets),
            },
        },
    }
