"""
Intraday risk monitor for baseline and internal Alpaca accounts.

Internal: ATR base stops, hybrid trailing (scripted 1%/70% + LLM), 15-min prediction, EOD.
Baseline: fixed base stop, pure LLM trailing, EOD — no prediction deferral.
"""
from __future__ import annotations

import json
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytz
import requests

from src import config
from src.apis.alpaca_client import AlpacaClient
from src.logger import setup_logger
from src.models.position import Position
from src.risk.risk_state import RiskStateStore
from src.risk.trailing_planner import TrailingPlanner, merge_hybrid_trailing

logger = setup_logger(__name__)


def _audit(event_type: str, action: str, system: str, payload: Dict, status: str = "ok") -> None:
    if not config.AUDIT_ENABLED:
        return
    from src.audit import record_event
    record_event(
        event_type=event_type,
        action=action,
        system=system,
        agent="RiskMonitor",
        status=status,
        payload=payload,
    )

ET = pytz.timezone("US/Eastern")


class RiskMonitor:
    """Single-account intraday risk session."""

    def __init__(self, system: str):
        if system not in ("baseline", "internal"):
            raise ValueError(f"Invalid system: {system}")

        self.system = system
        self.cfg = (
            config.BASELINE_RISK_CONFIG
            if system == "baseline"
            else config.INTERNAL_RISK_CONFIG
        )
        self.alpaca = AlpacaClient(system=system)
        self.state_store = RiskStateStore(system)
        self._atr_cache: Dict[Tuple[str, str], Optional[float]] = {}
        self._trailing_planner = None
        if self.cfg.get("llm_trailing_planner") or self.cfg.get("trailing_mode") in ("llm", "hybrid"):
            self._trailing_planner = TrailingPlanner(system=system)
        self._mc_client = None
        if system == "internal":
            from src.apis.marketcrunch_client import MarketCrunchClient
            self._mc_client = MarketCrunchClient()

    # ------------------------------------------------------------------
    # Market clock
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        clock = self.alpaca.get_clock()
        if clock is not None:
            return bool(clock.get("is_open"))
        now = datetime.now(ET)
        if now.weekday() >= 5:
            return False
        return dt_time(9, 30) <= now.time() <= dt_time(16, 0)

    def _is_eod_window(self) -> bool:
        clock = self.alpaca.get_clock()
        if clock and clock.get("is_open"):
            now_ts = clock["timestamp"]
            close_ts = clock["next_close"]
            if hasattr(now_ts, "tzinfo") and now_ts.tzinfo is None:
                now_ts = ET.localize(now_ts)
            if hasattr(close_ts, "tzinfo") and close_ts.tzinfo is None:
                close_ts = ET.localize(close_ts)
            minutes_to_close = (close_ts - now_ts).total_seconds() / 60.0
            return 0 <= minutes_to_close <= self.cfg.get("eod_window_minutes", 6.3)

        now = datetime.now(ET)
        if now.weekday() >= 5:
            return False
        start = dt_time(15, 53, 42)
        return dt_time(15, 53, 42) <= now.time() <= dt_time(16, 0)

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(high, low, close, period: int) -> Optional[float]:
        if len(high) < period + 1:
            return None
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-period:]))

    def _get_atr(self, ticker: str) -> Optional[float]:
        period = self.cfg.get("atr_period", 14)
        cache_key = (ticker, datetime.now(ET).date().isoformat())
        if cache_key in self._atr_cache:
            return self._atr_cache[cache_key]

        df = self.alpaca.get_historical_bars(ticker, lookback_days=period + 60)
        atr_val = None
        if df is not None and len(df) >= period + 1:
            atr_val = self._compute_atr(
                df["high"].tolist(),
                df["low"].tolist(),
                df["close"].tolist(),
                period,
            )
        self._atr_cache[cache_key] = atr_val
        return atr_val

    # ------------------------------------------------------------------
    # Returns & trailing stops
    # ------------------------------------------------------------------

    @staticmethod
    def _position_return(pos: Position) -> float:
        if pos.avg_entry_price <= 0:
            return 0.0
        if pos.qty < 0:
            return (pos.avg_entry_price - pos.current_price) / pos.avg_entry_price
        return (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price

    def _momentum_5d(self, ticker: str) -> Optional[float]:
        df = self.alpaca.get_historical_bars(ticker, lookback_days=10)
        if df is None or len(df) < 6:
            return None
        closes = df["close"]
        return float((closes.iloc[-1] / closes.iloc[-6]) - 1.0)

    def _mc_context(self, ticker: str) -> Optional[Dict]:
        if not self._mc_client:
            return None
        try:
            return self._mc_client.get_ai_estimates(ticker)
        except Exception as e:
            logger.warning(f"MC context fetch failed for {ticker}: {e}")
            return None

    def _llm_trailing_plan(
        self,
        state: Dict,
        ticker: str,
        pos: Position,
        current_return: float,
    ) -> Dict:
        atr = self._get_atr(ticker)
        atr_pct = (atr / pos.current_price) if atr and pos.current_price else None
        llm_cache = state.setdefault("llm_trailing_params", {})
        cached = llm_cache.get(ticker)
        plan = self._trailing_planner.plan(
            ticker=ticker,
            entry=pos.avg_entry_price,
            current_price=pos.current_price,
            current_return=current_return,
            atr_pct=atr_pct,
            momentum_5d=self._momentum_5d(ticker),
            mc_context=self._mc_context(ticker) if self.system == "internal" else None,
            cached=cached,
        )
        llm_cache[ticker] = plan
        return plan

    def _resolve_trailing_params(
        self,
        state: Dict,
        ticker: str,
        pos: Position,
        current_return: float,
    ) -> Optional[Tuple[float, float, str]]:
        if not self.cfg.get("use_trailing_stop", True):
            return None

        mode = self.cfg.get("trailing_mode", "llm")

        # Baseline: pure LLM only
        if mode == "llm":
            if not self._trailing_planner:
                return None
            plan = self._llm_trailing_plan(state, ticker, pos, current_return)
            activation = plan["activation_threshold"]
            lock_frac = plan["profit_lock_fraction"]
            policy = f"llm:{plan.get('rationale', '')[:80]}"
            
            # Log audit event for trailing stop planning
            _audit(
                event_type="trailing_stop_planned",
                action=f"Planned trailing stop for {ticker}: activation={activation*100:.1f}%, lock={lock_frac*100:.1f}%",
                system=self.system,
                payload={
                    "ticker": ticker,
                    "activation_threshold": activation,
                    "profit_lock_fraction": lock_frac,
                    "current_return": current_return,
                    "policy": "llm",
                    "rationale": plan.get("rationale")
                }
            )
            return (
                activation,
                lock_frac,
                policy,
            )

        # Internal: scripted floor + LLM refinement
        if mode == "hybrid":
            scripted_act = self.cfg["trailing_activation_threshold"]
            scripted_lock = self.cfg["profit_lock_fraction"]
            if not self._trailing_planner:
                return scripted_act, scripted_lock, "scripted_only"
            plan = self._llm_trailing_plan(state, ticker, pos, current_return)
            activation, lock_frac, policy = merge_hybrid_trailing(
                scripted_act, scripted_lock, plan
            )
            
            # Log audit event for trailing stop planning
            _audit(
                event_type="trailing_stop_planned",
                action=f"Planned trailing stop for {ticker}: activation={activation*100:.1f}%, lock={lock_frac*100:.1f}%",
                system=self.system,
                payload={
                    "ticker": ticker,
                    "activation_threshold": activation,
                    "profit_lock_fraction": lock_frac,
                    "current_return": current_return,
                    "policy": policy,
                    "rationale": plan.get("rationale")
                }
            )
            return activation, lock_frac, policy

        return None

    def _update_trailing_stop(
        self,
        state: Dict,
        ticker: str,
        pos: Position,
        current_return: float,
    ) -> None:
        resolved = self._resolve_trailing_params(state, ticker, pos, current_return)
        if not resolved:
            return

        activation, lock_frac, policy = resolved
        if current_return <= activation:
            return

        direction = "short" if pos.qty < 0 else "long"
        entry = pos.avg_entry_price
        profit_to_lock = current_return * lock_frac

        if direction == "long":
            new_stop = entry * (1 + profit_to_lock)
        else:
            new_stop = entry * (1 - profit_to_lock)

        stops = state.setdefault("trailing_stops", {})
        existing = stops.get(ticker)
        meta = {
            "stop_price": new_stop,
            "direction": direction,
            "policy": policy,
            "activation_threshold": activation,
            "profit_lock_fraction": lock_frac,
        }
        if not existing or existing.get("direction") != direction:
            stops[ticker] = meta
            logger.info(
                f"[{self.system}] Trailing stop init {ticker}: ${new_stop:.2f} "
                f"(return {current_return*100:.2f}%, policy={policy})"
            )
            _audit(
                "trailing_stop_init",
                f"Init trailing stop for {ticker} at ${new_stop:.2f}",
                self.system,
                meta
            )
            return

        old = existing["stop_price"]
        should_update = (
            (direction == "long" and new_stop > old)
            or (direction == "short" and new_stop < old)
        )
        if should_update:
            stops[ticker] = {**existing, **meta, "stop_price": new_stop}
            logger.info(
                f"[{self.system}] Trailing stop update {ticker}: "
                f"${old:.2f} → ${new_stop:.2f} ({policy})"
            )
            _audit(
                "trailing_stop_update",
                f"Updated trailing stop for {ticker}: ${old:.2f} -> ${new_stop:.2f}",
                self.system,
                {**meta, "old_stop_price": old}
            )

    def _trailing_triggered(
        self, state: Dict, ticker: str, current_price: float
    ) -> bool:
        stop_data = state.get("trailing_stops", {}).get(ticker)
        if not stop_data:
            return False
        level = stop_data["stop_price"]
        direction = stop_data["direction"]
        if direction == "long":
            return current_price <= level
        return current_price >= level

    def _cleanup_stale_stops(self, state: Dict, positions: Dict[str, Position]) -> None:
        stops = state.get("trailing_stops", {})
        stale = [t for t in stops if t not in positions]
        for t in stale:
            del stops[t]
            logger.info(f"[{self.system}] Cleared stale trailing stop for {t}")

        llm_params = state.get("llm_trailing_params", {})
        stale_llm = [t for t in llm_params if t not in positions]
        for t in stale_llm:
            del llm_params[t]

    # ------------------------------------------------------------------
    # Stop identification
    # ------------------------------------------------------------------

    def _base_stop_hit(
        self, ticker: str, pos: Position, current_return: float
    ) -> Tuple[bool, str]:
        threshold = self.cfg["base_stop_loss_threshold"]

        if current_return >= 0:
            return current_return <= threshold, "fixed_positive"

        if self.cfg.get("use_atr_base_stop", False):
            atr = self._get_atr(ticker)
            if atr and atr > 0:
                mult = self.cfg.get("atr_stop_multiplier", 1.5)
                dist = atr * mult
                if pos.qty >= 0:
                    stop_px = pos.avg_entry_price - dist
                    hit = pos.current_price <= stop_px
                else:
                    stop_px = pos.avg_entry_price + dist
                    hit = pos.current_price >= stop_px
                if hit:
                    return True, f"atr_stop@${stop_px:.2f}"
                return False, f"atr_active@${stop_px:.2f}"

        return current_return <= threshold, f"fixed_{threshold*100:.1f}%"

    def _identify_stop_candidates(
        self, positions: Dict[str, Position], state: Dict
    ) -> List[Tuple[str, Position, float, str]]:
        candidates = []
        for ticker, pos in positions.items():
            ret = self._position_return(pos)
            self._update_trailing_stop(state, ticker, pos, ret)

            trailing = self._trailing_triggered(state, ticker, pos.current_price)
            base_hit, base_reason = self._base_stop_hit(ticker, pos, ret)

            if trailing:
                candidates.append((ticker, pos, ret, "trailing_stop"))
                logger.warning(
                    f"[{self.system}] TRAILING STOP {ticker}: "
                    f"return={ret*100:.2f}% price=${pos.current_price:.2f}"
                )
            elif base_hit:
                candidates.append((ticker, pos, ret, f"base_stop:{base_reason}"))
                logger.warning(
                    f"[{self.system}] BASE STOP {ticker}: "
                    f"return={ret*100:.2f}% ({base_reason})"
                )

        return candidates

    def _eod_threshold(self, ticker: str, entry: float) -> Tuple[float, str]:
        fixed = self.cfg["eod_exit_threshold"]
        if not self.cfg.get("use_atr_base_stop", False) or entry <= 0:
            return fixed, "fixed"

        atr = self._get_atr(ticker)
        if atr and atr > 0:
            mult = self.cfg.get("atr_stop_multiplier", 1.5)
            return -(atr * mult) / entry, f"atr_{mult}x"
        return fixed, "fixed"

    def _identify_eod_candidates(
        self, positions: Dict[str, Position]
    ) -> List[Tuple[str, Position, float, str]]:
        if not self.cfg.get("eod_exit_enabled", True):
            return []

        candidates = []
        for ticker, pos in positions.items():
            ret = self._position_return(pos)
            thr, mode = self._eod_threshold(ticker, pos.avg_entry_price)
            if ret < thr:
                candidates.append((ticker, pos, ret, f"eod_{mode}"))
                logger.warning(
                    f"[{self.system}] EOD CANDIDATE {ticker}: "
                    f"return={ret*100:.2f}% < {thr*100:.2f}%"
                )
        return candidates

    # ------------------------------------------------------------------
    # 15-min prediction (internal only)
    # ------------------------------------------------------------------

    def _get_15min_prediction(self, ticker: str) -> Optional[float]:
        url = config.PREDICT_15MIN_URL
        try:
            resp = requests.post(
                url,
                json={"ticker": ticker},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("prediction", {}).get("predicted_close_price_15min")
        except Exception as e:
            logger.warning(f"15-min prediction failed for {ticker}: {e}")
            return None

    def _should_exit_with_prediction(
        self, ticker: str, pos: Position, reason: str
    ) -> Tuple[bool, Optional[float]]:
        if reason.startswith("eod_"):
            return True, None
        if not self.cfg.get("use_15min_prediction_gate", False):
            return True, None

        predicted = self._get_15min_prediction(ticker)
        if predicted is None:
            return True, None

        current = pos.current_price
        if pos.qty < 0:
            should_sell = predicted >= current
        else:
            should_sell = predicted <= current

        if should_sell:
            logger.info(
                f"[{self.system}] Prediction agrees exit {ticker}: "
                f"pred=${predicted:.2f} vs current=${current:.2f}"
            )
        else:
            logger.info(
                f"[{self.system}] Prediction defers exit {ticker}: "
                f"pred=${predicted:.2f} vs current=${current:.2f}"
            )
        return should_sell, predicted

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _close_position(self, ticker: str, pos: Position, dry_run: bool) -> Dict:
        if dry_run or config.DRY_RUN or not config.TRADING_ENABLED:
            return {"status": "simulated", "ticker": ticker, "qty": abs(pos.qty)}

        side = "sell" if pos.qty > 0 else "buy"
        order_id = self.alpaca.place_market_order(
            ticker=ticker,
            qty=abs(pos.qty),
            side=side,
            time_in_force="day",
        )
        if order_id:
            return {"status": "submitted", "order_id": order_id, "ticker": ticker}
        return {"status": "failed", "ticker": ticker}

    # ------------------------------------------------------------------
    # Main session
    # ------------------------------------------------------------------

    def run_check(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run one intraday risk check."""
        logger.info(f"[{self.system}] Starting risk monitor session")

        if not self.is_market_open():
            logger.info(f"[{self.system}] Market closed — skipping risk check")
            return {"skipped": True, "reason": "market_closed", "system": self.system}

        state = self.state_store.load()
        positions = self.alpaca.get_positions()
        self._cleanup_stale_stops(state, positions)

        # Audit the active positions and returns being checked
        position_returns = {}
        for ticker, pos in positions.items():
            ret = self._position_return(pos)
            position_returns[ticker] = {
                "current_price": pos.current_price,
                "entry_price": pos.avg_entry_price,
                "return_pct": round(ret * 100, 4)
            }
        _audit(
            event_type="risk_positions_checked",
            action=f"Checked returns for {len(positions)} positions",
            system=self.system,
            payload={"returns": position_returns}
        )

        results: Dict[str, Any] = {
            "system": self.system,
            "positions": len(positions),
            "stop_exits": [],
            "eod_exits": [],
            "held": [],
            "dry_run": dry_run or config.DRY_RUN,
        }

        # Regular stop-loss / trailing
        stop_candidates = self._identify_stop_candidates(positions, state)
        for ticker, pos, ret, reason in stop_candidates:
            should_exit, predicted = self._should_exit_with_prediction(ticker, pos, reason)
            if not should_exit:
                held = {
                    "ticker": ticker,
                    "reason": reason,
                    "predicted_close": predicted,
                }
                results["held"].append(held)
                _audit("risk_held", f"Held {ticker} (prediction)", self.system, held)
                continue

            exec_result = self._close_position(ticker, pos, dry_run)
            exit_record = {
                "ticker": ticker,
                "return_pct": round(ret * 100, 4),
                "reason": reason,
                "predicted_close": predicted,
                "execution": exec_result,
            }
            results["stop_exits"].append(exit_record)
            _audit(
                "risk_stop_exit",
                f"Stop exit {ticker}: {reason}",
                self.system,
                exit_record,
                status=exec_result.get("status", "ok"),
            )
            if exec_result.get("status") in ("submitted", "simulated"):
                state.get("trailing_stops", {}).pop(ticker, None)

        # EOD exit — once per calendar day
        today = datetime.now(ET).date().isoformat()
        if (
            self._is_eod_window()
            and state.get("eod_exit_done_date") != today
        ):
            eod_candidates = self._identify_eod_candidates(positions)
            for ticker, pos, ret, reason in eod_candidates:
                exec_result = self._close_position(ticker, pos, dry_run)
                eod_record = {
                    "ticker": ticker,
                    "return_pct": round(ret * 100, 4),
                    "reason": reason,
                    "execution": exec_result,
                }
                results["eod_exits"].append(eod_record)
                _audit(
                    "risk_eod_exit",
                    f"EOD exit {ticker}",
                    self.system,
                    eod_record,
                    status=exec_result.get("status", "ok"),
                )
                if exec_result.get("status") in ("submitted", "simulated"):
                    state.get("trailing_stops", {}).pop(ticker, None)
            state["eod_exit_done_date"] = today
            logger.info(f"[{self.system}] EOD exit guard set for {today}")
        elif self._is_eod_window():
            logger.info(f"[{self.system}] EOD window but already ran today")

        self.state_store.save(state)
        logger.info(
            f"[{self.system}] Risk session done: "
            f"{len(results['stop_exits'])} stops, "
            f"{len(results['eod_exits'])} eod, "
            f"{len(results['held'])} held"
        )
        return results


def run_risk_for_system(system: str, dry_run: bool = False) -> Dict[str, Any]:
    return RiskMonitor(system).run_check(dry_run=dry_run)


def run_risk_all(dry_run: bool = False) -> Dict[str, Any]:
    return {
        "baseline": run_risk_for_system("baseline", dry_run=dry_run),
        "internal": run_risk_for_system("internal", dry_run=dry_run),
    }
