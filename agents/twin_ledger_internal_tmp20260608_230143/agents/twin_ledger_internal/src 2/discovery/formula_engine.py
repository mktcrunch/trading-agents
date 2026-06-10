"""
Safe formula DSL for LLM-proposed OHLCV features.
Only whitelisted columns and composable ops — no arbitrary code execution.
"""
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__)

ALLOWED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})
BINARY_OPS = frozenset({"add", "sub", "mul", "div"})
MAX_FORMULA_DEPTH = 5
MAX_WINDOW = 120
MIN_WINDOW = 2
MAX_PERIODS = 60
MIN_PERIODS = 1
FEATURE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


def sanitize_feature_id(raw_id: str) -> Optional[str]:
    cleaned = raw_id.strip().lower().replace("-", "_").replace(" ", "_")
    if FEATURE_ID_PATTERN.match(cleaned):
        return cleaned
    return None


def validate_formula(node: Any, depth: int = 0) -> Optional[str]:
    """Return error message if invalid, else None."""
    if depth > MAX_FORMULA_DEPTH:
        return "formula exceeds max nesting depth"

    if not isinstance(node, dict):
        return "formula node must be an object"

    op = node.get("op")
    if not op or not isinstance(op, str):
        return "missing op"

    if op == "column":
        col = node.get("column") or node.get("name")
        if col not in ALLOWED_COLUMNS:
            return f"disallowed column: {col}"
        return None

    if op == "pct_change":
        col = node.get("column")
        periods = int(node.get("periods", 1))
        if col not in ALLOWED_COLUMNS:
            return f"disallowed column: {col}"
        if not MIN_PERIODS <= periods <= MAX_PERIODS:
            return f"periods out of range: {periods}"
        return None

    if op == "diff":
        col = node.get("column")
        periods = int(node.get("periods", 1))
        if col not in ALLOWED_COLUMNS:
            return f"disallowed column: {col}"
        if not MIN_PERIODS <= periods <= MAX_PERIODS:
            return f"periods out of range: {periods}"
        return None

    if op in ("rolling_mean", "rolling_std", "rolling_min", "rolling_max", "zscore"):
        col = node.get("column")
        window = int(node.get("window", 20))
        if col not in ALLOWED_COLUMNS:
            return f"disallowed column: {col}"
        if not MIN_WINDOW <= window <= MAX_WINDOW:
            return f"window out of range: {window}"
        return None

    if op == "range_pct":
        return None

    if op == "log1p":
        err = validate_formula(node.get("inner"), depth + 1)
        return err

    if op in BINARY_OPS:
        for side in ("left", "right"):
            if side not in node:
                return f"missing {side}"
            err = validate_formula(node[side], depth + 1)
            if err:
                return err
        return None

    return f"unknown op: {op}"


def validate_feature_spec(spec: Dict) -> Optional[Dict]:
    """Validate and normalize a feature proposal. Returns cleaned spec or None."""
    if not isinstance(spec, dict):
        return None

    fid = sanitize_feature_id(str(spec.get("id", "")))
    if not fid:
        logger.warning(f"Rejected feature with invalid id: {spec.get('id')}")
        return None

    formula = spec.get("formula")
    if not isinstance(formula, dict):
        logger.warning(f"Rejected {fid}: missing formula")
        return None

    err = validate_formula(formula)
    if err:
        logger.warning(f"Rejected {fid}: {err}")
        return None

    description = str(spec.get("description", fid))[:200]
    return {"id": fid, "description": description, "formula": formula}


def _bounded_window(node: Dict, default: int = 20) -> int:
    return max(MIN_WINDOW, min(MAX_WINDOW, int(node.get("window", default))))


def _bounded_periods(node: Dict, default: int = 1) -> int:
    return max(MIN_PERIODS, min(MAX_PERIODS, int(node.get("periods", default))))


def _min_periods(node: Dict, window: int) -> int:
    return max(2, min(window, int(node.get("min_periods", max(2, window // 2)))))


def compute_formula(df: pd.DataFrame, node: Dict) -> pd.Series:
    """Evaluate a validated formula node against a single-symbol OHLCV frame."""
    op = node["op"]

    if op == "column":
        col = node.get("column") or node.get("name")
        return df[col].astype(float)

    if op == "pct_change":
        return df[node["column"]].astype(float).pct_change(_bounded_periods(node))

    if op == "diff":
        return df[node["column"]].astype(float).diff(_bounded_periods(node))

    if op == "rolling_mean":
        window = _bounded_window(node)
        return (
            df[node["column"]]
            .astype(float)
            .rolling(window, min_periods=_min_periods(node, window))
            .mean()
        )

    if op == "rolling_std":
        window = _bounded_window(node)
        return (
            df[node["column"]]
            .astype(float)
            .rolling(window, min_periods=_min_periods(node, window))
            .std()
        )

    if op == "rolling_min":
        window = _bounded_window(node)
        return (
            df[node["column"]]
            .astype(float)
            .rolling(window, min_periods=_min_periods(node, window))
            .min()
        )

    if op == "rolling_max":
        window = _bounded_window(node)
        return (
            df[node["column"]]
            .astype(float)
            .rolling(window, min_periods=_min_periods(node, window))
            .max()
        )

    if op == "zscore":
        window = _bounded_window(node)
        col = df[node["column"]].astype(float)
        mean = col.rolling(window, min_periods=_min_periods(node, window)).mean()
        std = col.rolling(window, min_periods=_min_periods(node, window)).std()
        return (col - mean) / std.replace(0, np.nan)

    if op == "range_pct":
        close = df["close"].astype(float)
        return (df["high"].astype(float) - df["low"].astype(float)) / close.replace(0, np.nan)

    if op == "log1p":
        inner = compute_formula(df, node["inner"])
        return np.log1p(inner.abs())

    if op in BINARY_OPS:
        left = compute_formula(df, node["left"])
        right = compute_formula(df, node["right"])
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        if op == "div":
            return left / right.replace(0, np.nan)

    raise ValueError(f"Unsupported op: {op}")


def feature_column_name(feature_id: str) -> str:
    return f"feat_{feature_id}"


def specs_to_definitions(specs: List[Dict]) -> List[Dict]:
    return [
        {
            "id": s["id"],
            "column": feature_column_name(s["id"]),
            "description": s["description"],
            "formula": s["formula"],
            "proposed": True,
        }
        for s in specs
    ]


FORMULA_DSL_DOCS = """
Available formula ops (compose with nested objects):
- {"op": "column", "column": "close"}  — columns: open, high, low, close, volume
- {"op": "pct_change", "column": "close", "periods": 5}
- {"op": "diff", "column": "volume", "periods": 1}
- {"op": "rolling_mean", "column": "close", "window": 20}
- {"op": "rolling_std", "column": "volume", "window": 20}
- {"op": "rolling_min", "column": "low", "window": 10}
- {"op": "rolling_max", "column": "high", "window": 10}
- {"op": "zscore", "column": "volume", "window": 20}
- {"op": "range_pct"}  — (high-low)/close
- {"op": "log1p", "inner": {...}}
- {"op": "add"|"sub"|"mul"|"div", "left": {...}, "right": {...}}

Constraints: periods 1-60, window 2-120, max nesting depth 5.
Feature ids: lowercase snake_case, 3-40 chars, start with a letter.
"""
