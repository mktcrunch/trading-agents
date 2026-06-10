"""
Feature engineering from DataBento daily OHLCV bars.
Supports fixed baseline features plus LLM-proposed formula specs.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.discovery.formula_engine import (
    compute_formula,
    feature_column_name,
    specs_to_definitions,
    validate_feature_spec,
)

OHLCV_SCHEMAS = ("ohlcv-1d", "ohlcv-1h", "ohlcv-1m", "ohlcv-1s")


def supports_probe_schema(schema: str) -> bool:
    return schema.startswith("ohlcv")


def normalize_bars(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    """
    Normalize fetched bars to daily OHLCV suitable for feature engineering.
    Intraday schemas are aggregated to one bar per symbol per day.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if schema == "ohlcv-1d":
        return df

    if not schema.startswith("ohlcv"):
        return pd.DataFrame()

    work = df.reset_index() if "symbol" not in df.columns else df.copy()
    if "ts_event" in work.columns:
        work = work.rename(columns={"ts_event": "date"})
    elif work.index.name == "ts_event":
        work = work.reset_index().rename(columns={"ts_event": "date"})

    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    agg = (
        work.groupby(["symbol", "date"])
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    return agg


FEATURE_DEFINITIONS = [
    {
        "id": "volume_zscore",
        "description": "20-day volume z-score",
        "column": "volume_zscore",
    },
    {
        "id": "momentum_5d",
        "description": "5-day close-to-close return",
        "column": "momentum_5d",
    },
    {
        "id": "range_pct",
        "description": "Intraday range as % of close",
        "column": "range_pct",
    },
    {
        "id": "close_vs_sma20",
        "description": "Close relative to 20-day SMA",
        "column": "close_vs_sma20",
    },
]


def _apply_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    vol_mean = df["volume"].rolling(20, min_periods=10).mean()
    vol_std = df["volume"].rolling(20, min_periods=10).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    df["momentum_5d"] = df["close"].pct_change(5)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    sma20 = df["close"].rolling(20, min_periods=10).mean()
    df["close_vs_sma20"] = df["close"] / sma20 - 1.0
    return df


def _apply_proposed_features(
    df: pd.DataFrame,
    feature_specs: List[Dict],
) -> pd.DataFrame:
    for spec in feature_specs:
        col = feature_column_name(spec["id"])
        try:
            df[col] = compute_formula(df, spec["formula"])
        except Exception:
            df[col] = np.nan
    return df


def _prepare_symbol_df(
    symbol_df: pd.DataFrame,
    feature_specs: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    df = symbol_df.sort_index().copy()
    df["return_1d"] = df["close"].pct_change()
    df["forward_return_1d"] = df["close"].pct_change().shift(-1)
    df = _apply_baseline_features(df)
    if feature_specs:
        df = _apply_proposed_features(df, feature_specs)
    df["baseline_momentum"] = df["return_1d"].shift(1)
    return df


def build_feature_definitions(
    feature_specs: Optional[List[Dict]] = None,
    include_baseline: bool = True,
) -> List[Dict]:
    """Merge baseline + validated proposed feature definitions."""
    definitions = []
    if include_baseline:
        definitions.extend(FEATURE_DEFINITIONS)
    if feature_specs:
        validated = []
        for spec in feature_specs:
            clean = validate_feature_spec(spec)
            if clean:
                validated.append(clean)
        definitions.extend(specs_to_definitions(validated))
    return definitions


def build_feature_panel(
    ohlcv_df: pd.DataFrame,
    feature_specs: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """
    Build a long panel: one row per (date, symbol) with engineered features.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return pd.DataFrame()

    df = ohlcv_df.reset_index() if "symbol" not in ohlcv_df.columns else ohlcv_df.copy()
    if "ts_event" in df.columns:
        df = df.rename(columns={"ts_event": "date"})
    elif df.index.name == "ts_event":
        df = df.reset_index().rename(columns={"ts_event": "date"})

    validated_specs = []
    if feature_specs:
        for spec in feature_specs:
            clean = validate_feature_spec(spec)
            if clean:
                validated_specs.append(clean)

    panels = []
    for symbol, group in df.groupby("symbol"):
        prepared = _prepare_symbol_df(group.set_index("date"), validated_specs)
        prepared["symbol"] = symbol
        panels.append(prepared.reset_index())

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels, ignore_index=True)
    return panel.dropna(subset=["forward_return_1d"])


def feature_id_with_prefix(prefix: str, feature_id: str) -> str:
    if not prefix:
        return feature_id
    return f"{prefix}_{feature_id}"


def latest_ticker_features(
    panel: pd.DataFrame,
    prefix: str = "",
    definitions: Optional[List[Dict]] = None,
) -> Dict[str, Dict[str, float]]:
    """Latest feature values per ticker for trading enrichment."""
    if panel.empty:
        return {}

    defs = definitions or FEATURE_DEFINITIONS
    result: Dict[str, Dict[str, float]] = {}

    for symbol, group in panel.groupby("symbol"):
        latest = group.sort_values("date").iloc[-1]
        feats = {}
        for definition in defs:
            col = definition["column"]
            if col in latest and pd.notna(latest[col]):
                key = feature_id_with_prefix(prefix, definition["id"])
                feats[key] = round(float(latest[col]), 6)
        if feats:
            result[symbol] = feats

    return result
