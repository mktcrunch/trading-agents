"""
Scan DataBento metadata catalog for probeable equity datasets.
"""
import re
from typing import Dict, List, Optional

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)

EQUITY_DATASET_KEYWORDS = (
    "EQUS", "DBEQ", "XNAS", "ARCX", "IEXG", "GLBX", "XBOS", "XPSX", "BATS",
)

PREFERRED_SCHEMAS = (
    "ohlcv-1d",
    "ohlcv-1h",
    "statistics",
)

OHLCV_SCHEMA_RE = re.compile(r"^ohlcv-(\d+)([smhd])$", re.IGNORECASE)
TRADING_MINUTES_PER_DAY = 390


def is_equity_dataset(dataset: str) -> bool:
    upper = dataset.upper()
    return any(kw in upper for kw in EQUITY_DATASET_KEYWORDS)


def _excluded_probe_schemas() -> frozenset:
    return frozenset(config.DISCOVERY_CONFIG.get("excluded_probe_schemas", ()))


def _ohlcv_interval_minutes(schema: str) -> Optional[float]:
    match = OHLCV_SCHEMA_RE.match(schema)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "s":
        return value / 60.0
    if unit == "m":
        return float(value)
    if unit == "h":
        return value * 60.0
    if unit == "d":
        return value * TRADING_MINUTES_PER_DAY
    return None


def _is_too_fine_ohlcv(schema: str) -> bool:
    interval = _ohlcv_interval_minutes(schema)
    if interval is None:
        return False
    min_bar = config.DISCOVERY_CONFIG.get("min_probe_bar_minutes", 15)
    return interval < min_bar


def is_probeable_schema(schema: str) -> bool:
    if schema in _excluded_probe_schemas():
        return False
    if schema.startswith("ohlcv-"):
        return not _is_too_fine_ohlcv(schema)
    return schema in PREFERRED_SCHEMAS


def probe_lookback_days(schema: str) -> int:
    """Schema-specific sample window for discovery probes."""
    cfg = config.DISCOVERY_CONFIG
    per_schema = cfg.get("schema_sample_days") or {}
    if schema in per_schema:
        return int(per_schema[schema])
    if schema.startswith("ohlcv-"):
        return int(cfg.get("intraday_sample_days_default", 15))
    return int(cfg.get("sample_days", 90))


def scan_catalog(client) -> List[Dict]:
    """
    Build a catalog of dataset/schema pairs relevant to ETF discovery.
    """
    entries: List[Dict] = []
    try:
        datasets = client.list_datasets()
    except Exception as e:
        logger.error(f"Failed to list DataBento datasets: {e}")
        return entries

    for dataset in datasets:
        if not is_equity_dataset(dataset):
            continue
        try:
            schemas = client.list_schemas(dataset)
        except Exception as e:
            logger.warning(f"Could not list schemas for {dataset}: {e}")
            continue

        for schema in schemas:
            if not is_probeable_schema(schema):
                continue
            entries.append({
                "dataset": dataset,
                "schema": schema,
                "probe_key": f"{dataset}|{schema}",
            })

    logger.info(f"Catalog scan: {len(entries)} probeable dataset/schema pairs")
    return entries
