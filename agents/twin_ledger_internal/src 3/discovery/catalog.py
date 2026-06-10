"""
Scan DataBento metadata catalog for probeable equity datasets.
"""
from typing import Dict, List

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)

EQUITY_DATASET_KEYWORDS = (
    "EQUS", "DBEQ", "XNAS", "ARCX", "IEXG", "GLBX", "XBOS", "XPSX", "BATS",
)

PREFERRED_SCHEMAS = (
    "ohlcv-1d",
    "ohlcv-1h",
    "ohlcv-1m",
    "statistics",
)


def is_equity_dataset(dataset: str) -> bool:
    upper = dataset.upper()
    return any(kw in upper for kw in EQUITY_DATASET_KEYWORDS)


def is_probeable_schema(schema: str) -> bool:
    if schema in PREFERRED_SCHEMAS:
        return True
    return schema.startswith("ohlcv")


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
