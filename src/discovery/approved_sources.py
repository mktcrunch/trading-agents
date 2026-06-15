"""
Load/save approved DataBento discovery output.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)

APPROVED_SOURCES_PATH = config.DATA_DIR / "approved_datasources.json"


def save_approved_sources(payload: Dict) -> Path:
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    APPROVED_SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(APPROVED_SOURCES_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved approved sources to {APPROVED_SOURCES_PATH}")
    try:
        from src.gcs.store import get_gcs_store

        store = get_gcs_store()
        if store.data_bucket:
            store.upload_file(
                APPROVED_SOURCES_PATH, store.data_bucket, "data/approved_datasources.json"
            )
    except Exception:
        pass
    return APPROVED_SOURCES_PATH


def load_approved_sources() -> Dict:
    try:
        from src.gcs.store import get_gcs_store

        store = get_gcs_store()
        if store.data_bucket:
            blob = "data/approved_datasources.json"
            gcs_ts = store.blob_updated(store.data_bucket, blob)
            local_ts = (
                APPROVED_SOURCES_PATH.stat().st_mtime if APPROVED_SOURCES_PATH.exists() else None
            )
            if gcs_ts is not None and (local_ts is None or gcs_ts > local_ts):
                store.download_file(store.data_bucket, blob, APPROVED_SOURCES_PATH)
    except Exception:
        pass

    if not APPROVED_SOURCES_PATH.exists():
        return {}
    try:
        with open(APPROVED_SOURCES_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load approved sources: {e}")
        return {}


def has_usable_cached_sources(data: Optional[Dict] = None) -> bool:
    """True when GCS/local approved sources have sources or per-ticker features."""
    payload = load_approved_sources() if data is None else data
    if not payload:
        return False
    return bool(payload.get("sources") or payload.get("ticker_features"))


def discovery_run_meta(
    data: Dict,
    *,
    mode: str,
    error: str = "",
) -> Dict:
    """Normalized discovery summary for overnight workflow state/audit."""
    summary = data.get("summary") or {}
    sources = data.get("sources") or []
    features = data.get("ticker_features") or {}
    return {
        "success": mode in ("cached", "refreshed", "cache_fallback"),
        "mode": mode,
        "refreshed": False,
        "approved_count": summary.get("approved_count") or len(sources),
        "tickers_with_features": summary.get("tickers_with_features") or len(features),
        "probes_run": 0,
        "generated_at": data.get("generated_at"),
        "error": error or None,
    }


def is_stale(max_age_hours: Optional[float] = None) -> bool:
    data = load_approved_sources()
    if not data or not data.get("generated_at"):
        return True

    max_age = max_age_hours or config.DISCOVERY_CONFIG.get("max_age_hours", 24)
    generated = datetime.fromisoformat(data["generated_at"])
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - generated).total_seconds() / 3600
    return age_hours > max_age


def _source_key(source: Dict) -> str:
    return source.get("id") or f"{source.get('dataset')}|{source.get('feature')}"


def merge_probe_results(
    existing_sources: list,
    new_sources: list,
    existing_features: Dict,
    new_features: Dict,
) -> tuple[list, Dict]:
    """
    Merge a new probe into accumulated discovery state.
    Keeps the higher-|IC| source when feature ids collide.
    """
    by_id = {_source_key(s): s for s in existing_sources}
    for source in new_sources:
        key = _source_key(source)
        prev = by_id.get(key)
        if not prev:
            by_id[key] = source
            continue
        prev_ic = abs(prev.get("metrics", {}).get("ic", 0))
        new_ic = abs(source.get("metrics", {}).get("ic", 0))
        if new_ic >= prev_ic:
            by_id[key] = source

    merged_features: Dict[str, Dict] = {}
    all_tickers = set(existing_features) | set(new_features)
    for ticker in all_tickers:
        merged = dict(existing_features.get(ticker, {}))
        merged.update(new_features.get(ticker, {}))
        if merged:
            merged_features[ticker] = merged

    return list(by_id.values()), merged_features


def enrich_tickers(tickers: list[str], approved: Optional[Dict] = None) -> Dict[str, Dict]:
    """Map tickers to latest discovered DataBento feature values."""
    approved = approved or load_approved_sources()
    ticker_features = approved.get("ticker_features", {})
    if not ticker_features:
        return {}

    source_ids = [s.get("id") for s in approved.get("sources", [])]
    enriched = {}
    for ticker in tickers:
        features = ticker_features.get(ticker)
        if features:
            enriched[ticker] = {
                "databento_features": features,
                "approved_source_ids": source_ids,
                "gate_approved": bool(source_ids),
                "discovery_generated_at": approved.get("generated_at"),
            }
    return enriched
