"""
Registry of DataBento dataset probes — memory for the discovery agent.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)

REGISTRY_PATH = config.DATA_DIR / "discovery_registry.json"


def _empty_registry() -> Dict:
    return {"probes": {}, "daily_runs": []}


def load_registry() -> Dict:
    try:
        from src.gcs.store import get_gcs_store

        store = get_gcs_store()
        if store.data_bucket:
            blob = "data/discovery_registry.json"
            gcs_ts = store.blob_updated(store.data_bucket, blob)
            local_ts = REGISTRY_PATH.stat().st_mtime if REGISTRY_PATH.exists() else None
            if gcs_ts is not None and (local_ts is None or gcs_ts > local_ts):
                store.download_file(store.data_bucket, blob, REGISTRY_PATH)
    except Exception:
        pass

    if not REGISTRY_PATH.exists():
        return _empty_registry()
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load discovery registry: {e}")
        return _empty_registry()


def save_registry(registry: Dict) -> Path:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    try:
        from src.gcs.store import get_gcs_store

        store = get_gcs_store()
        if store.data_bucket:
            store.upload_file(REGISTRY_PATH, store.data_bucket, "data/discovery_registry.json")
    except Exception:
        pass
    return REGISTRY_PATH


def probe_key(dataset: str, schema: str) -> str:
    return f"{dataset}|{schema}"


def get_probe(registry: Dict, dataset: str, schema: str) -> Dict:
    return registry.get("probes", {}).get(probe_key(dataset, schema), {})


def hours_since(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    ts = datetime.fromisoformat(iso_ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600


def record_probe(
    registry: Dict,
    dataset: str,
    schema: str,
    status: str,
    approved_count: int = 0,
    best_ic: float = 0.0,
    sample_rows: int = 0,
    error: Optional[str] = None,
    rationale: Optional[str] = None,
    proposed_features: Optional[List[Dict]] = None,
    approved_feature_ids: Optional[List[str]] = None,
    feature_strategy: Optional[str] = None,
) -> Dict:
    """Update registry after probing a dataset/schema."""
    key = probe_key(dataset, schema)
    existing = registry.setdefault("probes", {}).get(key, {})
    entry = {
        "dataset": dataset,
        "schema": schema,
        "last_probed_at": datetime.now(timezone.utc).isoformat(),
        "last_status": status,
        "approved_count": approved_count,
        "best_ic": best_ic,
        "sample_rows": sample_rows,
        "probe_count": existing.get("probe_count", 0) + 1,
        "last_error": error,
        "last_rationale": rationale,
        "first_probed_at": existing.get("first_probed_at")
        or datetime.now(timezone.utc).isoformat(),
    }
    if proposed_features is not None:
        entry["proposed_features"] = proposed_features[-20:]
    if approved_feature_ids is not None:
        entry["approved_feature_ids"] = approved_feature_ids
    if feature_strategy:
        entry["last_feature_strategy"] = feature_strategy
    registry["probes"][key] = entry
    return registry


def append_daily_run(registry: Dict, summary: Dict) -> Dict:
    runs = registry.setdefault("daily_runs", [])
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    runs.append(summary)
    registry["daily_runs"] = runs[-30:]
    return registry


def never_probed(registry: Dict, dataset: str, schema: str) -> bool:
    return probe_key(dataset, schema) not in registry.get("probes", {})


def catalog_entry_summary(catalog: List[Dict], registry: Dict) -> List[Dict]:
    """Attach probe history to catalog for the LLM planner."""
    enriched = []
    for entry in catalog:
        key = entry["probe_key"]
        probe = registry.get("probes", {}).get(key, {})
        enriched.append({
            **entry,
            "probed_before": bool(probe),
            "last_status": probe.get("last_status"),
            "last_probed_hours_ago": hours_since(probe.get("last_probed_at")),
            "approved_count": probe.get("approved_count", 0),
            "best_ic": probe.get("best_ic", 0),
        })
    return enriched
