#!/usr/bin/env python3
"""
Upload all local trading-agent JSON history to GCS.
Run after deploy/setup_gcs.sh creates buckets, or anytime to backfill.

Usage:
    export GCS_AUDIT_BUCKET=mktcrunch-trading-agents-audit
    export GCS_DATA_BUCKET=mktcrunch-trading-agents-data
    python scripts/sync_history_to_gcs.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gcs.store import get_gcs_store
from src import config


def main() -> int:
    store = get_gcs_store()
    if not store.enabled:
        print("Set GCS_AUDIT_BUCKET and/or GCS_DATA_BUCKET in the environment.")
        return 1

    print(f"Data directory: {config.DATA_DIR}")
    print(f"Audit log:      {config.AUDIT_LOG_PATH} (exists={config.AUDIT_LOG_PATH.exists()})")
    print(f"Audit bucket:   {store.audit_bucket or '(none)'}")
    print(f"Data bucket:    {store.data_bucket or '(none)'}")
    print()

    results = store.sync_all_local_data()
    ok = sum(1 for v in results.values() if v)
    total = len(results)
    for name, success in results.items():
        status = "uploaded" if success else "skipped"
        print(f"  {name}: {status}")

    print(f"\nSynced {ok}/{total} files.")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
