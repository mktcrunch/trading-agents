"""
Read/write trading-agent JSON artifacts to Google Cloud Storage.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)

AUDIT_BLOB = "audit/audit_events.jsonl"


def load_jsonl_events(path: Path) -> List[Dict[str, Any]]:
    """Load audit events from a JSONL file."""
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def merge_audit_events(*event_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge audit event lists by event id (later lists win on duplicate ids).
    Sorted ascending by timestamp for stable JSONL storage.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for events in event_lists:
        for ev in events:
            key = str(ev.get("id") or json.dumps(ev, sort_keys=True, default=str))
            merged[key] = ev
    return sorted(merged.values(), key=lambda e: e.get("timestamp", ""))


def write_jsonl_events(path: Path, events: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev, default=str) + "\n")


class GCSStore:
    def __init__(
        self,
        audit_bucket: Optional[str] = None,
        data_bucket: Optional[str] = None,
    ):
        self.audit_bucket = audit_bucket or os.getenv("GCS_AUDIT_BUCKET", config.GCS_AUDIT_BUCKET or "")
        self.data_bucket = data_bucket or os.getenv(
            "GCS_DATA_BUCKET",
            os.getenv("GCS_RISK_STATE_BUCKET", config.GCS_DATA_BUCKET or config.GCS_RISK_STATE_BUCKET or ""),
        )
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.audit_bucket or self.data_bucket)

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        return self._client

    def _bucket(self, name: str):
        return self._get_client().bucket(name)

    def upload_file(self, local_path: Path, bucket: str, blob_name: str) -> bool:
        if not bucket or not local_path.exists():
            return False
        try:
            blob = self._bucket(bucket).blob(blob_name)
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded gs://{bucket}/{blob_name}")
            return True
        except Exception as e:
            logger.error(f"GCS upload failed gs://{bucket}/{blob_name}: {e}")
            return False

    def upload_json(self, data: Any, bucket: str, blob_name: str) -> bool:
        if not bucket:
            return False
        try:
            blob = self._bucket(bucket).blob(blob_name)
            blob.upload_from_string(
                json.dumps(data, indent=2, default=str),
                content_type="application/json",
            )
            logger.info(f"Uploaded gs://{bucket}/{blob_name}")
            return True
        except Exception as e:
            logger.error(f"GCS JSON upload failed gs://{bucket}/{blob_name}: {e}")
            return False

    def download_file(self, bucket: str, blob_name: str, local_path: Path) -> bool:
        if not bucket:
            return False
        try:
            blob = self._bucket(bucket).blob(blob_name)
            if not blob.exists():
                return False
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded gs://{bucket}/{blob_name} → {local_path}")
            return True
        except Exception as e:
            logger.warning(f"GCS download failed gs://{bucket}/{blob_name}: {e}")
            return False

    def blob_updated(self, bucket: str, blob_name: str) -> Optional[float]:
        if not bucket:
            return None
        try:
            blob = self._bucket(bucket).blob(blob_name)
            if not blob.exists():
                return None
            blob.reload()
            return blob.updated.timestamp() if blob.updated else None
        except Exception:
            return None

    def hydrate_audit_log(self) -> None:
        """Pull audit JSONL from GCS if cloud copy is newer or local is missing."""
        if not self.audit_bucket:
            return
        local = config.AUDIT_LOG_PATH
        gcs_ts = self.blob_updated(self.audit_bucket, AUDIT_BLOB)
        local_ts = local.stat().st_mtime if local.exists() else None
        if gcs_ts is None:
            return
        if local_ts is None or gcs_ts > local_ts:
            self.download_file(self.audit_bucket, AUDIT_BLOB, local)

    def _download_remote_audit_events(self) -> List[Dict[str, Any]]:
        """Fetch current audit JSONL from GCS without clobbering local file."""
        if not self.audit_bucket:
            return []
        try:
            blob = self._bucket(self.audit_bucket).blob(AUDIT_BLOB)
            if not blob.exists():
                return []
            text = blob.download_as_text()
        except Exception as e:
            logger.warning(f"GCS audit download for merge failed: {e}")
            return []

        events: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def sync_audit_log(self) -> bool:
        """
        Upload audit JSONL, merging with the latest GCS copy first.

        Baseline and Internal Agent Engines run in separate containers but share
        one audit blob. Merge-before-upload prevents last-writer-wins data loss.
        """
        local_path = config.AUDIT_LOG_PATH
        if not self.audit_bucket or not local_path.exists():
            return False

        local_events = load_jsonl_events(local_path)
        remote_events = self._download_remote_audit_events()
        merged = merge_audit_events(remote_events, local_events)

        if remote_events and len(merged) > len(local_events):
            logger.info(
                f"Merged audit log before upload: "
                f"local={len(local_events)} remote={len(remote_events)} "
                f"combined={len(merged)}"
            )

        write_jsonl_events(local_path, merged)
        return self.upload_file(local_path, self.audit_bucket, AUDIT_BLOB)

    def sync_all_local_data(self) -> Dict[str, bool]:
        """Upload every known local JSON artifact to GCS."""
        results: Dict[str, bool] = {}

        if config.AUDIT_LOG_PATH.exists():
            results["audit_events.jsonl"] = self.sync_audit_log()

        if not self.data_bucket:
            return results

        from src.learning.store import ALL_LEARNING_ROLES

        data_files = [
            (config.DATA_DIR / "approved_datasources.json", "data/approved_datasources.json"),
            (config.DATA_DIR / "discovery_registry.json", "data/discovery_registry.json"),
            (config.DATA_DIR / "risk_state_baseline.json", "risk_state/baseline.json"),
            (config.DATA_DIR / "risk_state_internal.json", "risk_state/internal.json"),
        ]
        for role in ALL_LEARNING_ROLES:
            for system in ("baseline", "internal"):
                if role == "discovery" and system != "internal":
                    continue
                local_path = config.DATA_DIR / f"learning_{role}_{system}.json"
                data_files.append((local_path, f"learning/{role}_{system}.json"))
        for local_path, blob_name in data_files:
            if local_path.exists():
                results[local_path.name] = self.upload_file(
                    local_path, self.data_bucket, blob_name
                )
            else:
                results[local_path.name] = False

        return results

    def hydrate_all_local_data(self) -> Dict[str, bool]:
        """Download JSON artifacts from GCS when cloud copy is newer or local missing."""
        results: Dict[str, bool] = {}
        self.hydrate_audit_log()
        results["audit_events.jsonl"] = config.AUDIT_LOG_PATH.exists()

        if not self.data_bucket:
            return results

        from src.learning.store import ALL_LEARNING_ROLES

        mappings = [
            (config.DATA_DIR / "approved_datasources.json", "data/approved_datasources.json"),
            (config.DATA_DIR / "discovery_registry.json", "data/discovery_registry.json"),
            (config.DATA_DIR / "risk_state_baseline.json", "risk_state/baseline.json"),
            (config.DATA_DIR / "risk_state_internal.json", "risk_state/internal.json"),
        ]
        for role in ALL_LEARNING_ROLES:
            for system in ("baseline", "internal"):
                if role == "discovery" and system != "internal":
                    continue
                local_path = config.DATA_DIR / f"learning_{role}_{system}.json"
                mappings.append((local_path, f"learning/{role}_{system}.json"))
        for local_path, blob_name in mappings:
            gcs_ts = self.blob_updated(self.data_bucket, blob_name)
            local_ts = local_path.stat().st_mtime if local_path.exists() else None
            if gcs_ts is not None and (local_ts is None or gcs_ts > local_ts):
                results[local_path.name] = self.download_file(
                    self.data_bucket, blob_name, local_path
                )
            else:
                results[local_path.name] = local_path.exists()

        return results


_store: Optional[GCSStore] = None


def get_gcs_store() -> GCSStore:
    global _store
    if _store is None:
        _store = GCSStore()
    return _store
