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

    def sync_audit_log(self) -> bool:
        if not self.audit_bucket or not config.AUDIT_LOG_PATH.exists():
            return False
        return self.upload_file(config.AUDIT_LOG_PATH, self.audit_bucket, AUDIT_BLOB)

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
