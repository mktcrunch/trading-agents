"""
Persistent trailing-stop state across Cloud Run invocations.
Uses local JSON by default; optional GCS bucket for serverless persistence.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)


def _empty_state() -> Dict[str, Any]:
    return {
        "trailing_stops": {},
        "eod_exit_done_date": None,
        "updated_at": None,
    }


class RiskStateStore:
    """Load/save per-system risk monitor state."""

    def __init__(self, system: str):
        self.system = system
        self.local_path = config.DATA_DIR / f"risk_state_{system}.json"
        self.bucket = os.getenv("GCS_RISK_STATE_BUCKET")

    def _gcs_blob_name(self) -> str:
        return f"risk_state/{self.system}.json"

    def _load_gcs(self) -> Optional[Dict]:
        if not self.bucket:
            return None
        try:
            from google.cloud import storage

            client = storage.Client()
            blob = client.bucket(self.bucket).blob(self._gcs_blob_name())
            if not blob.exists():
                return None
            return json.loads(blob.download_as_text())
        except Exception as e:
            logger.warning(f"GCS risk state load failed ({self.system}): {e}")
            return None

    def _save_gcs(self, state: Dict) -> bool:
        if not self.bucket:
            return False
        try:
            from google.cloud import storage

            client = storage.Client()
            blob = client.bucket(self.bucket).blob(self._gcs_blob_name())
            blob.upload_from_string(
                json.dumps(state, indent=2),
                content_type="application/json",
            )
            return True
        except Exception as e:
            logger.warning(f"GCS risk state save failed ({self.system}): {e}")
            return False

    def load(self) -> Dict[str, Any]:
        state = self._load_gcs()
        if state:
            return state

        if self.local_path.exists():
            try:
                with open(self.local_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load local risk state: {e}")

        return _empty_state()

    def save(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.local_path, "w") as f:
            json.dump(state, f, indent=2)

        if self._save_gcs(state):
            logger.info(f"Risk state synced to GCS for {self.system}")
