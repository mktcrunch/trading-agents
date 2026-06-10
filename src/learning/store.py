"""Persist per-agent learning state (local JSON + optional GCS)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src import config
from src.learning.roles import INTERNAL_ONLY_LEARNING_ROLES, TRADER_LEARNING_ROLES
from src.logger import setup_logger

logger = setup_logger(__name__)

ALL_LEARNING_ROLES = TRADER_LEARNING_ROLES + INTERNAL_ONLY_LEARNING_ROLES


def _empty_state(role: str, system: str) -> Dict[str, Any]:
    return {
        "system": system,
        "agent_role": role,
        "updated_at": None,
        "lessons_learned": "",
        "bad_patterns": [],
        "do_more": [],
        "scorecard": {},
        "recent_events": [],
    }


class LearningStore:
    def __init__(self, system: str, role: str):
        if role not in ALL_LEARNING_ROLES:
            raise ValueError(f"Invalid role: {role}")
        if role in INTERNAL_ONLY_LEARNING_ROLES:
            if system != "internal":
                raise ValueError("Discovery learning is Internal-only")
        elif system not in ("baseline", "internal"):
            raise ValueError(f"Invalid system: {system}")
        self.system = system
        self.role = role
        self.local_path = config.DATA_DIR / f"learning_{role}_{system}.json"
        self.bucket = os.getenv("GCS_DATA_BUCKET", config.GCS_DATA_BUCKET or "")

    def _blob_name(self) -> str:
        return f"learning/{self.role}_{self.system}.json"

    def _hydrate_from_gcs(self) -> None:
        if not self.bucket:
            return
        try:
            from src.gcs.store import get_gcs_store

            store = get_gcs_store()
            if store.data_bucket:
                gcs_ts = store.blob_updated(store.data_bucket, self._blob_name())
                local_ts = self.local_path.stat().st_mtime if self.local_path.exists() else None
                if gcs_ts is not None and (local_ts is None or gcs_ts > local_ts):
                    store.download_file(store.data_bucket, self._blob_name(), self.local_path)
        except Exception as e:
            logger.debug(f"Learning hydrate skipped ({self.role}/{self.system}): {e}")

    def load(self) -> Dict[str, Any]:
        self._hydrate_from_gcs()
        if self.local_path.exists():
            try:
                with open(self.local_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load learning state: {e}")
        return _empty_state(self.role, self.system)

    def save(self, state: Dict[str, Any]) -> None:
        state["system"] = self.system
        state["agent_role"] = self.role
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.local_path, "w") as f:
            json.dump(state, f, indent=2)

        if self.bucket:
            try:
                from src.gcs.store import get_gcs_store

                get_gcs_store().upload_file(
                    self.local_path, self.bucket, self._blob_name()
                )
            except Exception as e:
                logger.warning(f"Learning GCS sync failed: {e}")


def load_learning(system: str, role: str) -> Dict[str, Any]:
    return LearningStore(system, role).load()


def save_learning(system: str, role: str, state: Dict[str, Any]) -> None:
    LearningStore(system, role).save(state)


def load_all_learning(system: str) -> Dict[str, Dict[str, Any]]:
    from src.learning.roles import roles_for_system

    return {role: load_learning(system, role) for role in roles_for_system(system)}
