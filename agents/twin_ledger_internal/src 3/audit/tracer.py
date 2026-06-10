"""
Append-only audit event log with trace IDs for full action traceability.
"""
from __future__ import annotations

import contextvars
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)

_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)
_job_type: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("job_type", default=None)


class AuditTracer:
    """Records structured events to JSONL (+ optional GCS)."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or config.AUDIT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.bucket = os.getenv("GCS_AUDIT_BUCKET", config.GCS_AUDIT_BUCKET or "")

    def _append_local(self, event: Dict[str, Any]) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def _sync_gcs(self) -> None:
        if not self.bucket:
            return
        try:
            from src.gcs.store import get_gcs_store

            get_gcs_store().sync_audit_log()
        except Exception as e:
            logger.warning(f"GCS audit sync failed: {e}")

    def record(
        self,
        event_type: str,
        action: str,
        system: str = "both",
        agent: Optional[str] = None,
        status: str = "ok",
        payload: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = {
            "id": str(uuid.uuid4()),
            "trace_id": trace_id or _trace_id.get(),
            "job_type": _job_type.get(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": system,
            "agent": agent,
            "event_type": event_type,
            "action": action,
            "status": status,
            "payload": payload or {},
        }
        try:
            self._append_local(event)
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")
        return event

    def start_trace(self, job_type: str, system: str = "both", meta: Optional[Dict] = None) -> str:
        existing = _trace_id.get()
        if existing:
            self.record(
                event_type="sub_job_started",
                action=job_type,
                system=system,
                agent="orchestrator",
                payload=meta or {},
                trace_id=existing,
            )
            return existing

        trace_id = str(uuid.uuid4())[:12]
        _trace_id.set(trace_id)
        _job_type.set(job_type)
        self.record(
            event_type="job_started",
            action=job_type,
            system=system,
            agent="orchestrator",
            payload=meta or {},
            trace_id=trace_id,
        )
        return trace_id

    def end_trace(self, job_type: str, system: str = "both", success: bool = True, summary: Optional[Dict] = None) -> None:
        is_root = _job_type.get() == job_type
        self.record(
            event_type="job_completed" if is_root else "sub_job_completed",
            action=job_type,
            system=system,
            agent="orchestrator",
            status="ok" if success else "error",
            payload=summary or {},
        )
        if is_root:
            self._sync_gcs()
            _trace_id.set(None)
            _job_type.set(None)


_tracer: Optional[AuditTracer] = None


def get_tracer() -> AuditTracer:
    global _tracer
    if _tracer is None:
        _tracer = AuditTracer()
    return _tracer


def record_event(**kwargs) -> Dict[str, Any]:
    return get_tracer().record(**kwargs)


def start_trace(job_type: str, system: str = "both", meta: Optional[Dict] = None) -> str:
    return get_tracer().start_trace(job_type, system, meta)


def end_trace(job_type: str, system: str = "both", success: bool = True, summary: Optional[Dict] = None) -> None:
    get_tracer().end_trace(job_type, system, success, summary)


def get_trace_id() -> Optional[str]:
    return _trace_id.get()


def get_job_type() -> Optional[str]:
    return _job_type.get()
