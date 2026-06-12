"""GCS audit JSONL merge (concurrent baseline/internal writers)."""
from pathlib import Path

from src.gcs.store import load_jsonl_events, merge_audit_events, write_jsonl_events


def _ev(eid: str, system: str, ts: str, event_type: str = "ledger_decision"):
    return {
        "id": eid,
        "system": system,
        "timestamp": ts,
        "event_type": event_type,
        "action": "BUY SPY",
        "payload": {},
    }


def test_merge_combines_disjoint_events():
    baseline = [_ev("b1", "baseline", "2026-06-12T00:37:02Z")]
    internal = [_ev("i1", "internal", "2026-06-12T00:38:19Z", "agent_action")]
    merged = merge_audit_events(baseline, internal)
    assert len(merged) == 2
    assert merged[0]["id"] == "b1"
    assert merged[1]["id"] == "i1"


def test_merge_preserves_remote_when_local_is_subset():
    """Simulates internal uploading after baseline already synced to GCS."""
    remote = [
        _ev("b1", "baseline", "2026-06-12T00:37:02Z"),
        _ev("b2", "baseline", "2026-06-12T00:37:03Z", "order_placed"),
    ]
    local_internal_only = [_ev("i1", "internal", "2026-06-12T00:38:19Z")]
    merged = merge_audit_events(remote, local_internal_only)
    assert len(merged) == 3
    assert {e["system"] for e in merged} == {"baseline", "internal"}


def test_merge_local_wins_duplicate_id():
    remote = [_ev("x1", "baseline", "2026-06-12T00:37:00Z")]
    local = [{"id": "x1", "system": "baseline", "timestamp": "2026-06-12T00:37:01Z", "action": "updated"}]
    merged = merge_audit_events(remote, local)
    assert len(merged) == 1
    assert merged[0]["action"] == "updated"


def test_roundtrip_jsonl(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    events = [_ev("a", "baseline", "2026-06-12T00:37:00Z")]
    write_jsonl_events(path, events)
    assert load_jsonl_events(path) == events
