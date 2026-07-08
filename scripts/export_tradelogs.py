#!/usr/bin/env python3
"""Export Twin Ledger trade logs for baseline and internal since FIRST_TRADE_DATE."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config

AUDIT_TRADE_EVENT_TYPES = frozenset({
    "ledger_decision",
    "order_placed",
    "order_skipped",
    "order_cancelled_duplicate",
    "risk_rejected",
    "risk_stop_exit",
    "risk_eod_exit",
    "arena_decision",
})

AUDIT_RISK_EVENT_TYPES = frozenset({
    "risk_rejected",
    "risk_stop_exit",
    "risk_eod_exit",
    "risk_held",
    "risk_positions_checked",
    "trailing_stop_planned",
    "trailing_stop_init",
    "trailing_stop_update",
    "base_stop_planned",
    "order_cancelled_risk_exit",
})


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _order_to_dict(order: Any) -> Dict[str, Any]:
    def _s(val: Any) -> Optional[str]:
        if val is None:
            return None
        return str(val)

    return {
        "id": _s(order.id),
        "client_order_id": _s(order.client_order_id),
        "created_at": _s(order.created_at),
        "updated_at": _s(order.updated_at),
        "submitted_at": _s(order.submitted_at),
        "filled_at": _s(order.filled_at),
        "expired_at": _s(getattr(order, "expired_at", None)),
        "canceled_at": _s(getattr(order, "canceled_at", None)),
        "symbol": _s(order.symbol),
        "side": _s(order.side),
        "type": _s(order.type),
        "qty": _s(order.qty),
        "filled_qty": _s(order.filled_qty),
        "limit_price": _s(order.limit_price),
        "filled_avg_price": _s(order.filled_avg_price),
        "status": _s(order.status),
        "time_in_force": _s(order.time_in_force),
        "extended_hours": bool(getattr(order, "extended_hours", False)),
    }


def fetch_alpaca_orders(system: str, since: datetime) -> List[Dict[str, Any]]:
    from alpaca.common.enums import Sort
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    from src.apis.alpaca_client import AlpacaClient

    client = AlpacaClient(system=system).client
    end = datetime.now(timezone.utc)
    all_orders: List[Any] = []
    after = since
    while True:
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=100,
            after=after,
            until=end,
            direction=Sort.ASC,
        )
        batch = client.get_orders(request)
        if not batch:
            break
        all_orders.extend(batch)
        if len(batch) < 100:
            break
        after = batch[-1].created_at
    return [_order_to_dict(o) for o in all_orders]


def hydrate_audit(*, force_gcs: bool = False) -> Path:
    if force_gcs:
        try:
            from src.gcs.store import get_gcs_store

            store = get_gcs_store()
            if store.audit_bucket:
                local = config.AUDIT_LOG_PATH
                local.parent.mkdir(parents=True, exist_ok=True)
                store.download_file(store.audit_bucket, "audit/audit_events.jsonl", local)
        except Exception as exc:
            print(f"Warning: forced GCS audit download failed: {exc}", file=sys.stderr)
    else:
        try:
            from src.gcs.store import get_gcs_store

            get_gcs_store().hydrate_audit_log()
        except Exception:
            pass
    return config.AUDIT_LOG_PATH


def iter_audit_events(
    log_path: Path,
    *,
    system: str,
    since: datetime,
    event_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []

    allowed = set(event_types) if event_types else None
    events: List[Dict[str, Any]] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("system") not in (system, "both"):
                continue
            if allowed and ev.get("event_type") not in allowed:
                continue
            try:
                if _parse_ts(ev["timestamp"]) < since:
                    continue
            except (KeyError, ValueError):
                continue
            events.append(ev)
    events.sort(key=lambda e: e.get("timestamp", ""))
    return events


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def write_orders_csv(path: Path, orders: List[Dict[str, Any]], system: str) -> None:
    fields = [
        "system",
        "created_at",
        "filled_at",
        "symbol",
        "side",
        "status",
        "qty",
        "filled_qty",
        "limit_price",
        "filled_avg_price",
        "time_in_force",
        "id",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in orders:
            writer.writerow({"system": system, **row})


def _payload_get(payload: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def write_risk_csv(path: Path, events: List[Dict[str, Any]], system: str) -> None:
    fields = [
        "system",
        "timestamp",
        "trace_id",
        "event_type",
        "agent",
        "action",
        "status",
        "ticker",
        "side",
        "return_pct",
        "current_return",
        "activation_threshold",
        "profit_lock_fraction",
        "stop_price",
        "stop_loss_threshold",
        "policy",
        "rationale",
        "reason",
        "payload_json",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for ev in events:
            payload = ev.get("payload") or {}
            extra = {
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "ticker",
                    "side",
                    "return_pct",
                    "current_return",
                    "activation_threshold",
                    "profit_lock_fraction",
                    "stop_price",
                    "stop_loss_threshold",
                    "policy",
                    "rationale",
                    "reason",
                }
            }
            writer.writerow({
                "system": system,
                "timestamp": ev.get("timestamp"),
                "trace_id": ev.get("trace_id"),
                "event_type": ev.get("event_type"),
                "agent": ev.get("agent"),
                "action": ev.get("action"),
                "status": ev.get("status"),
                "ticker": _payload_get(payload, "ticker"),
                "side": _payload_get(payload, "side"),
                "return_pct": _payload_get(payload, "return_pct"),
                "current_return": _payload_get(payload, "current_return"),
                "activation_threshold": _payload_get(payload, "activation_threshold"),
                "profit_lock_fraction": _payload_get(payload, "profit_lock_fraction"),
                "stop_price": _payload_get(payload, "stop_price"),
                "stop_loss_threshold": _payload_get(payload, "stop_loss_threshold"),
                "policy": _payload_get(payload, "policy"),
                "rationale": _payload_get(payload, "rationale"),
                "reason": _payload_get(payload, "reason"),
                "payload_json": json.dumps(extra, default=str) if extra else "",
            })


def write_decisions_csv(path: Path, events: List[Dict[str, Any]], system: str) -> None:
    fields = [
        "system",
        "timestamp",
        "trace_id",
        "event_type",
        "ticker",
        "action",
        "size_pct",
        "confidence",
        "rationale",
        "invalidation",
        "competitive_note",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for ev in events:
            if ev.get("event_type") != "ledger_decision":
                continue
            payload = ev.get("payload") or {}
            writer.writerow({
                "system": system,
                "timestamp": ev.get("timestamp"),
                "trace_id": ev.get("trace_id"),
                "event_type": ev.get("event_type"),
                "ticker": payload.get("ticker"),
                "action": payload.get("action"),
                "size_pct": payload.get("size_pct"),
                "confidence": payload.get("confidence"),
                "rationale": payload.get("rationale"),
                "invalidation": payload.get("invalidation"),
                "competitive_note": payload.get("competitive_note"),
            })


def export_tradelogs(out_dir: Path, since_date: str, *, force_gcs: bool = False) -> Dict[str, Any]:
    since = datetime.fromisoformat(since_date).replace(tzinfo=timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = hydrate_audit(force_gcs=force_gcs)

    summary: Dict[str, Any] = {
        "since_date": since_date,
        "output_dir": str(out_dir),
        "systems": {},
    }

    for system in ("baseline", "internal"):
        system_dir = out_dir / system
        system_dir.mkdir(parents=True, exist_ok=True)

        orders = fetch_alpaca_orders(system, since)
        write_json(system_dir / "alpaca_orders.json", orders)
        write_orders_csv(system_dir / "alpaca_orders.csv", orders, system)

        audit_events = iter_audit_events(
            audit_path,
            system=system,
            since=since,
            event_types=AUDIT_TRADE_EVENT_TYPES,
        )
        write_jsonl(system_dir / "audit_trades.jsonl", audit_events)
        write_decisions_csv(system_dir / "audit_decisions.csv", audit_events, system)

        risk_events = iter_audit_events(
            audit_path,
            system=system,
            since=since,
            event_types=AUDIT_RISK_EVENT_TYPES,
        )
        write_jsonl(system_dir / "audit_risk.jsonl", risk_events)
        write_risk_csv(system_dir / "audit_risk.csv", risk_events, system)

        filled = [o for o in orders if (o.get("status") or "").endswith("FILLED")]
        decisions = [e for e in audit_events if e.get("event_type") == "ledger_decision"]
        placed = [e for e in audit_events if e.get("event_type") == "order_placed"]
        risk_by_type = {
            event_type: sum(1 for e in risk_events if e.get("event_type") == event_type)
            for event_type in sorted(AUDIT_RISK_EVENT_TYPES)
            if any(e.get("event_type") == event_type for e in risk_events)
        }

        summary["systems"][system] = {
            "alpaca_orders": len(orders),
            "alpaca_filled_orders": len(filled),
            "audit_trade_events": len(audit_events),
            "ledger_decisions": len(decisions),
            "orders_placed_audit": len(placed),
            "audit_risk_events": len(risk_events),
            "audit_risk_by_type": risk_by_type,
            "files": {
                "alpaca_orders_json": str(system_dir / "alpaca_orders.json"),
                "alpaca_orders_csv": str(system_dir / "alpaca_orders.csv"),
                "audit_trades_jsonl": str(system_dir / "audit_trades.jsonl"),
                "audit_decisions_csv": str(system_dir / "audit_decisions.csv"),
                "audit_risk_jsonl": str(system_dir / "audit_risk.jsonl"),
                "audit_risk_csv": str(system_dir / "audit_risk.csv"),
            },
        }

    write_json(out_dir / "export_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        default=config.FIRST_TRADE_DATE,
        help=f"Start date YYYY-MM-DD (default: {config.FIRST_TRADE_DATE})",
    )
    parser.add_argument(
        "--out",
        default=str(config.DATA_DIR / "exports" / "tradelogs"),
        help="Output directory",
    )
    parser.add_argument(
        "--force-gcs",
        action="store_true",
        help="Always re-download audit/audit_events.jsonl from GCS before export",
    )
    args = parser.parse_args()
    summary = export_tradelogs(Path(args.out), args.since, force_gcs=args.force_gcs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
