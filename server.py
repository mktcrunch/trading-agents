"""
Cloud Run HTTP entry — scheduler jobs + audit dashboard API.
"""
import asyncio
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src import config
from src.audit.store import get_performance, get_summary, get_trace, load_events
from src.logger import setup_logger

logger = setup_logger(__name__)

DASHBOARD_HTML = Path(__file__).parent / "src" / "dashboard" / "index.html"


def _check_auth(handler: BaseHTTPRequestHandler) -> bool:
    secret = config.SCHEDULER_SECRET
    if not secret:
        return True
    return handler.headers.get("X-Scheduler-Secret") == secret


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    payload = json.dumps(body, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


def _html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    payload = html.encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    handler.end_headers()
    handler.wfile.write(payload)


class JobHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _query(self) -> dict:
        parsed = urlparse(self.path)
        return {k: v[0] for k, v in parse_qs(parsed.query).items()}

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            _json_response(self, 200, {"status": "ok"})
            return

        if path in ("/", "/dashboard"):
            if DASHBOARD_HTML.exists():
                _html_response(self, DASHBOARD_HTML.read_text())
            else:
                _json_response(self, 404, {"error": "dashboard not found"})
            return

        if path == "/api/summary":
            q = self._query()
            hours = int(q.get("hours", 24))
            _json_response(self, 200, get_summary(since_hours=hours))
            return

        if path == "/api/performance":
            q = self._query()
            hours = int(q.get("hours", 168))
            try:
                _json_response(self, 200, get_performance(since_hours=hours))
            except Exception as e:
                logger.exception("Performance API failed")
                _json_response(self, 500, {"error": str(e), "live": None, "history": {}})
            return

        if path == "/api/events":
            q = self._query()
            events = load_events(
                limit=int(q.get("limit", 200)),
                system=q.get("system") or None,
                event_type=q.get("event_type") or None,
                since_hours=int(q["hours"]) if q.get("hours") else None,
            )
            _json_response(self, 200, {"events": events, "count": len(events)})
            return

        if path.startswith("/api/trace/"):
            trace_id = path.split("/api/trace/")[-1]
            _json_response(self, 200, get_trace(trace_id))
            return

        if path == "/api/chat/status":
            from src.dashboard.coordinator_chat import chat_status

            _json_response(self, 200, chat_status())
            return

        _json_response(self, 404, {"error": "not found"})

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode() or "{}")

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/chat":
            try:
                from src.dashboard.coordinator_chat import run_coordinator_chat

                body = self._read_json_body()
                result = asyncio.run(
                    run_coordinator_chat(
                        system=body.get("system", "baseline"),
                        message=body.get("message", ""),
                        session_id=body.get("session_id"),
                        user_id=body.get("user_id"),
                    )
                )
                status = 200 if result.get("success") else 400
                _json_response(self, status, result)
            except Exception as e:
                logger.exception("Chat API failed")
                _json_response(self, 500, {"success": False, "error": str(e)})
            return

        if not _check_auth(self):
            _json_response(self, 401, {"error": "unauthorized"})
            return

        dry_run = self.headers.get("X-Dry-Run", "").lower() == "true"

        try:
            if path == "/jobs/overnight":
                from main import run_overnight_job
                ok = asyncio.run(run_overnight_job())
                _json_response(self, 200, {"job": "overnight", "success": ok})
            elif path == "/jobs/risk":
                from main import run_risk_job
                ok = asyncio.run(run_risk_job(system="both", dry_run=dry_run))
                _json_response(self, 200, {"job": "risk", "success": ok})
            elif path == "/jobs/risk/baseline":
                from main import run_risk_job
                ok = asyncio.run(run_risk_job(system="baseline", dry_run=dry_run))
                _json_response(self, 200, {"job": "risk_baseline", "success": ok})
            elif path == "/jobs/risk/internal":
                from main import run_risk_job
                ok = asyncio.run(run_risk_job(system="internal", dry_run=dry_run))
                _json_response(self, 200, {"job": "risk_internal", "success": ok})
            else:
                _json_response(self, 404, {"error": "not found"})
        except Exception as e:
            logger.exception("Job failed")
            _json_response(self, 500, {"error": str(e)})


def run_server():
    try:
        from src.gcs.store import get_gcs_store

        store = get_gcs_store()
        if store.enabled:
            logger.info("Hydrating local data from GCS...")
            store.hydrate_all_local_data()
    except Exception as e:
        logger.warning(f"GCS hydrate on startup skipped: {e}")

    port = config.CLOUD_RUN_PORT
    server = HTTPServer(("0.0.0.0", port), JobHandler)
    logger.info(f"Trading agents server listening on :{port}")
    logger.info(f"Dashboard: http://0.0.0.0:{port}/dashboard")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
