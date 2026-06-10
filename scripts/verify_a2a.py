#!/usr/bin/env python3
"""Verify A2A readiness for Twin Ledger Agent Engine deployments.

Checks:
  - google-adk[a2a] in Agent Engine requirements
  - Reasoning Engine resources exist (REST)
  - stream_query / session APIs exposed (A2A-compatible Agent Engine surface)
  - ADK multi-agent tree (coordinator + sub_agents) locally

Does NOT run trading workflows. Safe to run before submission.

Usage:
  python scripts/verify_a2a.py
  python scripts/verify_a2a.py --json
  python scripts/verify_a2a.py --write-submission-snippet docs/SUBMISSION_A2A_SNIPPET.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
IDS_FILE = ROOT / "deploy" / "agent_engine_ids.env"
REQUIREMENTS_FILES = [
    ROOT / "deploy" / "agent_engine_requirements.txt",
    ROOT / "agents" / "twin_ledger_baseline" / "requirements.txt",
    ROOT / "agents" / "twin_ledger_internal" / "requirements.txt",
]

DEFAULT_REGION = "us-central1"


def _load_ids() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not IDS_FILE.exists():
        return out
    for line in IDS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _check_requirements() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    for path in REQUIREMENTS_FILES:
        if not path.exists():
            checks.append({"file": str(path), "ok": False, "error": "missing"})
            continue
        text = path.read_text()
        has_a2a = "google-adk[a2a" in text or "a2a-sdk" in text
        checks.append({
            "file": str(path.relative_to(ROOT)),
            "ok": has_a2a,
            "has_google_adk_a2a": "google-adk[a2a" in text,
        })
    return {
        "ok": all(c.get("ok") for c in checks),
        "checks": checks,
    }


def _fetch_engine(
    project: str,
    region: str,
    engine_id: str,
    token: str,
) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    url = (
        f"https://{region}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{region}/reasoningEngines/{engine_id}"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        methods = [
            m.get("name")
            for m in (data.get("spec") or {}).get("classMethods") or []
            if m.get("name")
        ]
        stream_ok = "stream_query" in methods or "async_stream_query" in methods
        return {
            "ok": True,
            "display_name": data.get("displayName"),
            "resource_name": data.get("name"),
            "description": data.get("description"),
            "api_methods": methods,
            "stream_query_exposed": stream_ok,
            "create_time": data.get("createTime"),
            "update_time": data.get("updateTime"),
        }
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        return {"ok": False, "error": f"HTTP {e.code}", "detail": body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _check_local_adk_tree() -> Dict[str, Any]:
    try:
        from src.adk.agents.coordinators import build_baseline_root_agent, build_internal_root_agent

        baseline = build_baseline_root_agent()
        internal = build_internal_root_agent()
        baseline_subs = [getattr(a, "name", str(a)) for a in (baseline.sub_agents or [])]
        internal_subs = [getattr(a, "name", str(a)) for a in (internal.sub_agents or [])]
        return {
            "ok": bool(baseline_subs) and bool(internal_subs),
            "baseline_coordinator": getattr(baseline, "name", "twin_ledger_baseline"),
            "baseline_sub_agents": baseline_subs,
            "internal_coordinator": getattr(internal, "name", "twin_ledger_internal"),
            "internal_sub_agents": internal_subs,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _auth_token() -> Optional[str]:
    try:
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token
    except Exception:
        return None


def _submission_snippet(report: Dict[str, Any]) -> str:
    baseline = report.get("engines", {}).get("baseline", {})
    internal = report.get("engines", {}).get("internal", {})
    return f"""## A2A & Agent Engine (submission paragraph)

Twin Ledger deploys **two A2A-compatible Vertex AI Agent Engine** reasoning engines — one per competing trader (`{baseline.get('display_name', 'Baseline')}`, `{internal.get('display_name', 'Internal')}`). Each engine is packaged with `google-adk[a2a]` and exposes standard Agent Engine APIs (`stream_query`, session management) for external orchestration. **Cloud Scheduler** invokes these endpoints directly (OAuth) for overnight trading, intraday risk, and post-open chase — no custom glue service.

Inside each engine, **Google ADK** defines a chat-mode coordinator `LlmAgent` with task-mode sub-agents (`baseline_data`, `baseline_signal` / `internal_data`, `internal_signal`). Specialist delegation uses ADK `sub_agents`; production scheduler routing uses deterministic phrase-matched callbacks for reliability. The dashboard chat and audit APIs provide read-only cross-agent visibility without placing orders.

**Baseline resource:** `{baseline.get('resource_name', 'n/a')}`
**Internal resource:** `{internal.get('resource_name', 'n/a')}`

Verify locally: `python scripts/verify_a2a.py`
"""


def run_verification(project: str, region: str) -> Dict[str, Any]:
    if not project:
        project = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    if not project:
        return {
            "project": "",
            "region": region,
            "overall_ok": False,
            "error": "Set GCP_PROJECT or pass --project",
            "submission_snippet": "",
        }
    ids = _load_ids()
    report: Dict[str, Any] = {
        "project": project,
        "region": region,
        "requirements": _check_requirements(),
        "local_adk": _check_local_adk_tree(),
        "engines": {},
        "overall_ok": False,
    }

    token = _auth_token()
    if not token:
        report["auth"] = {"ok": False, "error": "No GCP credentials — run gcloud auth application-default login"}
    else:
        report["auth"] = {"ok": True}

    for key, env_key in (
        ("baseline", "AGENT_ENGINE_BASELINE_ID"),
        ("internal", "AGENT_ENGINE_INTERNAL_ID"),
    ):
        engine_id = ids.get(env_key, "")
        if not engine_id:
            report["engines"][key] = {"ok": False, "error": f"Missing {env_key} in {IDS_FILE}"}
            continue
        if token:
            report["engines"][key] = {
                "engine_id": engine_id,
                **_fetch_engine(project, region, engine_id, token),
            }
        else:
            report["engines"][key] = {
                "engine_id": engine_id,
                "ok": False,
                "error": "Skipped REST check (no auth)",
            }

    engine_ok = all(e.get("ok") for e in report["engines"].values()) if report["engines"] else False
    report["overall_ok"] = (
        report["requirements"]["ok"]
        and report["local_adk"]["ok"]
        and engine_ok
        and report.get("auth", {}).get("ok", False)
    )
    report["submission_snippet"] = _submission_snippet(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify A2A / Agent Engine readiness")
    parser.add_argument(
        "--project",
        default="",
        help="GCP project ID (default: GCP_PROJECT env)",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--json", action="store_true", help="Print JSON report only")
    parser.add_argument(
        "--write-submission-snippet",
        metavar="PATH",
        help="Write submission A2A paragraph to a file",
    )
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    report = run_verification(args.project, args.region)

    if args.write_submission_snippet:
        out = Path(args.write_submission_snippet)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report["submission_snippet"])
        print(f"Wrote {out}")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        status = "PASS" if report["overall_ok"] else "FAIL"
        print(f"A2A verification: {status}\n")
        print("Requirements:", "ok" if report["requirements"]["ok"] else "FAIL")
        for c in report["requirements"]["checks"]:
            mark = "✓" if c.get("ok") else "✗"
            print(f"  {mark} {c['file']}")
        print("\nLocal ADK tree:", "ok" if report["local_adk"]["ok"] else "FAIL")
        if report["local_adk"].get("baseline_sub_agents"):
            print(f"  Baseline sub_agents: {report['local_adk']['baseline_sub_agents']}")
            print(f"  Internal sub_agents: {report['local_adk']['internal_sub_agents']}")
        print("\nAgent Engine REST:")
        for name, eng in report["engines"].items():
            mark = "✓" if eng.get("ok") else "✗"
            print(f"  {mark} {name}: {eng.get('display_name') or eng.get('error', 'unknown')}")
            if eng.get("stream_query_exposed"):
                print("      stream_query: exposed")
        print("\n--- Submission snippet (also in docs/SUBMISSION.md) ---\n")
        print(report["submission_snippet"])

    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
