"""
LLM + heuristic planner: decides which DataBento datasets to probe each day.
"""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from src import config
from src.apis.gemini_client import get_genai_client
from src.agents.ledger_utils import GEMINI_FLASH_MODEL, parse_ledger_response
from src.discovery.registry import hours_since, never_probed, probe_key
from src.logger import setup_logger

logger = setup_logger(__name__)



@dataclass
class ProbeTarget:
    dataset: str
    schema: str
    priority: int
    rationale: str
    action: str  # probe_new | reprobe_approved | reprobe_rejected


class DiscoveryPlanner:
    """Selects daily DataBento probe targets — agentic dataset exploration."""

    def __init__(self):
        self.cfg = config.DISCOVERY_CONFIG
        self.universe = config.TICKER_UNIVERSE
        self.client = get_genai_client()

    def _heuristic_plan(
        self,
        catalog: List[Dict],
        registry: Dict,
    ) -> List[ProbeTarget]:
        """Fallback planner when LLM unavailable."""
        max_probes = self.cfg.get("max_probes_per_day", 5)
        cooldown = self.cfg.get("probe_cooldown_hours", 168)
        reprobe_approved = self.cfg.get("reprobe_approved_hours", 24)

        candidates: List[ProbeTarget] = []
        probes = registry.get("probes", {})

        for entry in catalog:
            key = entry["probe_key"]
            probe = probes.get(key, {})
            hours = hours_since(probe.get("last_probed_at"))

            if never_probed(registry, entry["dataset"], entry["schema"]):
                candidates.append(ProbeTarget(
                    dataset=entry["dataset"],
                    schema=entry["schema"],
                    priority=100,
                    rationale="Never probed — explore new dataset",
                    action="probe_new",
                ))
                continue

            if probe.get("last_status") == "approved" and (
                hours is None or hours >= reprobe_approved
            ):
                candidates.append(ProbeTarget(
                    dataset=entry["dataset"],
                    schema=entry["schema"],
                    priority=80,
                    rationale="Re-evaluate previously approved source",
                    action="reprobe_approved",
                ))
            elif probe.get("last_status") in ("rejected", "error") and (
                hours is None or hours >= cooldown
            ):
                candidates.append(ProbeTarget(
                    dataset=entry["dataset"],
                    schema=entry["schema"],
                    priority=40,
                    rationale="Cooldown elapsed — retry rejected source",
                    action="reprobe_rejected",
                ))

        candidates.sort(key=lambda t: t.priority, reverse=True)
        return candidates[:max_probes]

    def _build_prompt(self, catalog_summary: List[Dict], registry: Dict) -> str:
        probed = registry.get("probes", {})
        return f"""You are an autonomous data discovery agent for an ETF trading system.

Your job each day is to choose which DataBento dataset/schema combinations to sample and evaluate.
Goal: find NEW predictive data sources for tickers {', '.join(self.universe)}.

You have memory of past probes. Prefer:
1. NEW dataset/schema pairs never probed before
2. Re-probing previously APPROVED sources (data may have updated)
3. Occasionally retry REJECTED sources if they might now work

Constraints:
- Return at most {self.cfg.get('max_probes_per_day', 5)} targets
- Only choose from the catalog below
- Prefer ohlcv-1d and ohlcv-1h schemas for swing trading
- Avoid expensive tick-level schemas unless high expected value
- Spread probes across different datasets when possible

Catalog (dataset/schema + probe history):
{json.dumps(catalog_summary[:40], indent=2)}

Past probe count: {len(probed)}

Return ONLY a JSON object:
{{
  "strategy_note": "one sentence on today's discovery strategy",
  "targets": [
    {{
      "dataset": "EQUS.SUMMARY",
      "schema": "ohlcv-1d",
      "priority": 1,
      "action": "probe_new",
      "rationale": "why probe this today"
    }}
  ]
}}"""

    async def plan_daily_probes(
        self,
        catalog: List[Dict],
        registry: Dict,
    ) -> tuple[List[ProbeTarget], str]:
        """
        LLM-guided daily probe plan with heuristic fallback.
        """
        if not self.cfg.get("llm_planner", True):
            targets = self._heuristic_plan(catalog, registry)
            return targets, "heuristic planner"

        from src.discovery.registry import catalog_entry_summary

        catalog_summary = catalog_entry_summary(catalog, registry)
        prompt = self._build_prompt(catalog_summary, registry)

        try:
            response = self.client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=prompt,
            )
            parsed = parse_ledger_response(response.text)
            if isinstance(parsed, list):
                parsed = {"targets": parsed}
            strategy = parsed.get("strategy_note", "")
            valid_keys = {e["probe_key"] for e in catalog}
            targets = []
            for item in parsed.get("targets", []):
                ds = item.get("dataset", "")
                sc = item.get("schema", "")
                key = probe_key(ds, sc)
                if key not in valid_keys:
                    continue
                targets.append(ProbeTarget(
                    dataset=ds,
                    schema=sc,
                    priority=int(item.get("priority", 50)),
                    rationale=str(item.get("rationale", "")),
                    action=str(item.get("action", "probe_new")),
                ))
            if targets:
                targets.sort(key=lambda t: t.priority)
                max_probes = self.cfg.get("max_probes_per_day", 5)
                logger.info(f"LLM discovery plan: {len(targets)} targets — {strategy}")
                return targets[:max_probes], strategy
        except Exception as e:
            logger.warning(f"LLM planner failed, using heuristics: {e}")

        return self._heuristic_plan(catalog, registry), "heuristic fallback"
