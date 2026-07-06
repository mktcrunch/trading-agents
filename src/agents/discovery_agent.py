"""
DataBento Discovery Agent
Agentic daily exploration: catalog scan → LLM/heuristic planner → probe → evaluate.
Writes approved sources to data/approved_datasources.json
"""
from typing import Dict, List, Optional

from src import config
from src.agents.base_agent import BaseAgent
from src.apis.databento_client import DataBentoClient
from src.discovery.approved_sources import (
    is_stale,
    load_approved_sources,
    merge_probe_results,
    save_approved_sources,
)
from src.discovery.catalog import probe_lookback_days, scan_catalog
from src.discovery.evaluator import evaluate_all_features
from src.discovery.feature_planner import FeaturePlanner
from src.discovery.features import (
    build_feature_definitions,
    build_feature_panel,
    latest_ticker_features,
    normalize_bars,
    supports_probe_schema,
)
from src.discovery.planner import DiscoveryPlanner, ProbeTarget
from src.discovery.registry import (
    append_daily_run,
    load_registry,
    record_probe,
    save_registry,
)
from src.logger import setup_logger

logger = setup_logger(__name__)


class DiscoveryAgent(BaseAgent):
    """
    Autonomous DataBento data discovery agent.
    Each day it scans the catalog, plans new probes, evaluates features,
    and accumulates approved sources for internal trading.
    """

    def __init__(self):
        super().__init__(system="discovery")
        self.universe = config.TICKER_UNIVERSE
        self.cfg = config.DISCOVERY_CONFIG

    def _feature_prefix(self, dataset: str, schema: str) -> str:
        return f"{dataset}_{schema}".replace(".", "_").replace("|", "_")

    def _probe_failure_result(
        self,
        target: ProbeTarget,
        error: str,
        *,
        status: str = "error",
    ) -> Dict:
        return {
            "dataset": target.dataset,
            "schema": target.schema,
            "status": status,
            "error": error,
            "approved_count": 0,
            "rejected_count": 0,
            "best_ic": 0.0,
            "sample_rows": 0,
            "sources": [],
            "ticker_features": {},
            "rationale": target.rationale,
            "action": target.action,
        }

    async def _probe_target(
        self,
        client: DataBentoClient,
        target: ProbeTarget,
        registry: Optional[Dict] = None,
    ) -> Dict:
        """Fetch, engineer, and evaluate one dataset/schema probe."""
        registry = registry or {}
        if not supports_probe_schema(target.schema):
            return self._probe_failure_result(
                target,
                f"unsupported schema: {target.schema}",
                status="skipped",
            )

        lookback_days = probe_lookback_days(target.schema)
        bars = client.fetch_range(
            symbols=self.universe,
            dataset=target.dataset,
            schema=target.schema,
            lookback_days=lookback_days,
        )
        if bars is None or bars.empty:
            if client.last_fetch_skip_reason:
                return self._probe_failure_result(
                    target,
                    client.last_fetch_skip_reason,
                    status="skipped",
                )
            return self._probe_failure_result(target, "empty_fetch")

        feature_planner = FeaturePlanner()
        feature_specs, feature_strategy = await feature_planner.propose_features(
            dataset=target.dataset,
            schema=target.schema,
            registry=registry,
            action=target.action,
        )
        self.log_action(
            f"Feature plan for {target.dataset}/{target.schema}: "
            f"{len(feature_specs)} proposals — {feature_strategy}"
        )

        daily = normalize_bars(bars, target.schema)
        panel = build_feature_panel(daily, feature_specs=feature_specs)
        if panel.empty:
            return self._probe_failure_result(target, "empty_feature_panel")

        prefix = self._feature_prefix(target.dataset, target.schema)
        definitions = build_feature_definitions(
            feature_specs=feature_specs,
            include_baseline=self.cfg.get("include_baseline_features", True),
        )
        eval_result = evaluate_all_features(
            panel=panel,
            universe=self.universe,
            dataset=target.dataset,
            schema=target.schema,
            feature_prefix=prefix,
            feature_specs=feature_specs,
            include_baseline=self.cfg.get("include_baseline_features", True),
        )
        approved = eval_result["approved"]
        feature_evaluations = eval_result["evaluations"]
        ticker_features = latest_ticker_features(
            panel, prefix=prefix, definitions=definitions
        )
        best_ic = max(
            (abs(s["metrics"]["ic"]) for s in approved),
            default=0.0,
        )

        return {
            "dataset": target.dataset,
            "schema": target.schema,
            "status": "approved" if approved else "rejected",
            "approved_count": len(approved),
            "rejected_count": len(feature_evaluations) - len(approved),
            "best_ic": best_ic,
            "sample_rows": len(panel),
            "sources": approved,
            "ticker_features": ticker_features,
            "rationale": target.rationale,
            "action": target.action,
            "feature_strategy": feature_strategy,
            "proposed_features": feature_specs,
            "approved_feature_ids": [s["id"] for s in approved],
            "feature_evaluations": feature_evaluations,
        }

    async def _run_legacy_discovery(self, client: DataBentoClient) -> Dict:
        """Single fixed dataset/schema path (pre-agentic behavior)."""
        target = ProbeTarget(
            dataset=self.cfg.get("dataset", "EQUS.MINI"),
            schema=self.cfg.get("schema", "ohlcv-1d"),
            priority=1,
            rationale="legacy fixed dataset",
            action="reprobe_approved",
        )
        result = await self._probe_target(client, target)
        payload = {
            "mode": "legacy",
            "dataset": target.dataset,
            "schema": target.schema,
            "universe": self.universe,
            "sources": result["sources"],
            "ticker_features": result["ticker_features"],
            "probes_today": [result],
            "summary": {
                "approved_count": result["approved_count"],
                "tickers_with_features": len(result["ticker_features"]),
                "sample_rows": result["sample_rows"],
                "probes_run": 1,
            },
        }
        save_approved_sources(payload)
        return payload

    async def run_daily_discovery(self) -> Dict:
        """Execute agentic discovery pipeline and persist results."""
        self.log_action("Starting agentic DataBento discovery pipeline")

        try:
            client = DataBentoClient()
        except Exception as e:
            self.log_error(f"DataBento client init failed: {e}")
            return self._save_empty_result(error=str(e))

        if not self.cfg.get("agentic_discovery", True):
            return await self._run_legacy_discovery(client)

        registry = load_registry()
        previous = load_approved_sources()
        catalog = scan_catalog(client)

        if not catalog:
            self.log_error("Catalog scan returned no probeable datasets")
            return self._save_empty_result(error="empty_catalog")

        planner = DiscoveryPlanner()
        targets, strategy = await planner.plan_daily_probes(catalog, registry)
        if not targets:
            self.log_action("No probe targets selected today — using cached sources")
            return previous or self._save_empty_result(error="no_targets")

        self.log_action(
            f"Discovery plan: {len(targets)} probes — {strategy}"
        )

        all_sources: List[Dict] = list(previous.get("sources", []))
        all_ticker_features: Dict = dict(previous.get("ticker_features", {}))
        probe_summaries = []

        for target in targets:
            lookback_days = probe_lookback_days(target.schema)
            self.log_action(
                f"Probing {target.dataset}/{target.schema} "
                f"({target.action}, {lookback_days}d lookback)"
            )
            try:
                result = await self._probe_target(client, target, registry=registry)
            except Exception as e:
                self.log_error(
                    f"Probe failed for {target.dataset}/{target.schema}: {e}"
                )
                result = self._probe_failure_result(target, str(e))
            probe_summaries.append({
                k: result[k]
                for k in (
                    "dataset", "schema", "status", "approved_count", "rejected_count",
                    "best_ic", "sample_rows", "rationale", "action", "error",
                    "feature_strategy", "proposed_features", "approved_feature_ids",
                    "feature_evaluations",
                )
                if k in result
            })

            all_sources, all_ticker_features = merge_probe_results(
                all_sources,
                result["sources"],
                all_ticker_features,
                result["ticker_features"],
            )

            record_probe(
                registry,
                dataset=target.dataset,
                schema=target.schema,
                status=result["status"],
                approved_count=result["approved_count"],
                best_ic=result["best_ic"],
                sample_rows=result["sample_rows"],
                error=result.get("error"),
                rationale=target.rationale,
                proposed_features=result.get("proposed_features"),
                approved_feature_ids=result.get("approved_feature_ids"),
                feature_strategy=result.get("feature_strategy"),
            )
            self.log_action(
                f"Probe {target.dataset}/{target.schema}: {result['status']} "
                f"({result['approved_count']} approved, "
                f"{result.get('rejected_count', 0)} rejected)",
                data={
                    "dataset": target.dataset,
                    "schema": target.schema,
                    "status": result["status"],
                    "approved_count": result["approved_count"],
                    "rejected_count": result.get("rejected_count", 0),
                    "best_ic": result["best_ic"],
                    "feature_strategy": result.get("feature_strategy"),
                    "proposed_features": result.get("proposed_features"),
                    "feature_evaluations": result.get("feature_evaluations"),
                },
                event_type="discovery_probe",
            )

        daily_summary = {
            "strategy": strategy,
            "probes_run": len(targets),
            "catalog_size": len(catalog),
            "approved_total": len(all_sources),
            "probe_results": probe_summaries,
        }
        append_daily_run(registry, daily_summary)
        save_registry(registry)

        payload = {
            "mode": "agentic",
            "strategy_note": strategy,
            "universe": self.universe,
            "catalog_size": len(catalog),
            "probes_today": probe_summaries,
            "sources": all_sources,
            "ticker_features": all_ticker_features,
            "summary": {
                "approved_count": len(all_sources),
                "tickers_with_features": len(all_ticker_features),
                "probes_run": len(targets),
                "probes_approved": sum(
                    1 for p in probe_summaries if p.get("status") == "approved"
                ),
            },
        }
        save_approved_sources(payload)
        self.log_action(
            f"Agentic discovery complete: {len(targets)} probes, "
            f"{len(all_sources)} total approved sources, "
            f"{len(all_ticker_features)} tickers enriched"
        )
        return payload

    def _save_empty_result(self, error: str) -> Dict:
        payload = {
            "mode": "agentic",
            "universe": self.universe,
            "sources": [],
            "ticker_features": {},
            "probes_today": [],
            "summary": {
                "approved_count": 0,
                "tickers_with_features": 0,
                "error": error,
            },
        }
        save_approved_sources(payload)
        return payload

    async def ensure_fresh_sources(self, force: bool = False) -> Dict:
        """
        Run discovery if output is missing or stale.

        Overnight workflows call this via ``_ensure_discovery_fresh()`` (try discovery,
        fall back to GCS cache on failure, then continue without enrichment).
        """
        if not config.DATABENTO_DISCOVERY_ENABLED:
            data = load_approved_sources()
            if data:
                self.log_action(
                    "DataBento discovery disabled — using cached approved sources "
                    f"from {data.get('generated_at', 'unknown')}"
                )
                return data
            self.log_action("DataBento discovery disabled and no cached sources")
            return self._save_empty_result(error="discovery_disabled")

        if force or is_stale():
            reason = "forced refresh" if force else "missing or stale approved sources"
            self.log_action(f"Running discovery ({reason})")
            return await self.run_daily_discovery()

        data = load_approved_sources()
        from src.audit.serialize import discovery_snapshot

        self.log_action(
            f"Using cached discovery from {data.get('generated_at', 'unknown')} "
            f"({data.get('summary', {}).get('approved_count', 0)} sources)",
            data=discovery_snapshot(data),
        )
        return data

    async def execute(self) -> bool:
        if not config.DATABENTO_DISCOVERY_ENABLED:
            self.log_action("DataBento discovery disabled — skipping execute()")
            return True
        await self.run_daily_discovery()
        return True
