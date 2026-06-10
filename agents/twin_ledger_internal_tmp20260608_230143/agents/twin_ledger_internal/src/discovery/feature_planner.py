"""
LLM feature planner — proposes schema-specific feature formulas per probe.
"""
import json
from typing import Dict, List, Optional

from src import config
from src.apis.gemini_client import get_genai_client
from src.agents.ledger_utils import GEMINI_FLASH_MODEL, parse_ledger_response
from src.discovery.formula_engine import FORMULA_DSL_DOCS, validate_feature_spec
from src.discovery.registry import get_probe, probe_key
from src.logger import setup_logger

logger = setup_logger(__name__)


SCHEMA_HINTS = {
    "ohlcv-1d": "Daily bars — favor multi-day momentum, SMA distance, volume regime shifts.",
    "ohlcv-1h": "Hourly bars aggregated to daily — favor intraday range/volume patterns.",
    "ohlcv-1m": "Minute bars aggregated to daily — favor microstructure volume spikes.",
}


def heuristic_feature_proposals(schema: str, registry: Dict, dataset: str) -> List[Dict]:
    """Rotate through formula variants when LLM is off or fails."""
    probe = get_probe(registry, dataset, schema)
    probe_count = probe.get("probe_count", 0)
    offset = probe_count % 3

    daily_sets = [
        [
            {
                "id": "momentum_10d",
                "description": "10-day close momentum",
                "formula": {"op": "pct_change", "column": "close", "periods": 10},
            },
            {
                "id": "vol_zscore_15",
                "description": "15-day volume z-score",
                "formula": {"op": "zscore", "column": "volume", "window": 15},
            },
            {
                "id": "close_vs_sma30",
                "description": "Close vs 30-day SMA",
                "formula": {
                    "op": "div",
                    "left": {"op": "column", "column": "close"},
                    "right": {"op": "rolling_mean", "column": "close", "window": 30},
                },
            },
        ],
        [
            {
                "id": "high_low_spread_z",
                "description": "Z-score of intraday range",
                "formula": {"op": "zscore", "column": "close", "window": 20},
            },
            {
                "id": "vol_momentum_5",
                "description": "5-day volume change",
                "formula": {"op": "pct_change", "column": "volume", "periods": 5},
            },
            {
                "id": "close_open_gap",
                "description": "Open-to-close return proxy",
                "formula": {
                    "op": "div",
                    "left": {"op": "sub", "left": {"op": "column", "column": "close"},
                             "right": {"op": "column", "column": "open"}},
                    "right": {"op": "column", "column": "open"},
                },
            },
        ],
        [
            {
                "id": "range_momentum",
                "description": "Range pct times 5d momentum",
                "formula": {
                    "op": "mul",
                    "left": {"op": "range_pct"},
                    "right": {"op": "pct_change", "column": "close", "periods": 5},
                },
            },
            {
                "id": "volatility_20d",
                "description": "20-day return volatility",
                "formula": {
                    "op": "rolling_std",
                    "column": "close",
                    "window": 20,
                },
            },
            {
                "id": "low_vs_close",
                "description": "Distance from low to close",
                "formula": {
                    "op": "div",
                    "left": {"op": "sub", "left": {"op": "column", "column": "close"},
                             "right": {"op": "column", "column": "low"}},
                    "right": {"op": "column", "column": "close"},
                },
            },
        ],
    ]

    intraday_extra = [
        {
            "id": "intraday_range_z",
            "description": "Z-scored daily range from intraday bars",
            "formula": {"op": "zscore", "column": "high", "window": 10},
        },
    ]

    base = daily_sets[offset]
    if schema != "ohlcv-1d":
        base = base + intraday_extra

    validated = []
    for spec in base:
        clean = validate_feature_spec(spec)
        if clean:
            validated.append(clean)
    return validated


class FeaturePlanner:
    """Proposes new feature formulas tailored to a dataset/schema probe."""

    def __init__(self):
        self.cfg = config.DISCOVERY_CONFIG
        self.universe = config.TICKER_UNIVERSE
        self.client = get_genai_client()

    def _build_prompt(
        self,
        dataset: str,
        schema: str,
        registry: Dict,
        action: str,
    ) -> str:
        probe = get_probe(registry, dataset, schema)
        past_features = probe.get("proposed_features", [])
        schema_hint = SCHEMA_HINTS.get(schema, "OHLCV bars — propose predictive swing-trading features.")

        return f"""You are a quantitative feature engineer for ETF swing trading.

Propose NEW feature formulas to test on DataBento data:
- Dataset: {dataset}
- Schema: {schema}
- Tickers: {', '.join(self.universe)}
- Probe action: {action}

Schema guidance: {schema_hint}

Previously proposed features for this dataset/schema (avoid duplicates):
{json.dumps(past_features[-12:], indent=2)}

Past probe stats: status={probe.get('last_status')}, best_ic={probe.get('best_ic', 0)}

{FORMULA_DSL_DOCS}

Propose {self.cfg.get('max_features_per_probe', 6)} diverse features that might predict next-day returns.
Prefer economically interpretable signals: momentum, volume anomalies, range/volatility, mean-reversion distance.
For intraday schemas (ohlcv-1h/1m), emphasize patterns visible after daily aggregation.

Return ONLY JSON:
{{
  "feature_strategy": "one sentence",
  "features": [
    {{
      "id": "vol_breakout_20",
      "description": "volume z-score vs 20d",
      "formula": {{"op": "zscore", "column": "volume", "window": 20}}
    }}
  ]
}}"""

    def _parse_features(self, parsed: Dict, seen_ids: set) -> List[Dict]:
        validated = []
        for raw in parsed.get("features", []):
            clean = validate_feature_spec(raw)
            if not clean or clean["id"] in seen_ids:
                continue
            seen_ids.add(clean["id"])
            validated.append(clean)
        return validated

    async def propose_features(
        self,
        dataset: str,
        schema: str,
        registry: Dict,
        action: str = "probe_new",
    ) -> tuple[List[Dict], str]:
        """
        Return validated feature specs and strategy note.
        """
        max_features = self.cfg.get("max_features_per_probe", 6)
        seen_ids: set = set()

        if not self.cfg.get("llm_feature_planner", True):
            specs = heuristic_feature_proposals(schema, registry, dataset)
            return specs[:max_features], "heuristic feature variants"

        prompt = self._build_prompt(dataset, schema, registry, action)

        try:
            response = self.client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=prompt,
            )
            parsed = parse_ledger_response(response.text)
            if isinstance(parsed, list):
                parsed = {"features": parsed}
            strategy = parsed.get("feature_strategy", "")
            validated = self._parse_features(parsed, seen_ids)
            if validated:
                logger.info(
                    f"LLM proposed {len(validated)} features for "
                    f"{dataset}/{schema}: {strategy}"
                )
                return validated[:max_features], strategy
        except Exception as e:
            logger.warning(f"LLM feature planner failed for {dataset}/{schema}: {e}")

        specs = heuristic_feature_proposals(schema, registry, dataset)
        return specs[:max_features], "heuristic feature fallback"
