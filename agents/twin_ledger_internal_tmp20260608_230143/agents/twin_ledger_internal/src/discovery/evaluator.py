"""
Three-gate evaluation for discovered DataBento features.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from src import config
from src.discovery.features import FEATURE_DEFINITIONS, build_feature_definitions
from src.logger import setup_logger

logger = setup_logger(__name__)


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    clean = pd.concat([x, y], axis=1).dropna()
    if len(clean) < 20:
        return 0.0, 1.0
    ic, p_value = spearmanr(clean.iloc[:, 0], clean.iloc[:, 1])
    if np.isnan(ic):
        return 0.0, 1.0
    return float(ic), float(p_value)


def _t_stat_from_ic(ic: float, n: int) -> float:
    if n < 3 or abs(ic) >= 1:
        return 0.0
    return float(ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2))


def _mutual_information(x: pd.Series, y: pd.Series) -> float:
    clean = pd.concat([x, y], axis=1).dropna()
    if len(clean) < 20:
        return 0.0
    try:
        mi = mutual_info_regression(
            clean.iloc[:, 0].values.reshape(-1, 1),
            clean.iloc[:, 1].values,
            random_state=0,
        )[0]
        return float(mi)
    except Exception:
        return 0.0


def evaluate_feature(
    panel: pd.DataFrame,
    feature_col: str,
    feature_id: str,
    description: str,
    universe: List[str],
) -> Optional[Dict]:
    """Run gate 1/2/3 checks for a single feature."""
    if panel.empty or feature_col not in panel.columns:
        return None

    cfg = config.DISCOVERY_CONFIG
    symbols_with_data = panel["symbol"].nunique()
    coverage_pct = (symbols_with_data / len(universe)) * 100 if universe else 0
    if coverage_pct < cfg["min_universe_coverage_pct"]:
        logger.warning(
            f"{feature_id}: coverage {coverage_pct:.1f}% "
            f"< {cfg['min_universe_coverage_pct']}%"
        )
        return None

    x = panel[feature_col]
    y = panel["forward_return_1d"]
    baseline = panel.get("baseline_momentum", panel["return_1d"].shift(1))

    mi = _mutual_information(x, y)
    if mi < cfg["gate_1_mi_threshold"]:
        logger.info(f"{feature_id}: failed gate 1 (MI={mi:.4f})")
        return None

    ic, _ = _safe_spearman(x, y)
    n = len(pd.concat([x, y], axis=1).dropna())
    t_stat = _t_stat_from_ic(ic, n)
    if abs(ic) < cfg["gate_2_ic_threshold"] or abs(t_stat) < cfg["gate_2_t_stat_threshold"]:
        logger.info(
            f"{feature_id}: failed gate 2 (IC={ic:.4f}, t={t_stat:.2f})"
        )
        return None

    baseline_ic, _ = _safe_spearman(baseline, y)
    incremental_alpha = abs(ic) - abs(baseline_ic)
    if incremental_alpha < cfg["gate_3_incremental_alpha_threshold"]:
        logger.info(
            f"{feature_id}: failed gate 3 (incremental_alpha={incremental_alpha:.4f})"
        )
        return None

    logger.info(
        f"{feature_id}: APPROVED | MI={mi:.4f} IC={ic:.4f} "
        f"t={t_stat:.2f} inc_alpha={incremental_alpha:.4f}"
    )
    return {
        "id": feature_id,
        "feature": feature_col,
        "description": description,
        "metrics": {
            "mi": round(mi, 6),
            "ic": round(ic, 6),
            "t_stat": round(t_stat, 4),
            "incremental_alpha": round(incremental_alpha, 6),
            "universe_coverage_pct": round(coverage_pct, 2),
            "sample_rows": int(n),
        },
    }


def evaluate_all_features(
    panel: pd.DataFrame,
    universe: List[str],
    dataset: str,
    schema: str,
    feature_prefix: str = "",
    feature_specs: Optional[List[Dict]] = None,
    include_baseline: bool = True,
) -> List[Dict]:
    """Evaluate baseline + proposed features and return approved sources."""
    from src.discovery.features import feature_id_with_prefix

    definitions = build_feature_definitions(
        feature_specs=feature_specs,
        include_baseline=include_baseline,
    )
    approved = []
    for definition in definitions:
        fid = feature_id_with_prefix(feature_prefix, definition["id"])
        result = evaluate_feature(
            panel=panel,
            feature_col=definition["column"],
            feature_id=fid,
            description=f"[{dataset}/{schema}] {definition['description']}",
            universe=universe,
        )
        if result:
            result["dataset"] = dataset
            result["schema"] = schema
            result["proposed"] = definition.get("proposed", False)
            if definition.get("formula"):
                result["formula"] = definition["formula"]
            approved.append(result)
    return approved
