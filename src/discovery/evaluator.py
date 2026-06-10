"""
Three-gate evaluation for discovered DataBento features.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from src import config
from src.discovery.features import build_feature_definitions
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


def _thresholds() -> Dict[str, float]:
    cfg = config.DISCOVERY_CONFIG
    return {
        "min_universe_coverage_pct": cfg["min_universe_coverage_pct"],
        "gate_1_mi_threshold": cfg["gate_1_mi_threshold"],
        "gate_2_ic_threshold": cfg["gate_2_ic_threshold"],
        "gate_2_t_stat_threshold": cfg["gate_2_t_stat_threshold"],
        "gate_3_incremental_alpha_threshold": cfg["gate_3_incremental_alpha_threshold"],
    }


def _gate_check(
    name: str,
    passed: bool,
    value: Any,
    threshold: Any,
) -> Dict[str, Any]:
    return {
        "gate": name,
        "passed": passed,
        "value": value,
        "threshold": threshold,
    }


def _first_failed_gate(gates: List[Dict[str, Any]]) -> Optional[str]:
    for gate in gates:
        if not gate["passed"]:
            return gate["gate"]
    return None


def evaluate_feature_report(
    panel: pd.DataFrame,
    feature_col: str,
    feature_id: str,
    description: str,
    universe: List[str],
    proposed: bool = False,
) -> Dict[str, Any]:
    """Evaluate one feature and return approve/reject decision with full gate metrics."""
    thresholds = _thresholds()
    report: Dict[str, Any] = {
        "feature_id": feature_id,
        "feature": feature_col,
        "description": description,
        "proposed": proposed,
        "status": "rejected",
        "failed_gate": None,
        "metrics": {},
        "thresholds": thresholds,
        "gates": [],
    }

    if panel.empty or feature_col not in panel.columns:
        report["failed_gate"] = "missing_data"
        report["gates"].append(
            _gate_check("missing_data", False, "column_not_in_panel", feature_col)
        )
        return report

    symbols_with_data = panel["symbol"].nunique()
    coverage_pct = (symbols_with_data / len(universe)) * 100 if universe else 0.0
    report["metrics"]["universe_coverage_pct"] = round(coverage_pct, 2)
    report["metrics"]["symbols_with_data"] = int(symbols_with_data)

    coverage_gate = _gate_check(
        "coverage",
        coverage_pct >= thresholds["min_universe_coverage_pct"],
        coverage_pct,
        thresholds["min_universe_coverage_pct"],
    )
    report["gates"].append(coverage_gate)

    x = panel[feature_col]
    y = panel["forward_return_1d"]
    baseline = panel.get("baseline_momentum", panel["return_1d"].shift(1))
    clean_xy = pd.concat([x, y], axis=1).dropna()
    n = len(clean_xy)

    report["metrics"]["sample_rows"] = int(n)

    mi = _mutual_information(x, y)
    report["metrics"]["mi"] = round(mi, 6)
    gate_1 = _gate_check(
        "gate_1_mi",
        mi >= thresholds["gate_1_mi_threshold"],
        mi,
        thresholds["gate_1_mi_threshold"],
    )
    report["gates"].append(gate_1)

    ic, _ = _safe_spearman(x, y)
    t_stat = _t_stat_from_ic(ic, n)
    report["metrics"]["ic"] = round(ic, 6)
    report["metrics"]["t_stat"] = round(t_stat, 4)
    gate_2_passed = (
        abs(ic) >= thresholds["gate_2_ic_threshold"]
        and abs(t_stat) >= thresholds["gate_2_t_stat_threshold"]
    )
    gate_2 = _gate_check(
        "gate_2_ic_t_stat",
        gate_2_passed,
        {"ic": round(ic, 6), "t_stat": round(t_stat, 4)},
        {
            "ic": thresholds["gate_2_ic_threshold"],
            "t_stat": thresholds["gate_2_t_stat_threshold"],
        },
    )
    report["gates"].append(gate_2)

    baseline_ic, _ = _safe_spearman(baseline, y)
    incremental_alpha = abs(ic) - abs(baseline_ic)
    report["metrics"]["baseline_ic"] = round(baseline_ic, 6)
    report["metrics"]["incremental_alpha"] = round(incremental_alpha, 6)
    gate_3 = _gate_check(
        "gate_3_incremental_alpha",
        incremental_alpha >= thresholds["gate_3_incremental_alpha_threshold"],
        incremental_alpha,
        thresholds["gate_3_incremental_alpha_threshold"],
    )
    report["gates"].append(gate_3)

    failed = _first_failed_gate(report["gates"])
    if failed:
        report["failed_gate"] = failed
        logger.info(
            f"{feature_id}: REJECTED at {failed} | "
            f"MI={report['metrics']['mi']:.4f} IC={report['metrics']['ic']:.4f} "
            f"t={report['metrics']['t_stat']:.2f} "
            f"inc_alpha={report['metrics']['incremental_alpha']:.4f} "
            f"coverage={coverage_pct:.1f}%"
        )
        return report

    report["status"] = "approved"
    logger.info(
        f"{feature_id}: APPROVED | MI={report['metrics']['mi']:.4f} "
        f"IC={report['metrics']['ic']:.4f} t={report['metrics']['t_stat']:.2f} "
        f"inc_alpha={report['metrics']['incremental_alpha']:.4f}"
    )
    return report


def evaluate_feature(
    panel: pd.DataFrame,
    feature_col: str,
    feature_id: str,
    description: str,
    universe: List[str],
) -> Optional[Dict]:
    """Run gate 1/2/3 checks for a single feature. Returns approved source or None."""
    report = evaluate_feature_report(
        panel, feature_col, feature_id, description, universe
    )
    if report["status"] != "approved":
        return None
    return {
        "id": feature_id,
        "feature": feature_col,
        "description": description,
        "metrics": report["metrics"],
    }


def evaluate_all_features(
    panel: pd.DataFrame,
    universe: List[str],
    dataset: str,
    schema: str,
    feature_prefix: str = "",
    feature_specs: Optional[List[Dict]] = None,
    include_baseline: bool = True,
) -> Dict[str, Any]:
    """Evaluate baseline + proposed features; return approved sources and per-feature reports."""
    from src.discovery.features import feature_id_with_prefix

    definitions = build_feature_definitions(
        feature_specs=feature_specs,
        include_baseline=include_baseline,
    )
    approved: List[Dict] = []
    evaluations: List[Dict] = []

    for definition in definitions:
        fid = feature_id_with_prefix(feature_prefix, definition["id"])
        report = evaluate_feature_report(
            panel=panel,
            feature_col=definition["column"],
            feature_id=fid,
            description=f"[{dataset}/{schema}] {definition['description']}",
            universe=universe,
            proposed=definition.get("proposed", False),
        )
        evaluations.append(report)
        if report["status"] == "approved":
            source = {
                "id": fid,
                "feature": definition["column"],
                "description": report["description"],
                "metrics": report["metrics"],
                "dataset": dataset,
                "schema": schema,
                "proposed": definition.get("proposed", False),
            }
            if definition.get("formula"):
                source["formula"] = definition["formula"]
            approved.append(source)

    return {"approved": approved, "evaluations": evaluations}
