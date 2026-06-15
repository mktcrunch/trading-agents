"""Discovery probe loop isolation."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.discovery_agent import DiscoveryAgent
from src.discovery.planner import ProbeTarget


@pytest.mark.asyncio
async def test_probe_loop_continues_after_probe_exception():
    agent = DiscoveryAgent()
    target_ok = ProbeTarget(
        dataset="EQUS.MINI",
        schema="ohlcv-1d",
        priority=1,
        rationale="ok",
        action="probe_new",
    )
    target_bad = ProbeTarget(
        dataset="EQUS.SUMMARY",
        schema="statistics",
        priority=2,
        rationale="bad",
        action="probe_new",
    )

    ok_result = {
        "dataset": "EQUS.MINI",
        "schema": "ohlcv-1d",
        "status": "approved",
        "approved_count": 1,
        "rejected_count": 0,
        "best_ic": 0.05,
        "sample_rows": 90,
        "sources": [{"id": "feat1"}],
        "ticker_features": {"SPY": {"x": 1.0}},
        "rationale": "ok",
        "action": "probe_new",
    }

    with patch.object(
        agent,
        "_probe_target",
        new_callable=AsyncMock,
        side_effect=[RuntimeError("boom"), ok_result],
    ):
        with patch("src.agents.discovery_agent.record_probe"):
            with patch("src.agents.discovery_agent.merge_probe_results") as merge:
                merge.side_effect = [
                    ([{"id": "feat1"}], {"SPY": {"x": 1.0}}),
                    ([{"id": "feat1"}], {"SPY": {"x": 1.0}}),
                ]
                summaries = []
                all_sources = []
                all_features = {}
                for target in [target_bad, target_ok]:
                    try:
                        result = await agent._probe_target(
                            MagicMock(), target, registry={}
                        )
                    except Exception as e:
                        result = agent._probe_failure_result(target, str(e))
                    summaries.append(result["status"])
                    all_sources, all_features = merge(
                        all_sources,
                        result["sources"],
                        all_features,
                        result["ticker_features"],
                    )

    assert summaries == ["error", "approved"]
    assert all_sources == [{"id": "feat1"}]
