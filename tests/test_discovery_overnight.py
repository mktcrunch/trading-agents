"""Overnight discovery: try probes first, cache fallback on failure."""
from unittest.mock import AsyncMock, patch

import pytest

from src import config
from src.adk.workflows.daily_pipeline import _ensure_discovery_fresh
from src.discovery.approved_sources import (
    discovery_run_meta,
    has_usable_cached_sources,
)


def test_has_usable_cached_sources_with_features():
    assert has_usable_cached_sources({"ticker_features": {"SPY": {"x": 1}}}) is True


def test_has_usable_cached_sources_empty():
    assert has_usable_cached_sources({}) is False
    assert has_usable_cached_sources({"sources": [], "ticker_features": {}}) is False


def test_discovery_run_meta_cached():
    meta = discovery_run_meta(
        {
            "generated_at": "2026-06-12T00:00:00+00:00",
            "sources": [{"id": "a"}],
            "ticker_features": {"SPY": {"f": 1}},
            "summary": {"approved_count": 1, "tickers_with_features": 1},
        },
        mode="cached",
    )
    assert meta["mode"] == "cached"
    assert meta["success"] is True
    assert meta["approved_count"] == 1
    assert meta["probes_run"] == 0


@pytest.mark.asyncio
async def test_ensure_discovery_fresh_runs_discovery_first():
    result = {
        "generated_at": "2026-06-15T00:00:00+00:00",
        "sources": [{"id": "src1"}],
        "ticker_features": {"QQQ": {"feat": 0.1}},
        "summary": {"approved_count": 1, "tickers_with_features": 1, "probes_run": 3},
    }
    with patch("src.discovery.approved_sources.is_stale", return_value=True):
        with patch(
            "src.agents.discovery_agent.DiscoveryAgent.ensure_fresh_sources",
            new_callable=AsyncMock,
            return_value=result,
        ) as mock_probe:
            meta = await _ensure_discovery_fresh()
    mock_probe.assert_awaited_once_with(force=False)
    assert meta["mode"] == "refreshed"
    assert meta["refreshed"] is True
    assert meta["probes_run"] == 3


@pytest.mark.asyncio
async def test_ensure_discovery_fresh_cache_fallback_on_failure():
    cached = {
        "generated_at": "2026-06-12T00:00:00+00:00",
        "sources": [{"id": "src1"}],
        "ticker_features": {"QQQ": {"feat": 0.1}},
    }
    with patch("src.discovery.approved_sources.is_stale", return_value=True):
        with patch(
            "src.agents.discovery_agent.DiscoveryAgent.ensure_fresh_sources",
            new_callable=AsyncMock,
            side_effect=RuntimeError("probe OOM"),
        ):
            with patch(
                "src.discovery.approved_sources.load_approved_sources",
                return_value=cached,
            ):
                meta = await _ensure_discovery_fresh()
    assert meta["mode"] == "cache_fallback"
    assert meta["success"] is True
    assert meta["approved_count"] == 1


@pytest.mark.asyncio
async def test_ensure_discovery_fresh_skips_when_discovery_disabled():
    cached = {
        "generated_at": "2026-06-20T00:00:00+00:00",
        "sources": [{"id": "feat1"}],
        "ticker_features": {"SPY": {"x": 1.0}},
        "summary": {"approved_count": 1, "tickers_with_features": 1},
    }
    with patch.object(config, "DATABENTO_DISCOVERY_ENABLED", False):
        with patch(
            "src.discovery.approved_sources.load_approved_sources",
            return_value=cached,
        ):
            meta = await _ensure_discovery_fresh()
    assert meta["mode"] == "cached"
    assert meta["success"] is True
    assert meta.get("discovery_disabled") is True
    assert meta["approved_count"] == 1


@pytest.mark.asyncio
async def test_ensure_discovery_fresh_skips_when_discovery_and_cache_fail():
    with patch("src.discovery.approved_sources.is_stale", return_value=True):
        with patch(
            "src.agents.discovery_agent.DiscoveryAgent.ensure_fresh_sources",
            new_callable=AsyncMock,
            side_effect=RuntimeError("probe failed"),
        ):
            with patch(
                "src.discovery.approved_sources.load_approved_sources",
                return_value={},
            ):
                meta = await _ensure_discovery_fresh()
    assert meta["mode"] == "skipped"
    assert meta["success"] is False
    assert "probe failed" in meta["error"]
