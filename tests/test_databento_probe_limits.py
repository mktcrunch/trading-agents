"""DataBento probe download guards."""
from unittest.mock import MagicMock, patch

import pandas as pd

from src.apis.databento_client import DataBentoClient


def _client() -> DataBentoClient:
    with patch.object(DataBentoClient, "__init__", lambda self, api_key=None: None):
        client = DataBentoClient()
    client.client = MagicMock()
    return client


def test_check_probe_download_allowed_rejects_oversized_estimate():
    client = _client()
    client.client.metadata.get_billable_size.return_value = 50 * 1024 * 1024
    client.client.metadata.get_cost.return_value = 0.01

    allowed, reason, estimated = client.check_probe_download_allowed(
        symbols=["SPY"],
        dataset="EQUS.SUMMARY",
        schema="statistics",
        lookback_days=90,
    )

    assert allowed is False
    assert "download_size_exceeded" in reason
    assert estimated == 50 * 1024 * 1024
    client.client.metadata.get_cost.assert_not_called()


def test_check_probe_download_allowed_rejects_high_cost():
    client = _client()
    client.client.metadata.get_billable_size.return_value = 1024
    client.client.metadata.get_cost.return_value = 2.5

    allowed, reason, _ = client.check_probe_download_allowed(
        symbols=["SPY"],
        dataset="EQUS.MINI",
        schema="ohlcv-1d",
        lookback_days=90,
    )

    assert allowed is False
    assert "download_cost_exceeded" in reason


def test_fetch_range_skips_download_when_over_limit():
    client = _client()
    client.check_probe_download_allowed = MagicMock(
        return_value=(False, "download_size_exceeded:50.0MB>10MB", 50 * 1024 * 1024)
    )

    df = client.fetch_range(
        symbols=["SPY"],
        dataset="EQUS.SUMMARY",
        schema="statistics",
        lookback_days=90,
    )

    assert df.empty
    assert client.last_fetch_skip_reason == "download_size_exceeded:50.0MB>10MB"
    client.client.timeseries.get_range.assert_not_called()


def test_fetch_range_downloads_when_within_limit():
    client = _client()
    client.check_probe_download_allowed = MagicMock(return_value=(True, "", 1024))
    store = MagicMock()
    store.to_df.return_value = pd.DataFrame(
        {"symbol": ["SPY"], "close": [100.0]},
    )
    client.client.timeseries.get_range.return_value = store

    df = client.fetch_range(
        symbols=["SPY"],
        dataset="EQUS.MINI",
        schema="ohlcv-1d",
        lookback_days=90,
    )

    assert not df.empty
    client.client.timeseries.get_range.assert_called_once()


def test_high_risk_probe_skipped_when_size_estimate_fails():
    client = _client()
    client.client.metadata.get_billable_size.side_effect = RuntimeError("api down")

    allowed, reason, _ = client.check_probe_download_allowed(
        symbols=["SPY"],
        dataset="EQUS.SUMMARY",
        schema="statistics",
        lookback_days=90,
    )

    assert allowed is False
    assert reason == "size_estimate_failed:statistics"
