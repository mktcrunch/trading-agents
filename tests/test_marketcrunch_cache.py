from unittest.mock import MagicMock, patch

from src.apis.marketcrunch_client import (
    MarketCrunchClient,
    clear_mc_cache,
    mc_estimate_needs_retry,
)


def test_mc_estimate_needs_retry_empty_confidence_or_zero_target():
    assert mc_estimate_needs_retry({"confidence": None, "target_delta_pct": "0.00%"})
    assert mc_estimate_needs_retry({"confidence": "Unknown", "target_delta_pct": "1.0%"})
    assert mc_estimate_needs_retry({"confidence": 0, "target_delta_pct": "1.0%"})
    assert mc_estimate_needs_retry({"confidence": "Low", "target_delta_pct": "0.00%"})
    assert not mc_estimate_needs_retry(
        {"confidence": "Low", "target_delta_pct": "0.23%", "target_price": "$53.25"}
    )
    assert not mc_estimate_needs_retry({"confidence": "High", "target_delta_pct": "-2.08%"})


@patch("src.apis.marketcrunch_client.config.MC_API_CACHE_TTL_SEC", 900)
@patch("src.apis.marketcrunch_client.request_with_retry")
def test_get_ai_estimates_uses_cache(mock_request):
    clear_mc_cache()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ai_estimate": {"target_delta_pct": "1.25%", "confidence": "High"},
    }
    mock_request.return_value = mock_response

    client = MarketCrunchClient()
    first = client.get_ai_estimates("SPY")
    second = client.get_ai_estimates("SPY")

    assert first is not None
    assert second is not None
    assert mock_request.call_count == 1
    clear_mc_cache()


@patch("src.apis.marketcrunch_client.config.MC_API_CACHE_TTL_SEC", 0)
@patch("src.apis.marketcrunch_client.request_with_retry")
def test_cache_disabled_when_ttl_zero(mock_request):
    clear_mc_cache()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ai_estimate": {"target_delta_pct": "0.50%", "confidence": "Low"},
    }
    mock_request.return_value = mock_response

    client = MarketCrunchClient()
    client.get_ai_estimates("QQQ")
    client.get_ai_estimates("QQQ")

    assert mock_request.call_count == 2
    clear_mc_cache()


@patch("src.apis.marketcrunch_client.config.MC_API_CACHE_TTL_SEC", 0)
@patch("src.apis.marketcrunch_client.config.MC_API_MAX_RETRIES", 3)
@patch("src.apis.marketcrunch_client.config.MC_API_RETRY_BACKOFF_SEC", 0)
@patch("src.apis.marketcrunch_client.time.sleep")
@patch("src.apis.marketcrunch_client.request_with_retry")
def test_get_ai_estimates_retries_incomplete_estimate(mock_request, _sleep):
    clear_mc_cache()
    empty = MagicMock()
    empty.json.return_value = {
        "ai_estimate": {"target_delta_pct": "0.00%", "confidence": None},
    }
    good = MagicMock()
    good.json.return_value = {
        "ai_estimate": {
            "target_delta_pct": "1.25%",
            "confidence": "High",
            "target_price": "$100.00",
        },
    }
    mock_request.side_effect = [empty, good]

    client = MarketCrunchClient()
    result = client.get_ai_estimates("VTI")

    assert result is not None
    assert result["ai_estimate"]["target_delta_numeric"] == 1.25
    assert mock_request.call_count == 2
    clear_mc_cache()


@patch("src.apis.marketcrunch_client.config.MC_API_CACHE_TTL_SEC", 0)
@patch("src.apis.marketcrunch_client.config.MC_API_MAX_RETRIES", 3)
@patch("src.apis.marketcrunch_client.config.MC_API_RETRY_BACKOFF_SEC", 0)
@patch("src.apis.marketcrunch_client.time.sleep")
@patch("src.apis.marketcrunch_client.request_with_retry")
def test_get_ai_estimates_returns_none_when_estimate_stays_empty(mock_request, _sleep):
    clear_mc_cache()
    empty = MagicMock()
    empty.json.return_value = {
        "ai_estimate": {"target_delta_pct": "0.00%", "confidence": None},
    }
    mock_request.return_value = empty

    client = MarketCrunchClient()
    result = client.get_ai_estimates("GLD")

    assert result is None
    assert mock_request.call_count == 3
    clear_mc_cache()
