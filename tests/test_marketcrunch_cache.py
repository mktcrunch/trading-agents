from unittest.mock import MagicMock, patch

from src.apis.marketcrunch_client import MarketCrunchClient, clear_mc_cache


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
