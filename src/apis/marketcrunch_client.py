"""
MarketCrunch API client
Fetches predictions and technical analysis
"""
import time
import requests
from typing import Any, Dict, Optional, Tuple

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
_mc_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def clear_mc_cache() -> None:
    """Clear in-process MC response cache (mainly for tests)."""
    _mc_cache.clear()


def _mc_cache_key(path: str, ticker: str) -> str:
    return f"{path}:{ticker.upper()}"


def _mc_cache_get(key: str) -> Optional[Dict[str, Any]]:
    ttl = config.MC_API_CACHE_TTL_SEC
    if ttl <= 0:
        return None
    entry = _mc_cache.get(key)
    if not entry:
        return None
    expires_at, data = entry
    if time.time() >= expires_at:
        _mc_cache.pop(key, None)
        return None
    return data


def _mc_cache_set(key: str, data: Dict[str, Any]) -> None:
    ttl = config.MC_API_CACHE_TTL_SEC
    if ttl <= 0 or not data:
        return
    _mc_cache[key] = (time.time() + ttl, data)


def _confidence_is_empty(confidence: Any) -> bool:
    """True when MC returned no usable confidence label/score."""
    if confidence is None:
        return True
    if isinstance(confidence, (int, float)):
        return confidence == 0
    if isinstance(confidence, str):
        label = confidence.strip()
        if not label:
            return True
        if label.lower() in ("unknown", "none", "0", "0.0"):
            return True
    return False


def _parse_target_delta_numeric(ai_est: Dict[str, Any]) -> Optional[float]:
    target_delta_str = ai_est.get("target_delta_pct")
    if target_delta_str is None:
        return None
    try:
        return float(str(target_delta_str).replace("%", "").strip())
    except (ValueError, AttributeError, TypeError):
        return None


def _target_value_missing(ai_est: Dict[str, Any]) -> bool:
    """True when MC returned no target delta and no target price."""
    target_price = ai_est.get("target_price")
    has_price = target_price not in (None, "", "null")
    delta = _parse_target_delta_numeric(ai_est)
    if delta is None and not has_price:
        return True
    if delta == 0 and not has_price:
        return True
    return False


def mc_estimate_needs_retry(ai_est: Optional[Dict[str, Any]]) -> bool:
    """Retry when MC HTTP succeeded but the ai_estimate payload is empty."""
    if not ai_est:
        return True
    return (
        _confidence_is_empty(ai_est.get("confidence"))
        or _target_value_missing(ai_est)
    )


def request_with_retry(
    method: str,
    url: str,
    *,
    session: Optional[requests.Session] = None,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    label: str = "MC API",
) -> Optional[requests.Response]:
    """HTTP request with connect/read timeouts and retries on transient failures."""
    http = session or requests
    timeout = (config.MC_API_CONNECT_TIMEOUT, config.MC_API_READ_TIMEOUT)
    max_retries = config.MC_API_MAX_RETRIES
    backoff = config.MC_API_RETRY_BACKOFF_SEC

    for attempt in range(1, max_retries + 1):
        try:
            response = http.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json,
                timeout=timeout,
            )
            if response.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                wait = backoff * attempt
                logger.warning(
                    f"{label}: HTTP {response.status_code}, "
                    f"retry {attempt}/{max_retries} in {wait:.0f}s"
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries:
                wait = backoff * attempt
                logger.warning(
                    f"{label}: {e}, retry {attempt}/{max_retries} in {wait:.0f}s"
                )
                time.sleep(wait)
            else:
                logger.error(f"{label}: {e} (exhausted {max_retries} attempts)")
        except requests.exceptions.HTTPError as e:
            logger.error(f"{label}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"{label}: {e}")
            return None

    return None


class MarketCrunchClient:
    """
    Client for MarketCrunch AI Estimates API

    Endpoints:
    - /ai-estimates?ticker=AAPL
    - /technical?ticker=AAPL
    - /factors?ticker=AAPL
    - /weekly?ticker=AAPL
    - /analyze?ticker=AAPL
    """

    def __init__(self):
        self.api_url = config.MC_API_URL
        self.api_key_id = config.MC_API_KEY_ID
        self.api_secret_key = config.MC_API_SECRET_KEY
        self.session = requests.Session()

        # Set headers for all requests
        self.headers = {
            "MC-API-KEY-ID": self.api_key_id,
            "MC-API-SECRET-KEY": self.api_secret_key,
            "Content-Type": "application/json",
        }

    def _get_json(
        self,
        path: str,
        ticker: str,
        label: str,
        *,
        use_cache: bool = True,
        cache_result: bool = True,
    ) -> Optional[Dict[str, Any]]:
        cache_key = _mc_cache_key(path, ticker)
        if use_cache:
            cached = _mc_cache_get(cache_key)
            if cached is not None:
                logger.info(f"✓ MC cache hit for {label} {ticker}")
                return cached

        url = f"{self.api_url}{path}"
        params = {"ticker": ticker.upper()}
        response = request_with_retry(
            "GET",
            url,
            session=self.session,
            headers=self.headers,
            params=params,
            label=f"MC {label} {ticker}",
        )
        if response is None:
            return None
        data = response.json()
        if cache_result:
            _mc_cache_set(cache_key, data)
        return data

    def get_ai_estimates(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get complete analysis for a ticker (uses /analyze endpoint)

        Returns rich data including:
        - ai_estimate: predictions (target_price, target_delta_pct, confidence)
        - technical: scores and summary
        - factors: positive/negative technical factors
        - weekly_range: price predictions for the week
        """
        try:
            max_retries = config.MC_API_MAX_RETRIES
            backoff = config.MC_API_RETRY_BACKOFF_SEC
            cache_key = _mc_cache_key("/analyze", ticker)

            cached = _mc_cache_get(cache_key)
            if cached is not None:
                ai_est = cached.get("ai_estimate") or {}
                if not mc_estimate_needs_retry(ai_est):
                    return cached

            for attempt in range(1, max_retries + 1):
                data = self._get_json(
                    "/analyze",
                    ticker,
                    "analyze",
                    use_cache=False,
                    cache_result=False,
                )
                if data is None:
                    if attempt < max_retries:
                        wait = backoff * attempt
                        logger.warning(
                            f"MC analyze {ticker}: no response, "
                            f"retry {attempt}/{max_retries} in {wait:.0f}s"
                        )
                        time.sleep(wait)
                        continue
                    return None

                ai_est = data.get("ai_estimate") or {}
                if mc_estimate_needs_retry(ai_est):
                    if attempt < max_retries:
                        wait = backoff * attempt
                        logger.warning(
                            f"MC analyze {ticker}: incomplete estimate "
                            f"(confidence={ai_est.get('confidence')!r}, "
                            f"target_delta_pct={ai_est.get('target_delta_pct')!r}), "
                            f"retry {attempt}/{max_retries} in {wait:.0f}s"
                        )
                        time.sleep(wait)
                        continue
                    logger.error(
                        f"MC analyze {ticker}: incomplete estimate after "
                        f"{max_retries} attempts"
                    )
                    return None

                target_delta_numeric = _parse_target_delta_numeric(ai_est)
                if target_delta_numeric is None:
                    target_delta_numeric = 0.0
                confidence = ai_est.get("confidence", "Unknown")

                logger.info(
                    f"✓ Fetched complete analysis for {ticker}: "
                    f"target={target_delta_numeric:.2f}%, confidence={confidence}"
                )

                if "ai_estimate" in data:
                    data["ai_estimate"]["target_delta_numeric"] = target_delta_numeric

                _mc_cache_set(cache_key, data)
                return data

            return None

        except Exception as e:
            logger.error(f"Unexpected error fetching analysis for {ticker}: {e}")
            return None

    def get_technical(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get technical analysis for a ticker

        Returns:
        {
            "ticker": "AAPL",
            "summary": "Bullish",
            "scores": {...},
            "timeframe": "1d"
        }
        """
        data = self._get_json("/technical", ticker, "technical")
        if data:
            logger.info(f"✓ Fetched technical analysis for {ticker}")
        return data

    def get_factors(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get positive and negative factors for a ticker

        Returns:
        {
            "model": "...",
            "positive_factors": [{"key": "...", "label": "..."}],
            "negative_factors": [{"key": "...", "label": "..."}]
        }
        """
        data = self._get_json("/factors", ticker, "factors")
        if data:
            logger.info(f"✓ Fetched factors for {ticker}")
        return data

    def get_weekly_range(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get weekly price range for a ticker

        Returns:
        {
            "week_ending": "2026-06-06",
            "min": 180.50,
            "max": 188.20,
            "current": 185.75,
            "refreshed_at": "2026-06-05T16:00:00Z",
            "history": [...]
        }
        """
        data = self._get_json("/weekly", ticker, "weekly")
        if data:
            logger.info(f"✓ Fetched weekly range for {ticker}")
        return data

    def get_analyze(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get complete analysis for a ticker (combines all endpoints)

        Returns combined data from ai-estimates, technical, weekly, factors
        """
        data = self._get_json("/analyze", ticker, "analyze")
        if data:
            logger.info(f"✓ Fetched complete analysis for {ticker}")
        return data

    def fetch_signals_for_tickers(self, tickers: list) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fetch AI estimates for multiple tickers

        Args:
            tickers: List of ticker symbols

        Returns:
            Dict mapping ticker -> ai_estimates response
        """
        signals = {}
        for ticker in tickers:
            signals[ticker] = self.get_ai_estimates(ticker)

        successful = sum(1 for s in signals.values() if s is not None)
        logger.info(f"Fetched signals for {successful}/{len(tickers)} tickers")

        return signals

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            result = self.get_ai_estimates("AAPL")
            if result:
                logger.info("✓ MarketCrunch API connection successful")
                return True
            else:
                logger.error("✗ MarketCrunch API returned no data")
                return False
        except Exception as e:
            logger.error(f"✗ MarketCrunch API connection failed: {e}")
            return False


if __name__ == "__main__":
    # Test the client
    client = MarketCrunchClient()
    print("Testing MarketCrunch API connection...")
    if client.test_connection():
        print("✓ Connection successful!")

        # Fetch some sample signals
        tickers = ["AAPL", "QQQ", "SPY"]
        signals = client.fetch_signals_for_tickers(tickers)
        for ticker, signal in signals.items():
            if signal:
                print(f"\n{ticker}:")
                print(f"  Target: {signal.get('target_delta_pct')}%")
                print(f"  Confidence: {signal.get('confidence'):.2f}")
    else:
        print("✗ Connection failed!")
