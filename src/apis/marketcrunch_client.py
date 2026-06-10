"""
MarketCrunch API client
Fetches predictions and technical analysis
"""
import time
import requests
from typing import Optional, Dict, Any
from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


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

    def _get_json(self, path: str, ticker: str, label: str) -> Optional[Dict[str, Any]]:
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
        return response.json()

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
            data = self._get_json("/analyze", ticker, "analyze")
            if data is None:
                return None

            # Extract and parse key prediction data
            ai_est = data.get('ai_estimate', {})
            target_delta_str = ai_est.get('target_delta_pct', '0%')

            # Parse target_delta: convert "0.00%" string to numeric value
            try:
                target_delta_numeric = float(target_delta_str.replace('%', '').strip())
            except (ValueError, AttributeError):
                target_delta_numeric = 0.0

            confidence = ai_est.get('confidence', 'Unknown')

            logger.info(
                f"✓ Fetched complete analysis for {ticker}: "
                f"target={target_delta_numeric:.2f}%, confidence={confidence}"
            )

            # Store parsed numeric value back in data for easier access
            if 'ai_estimate' in data:
                data['ai_estimate']['target_delta_numeric'] = target_delta_numeric

            return data

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
