"""
PostgreSQL database client
Fetches historical ticker data and features
"""
from typing import Optional, Dict, List, Any
import pandas as pd
from urllib.parse import quote

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)


class PostgresClient:
    """
    Client for PostgreSQL database
    Accesses ticker_historical_data and ticker_historical_features tables
    """

    def __init__(self):
        """Initialize database connection"""
        if not PSYCOPG2_AVAILABLE:
            logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
            raise ImportError("psycopg2 not installed")

        # Build connection string with URL-encoded password
        db_password = quote(config.DB_PASSWORD, safe='')
        self.connection_string = f"postgresql://{config.DB_USER}:{db_password}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            logger.info(f"✓ Connected to PostgreSQL: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self.conn = None

    def execute_query(self, query: str, params: tuple = None) -> Optional[List[Dict]]:
        """
        Execute a SELECT query

        Returns:
            List of dictionaries (rows)
        """
        if not self.conn:
            logger.warning("Database not connected")
            return None

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params or ())
                results = cursor.fetchall()
                return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None

    def get_historical_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for a ticker

        Args:
            ticker: Stock symbol
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        query = """
        SELECT
            date,
            open,
            high,
            low,
            close,
            volume
        FROM ticker_historical_data
        WHERE ticker = %s
        AND date BETWEEN %s AND %s
        ORDER BY date ASC
        """

        try:
            results = self.execute_query(query, (ticker, start_date, end_date))
            if results:
                df = pd.DataFrame(results)
                logger.info(f"✓ Retrieved {len(df)} historical rows for {ticker}")
                return df
            else:
                logger.warning(f"No historical data found for {ticker} between {start_date} and {end_date}")
                return None

        except Exception as e:
            logger.error(f"Failed to get historical data for {ticker}: {e}")
            return None

    def get_features(
        self,
        ticker: str,
        date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get technical features for a ticker on a specific date

        Args:
            ticker: Stock symbol
            date: YYYY-MM-DD

        Returns:
            Dictionary of features
        """
        query = """
        SELECT *
        FROM ticker_historical_features
        WHERE ticker = %s
        AND date = %s
        LIMIT 1
        """

        try:
            results = self.execute_query(query, (ticker, date))
            if results:
                features = dict(results[0])
                logger.info(f"✓ Retrieved features for {ticker} on {date}")
                return features
            else:
                logger.debug(f"No features found for {ticker} on {date}")
                return {}

        except Exception as e:
            logger.error(f"Failed to get features for {ticker}: {e}")
            return None

    def get_latest_close(self, ticker: str) -> Optional[float]:
        """
        Get the latest closing price for a ticker

        Returns:
            Close price or None
        """
        query = """
        SELECT close
        FROM ticker_historical_data
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            results = self.execute_query(query, (ticker,))
            if results:
                close = float(results[0]['close'])
                return close
            else:
                logger.warning(f"No price data found for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Failed to get latest close for {ticker}: {e}")
            return None

    def get_ohlcv_lookback(
        self,
        ticker: str,
        lookback_days: int = 90,
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV data for the last N calendar days."""
        from datetime import datetime, timedelta

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        return self.get_historical_data(ticker, start_date, end_date)

    def get_price_series(
        self,
        ticker: str,
        lookback_days: int = 90
    ) -> Optional[pd.Series]:
        """
        Get price series for technical analysis

        Args:
            ticker: Stock symbol
            lookback_days: Number of days back

        Returns:
            Series of close prices (indexed by date)
        """
        try:
            df = self.get_ohlcv_lookback(ticker, lookback_days)
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                logger.info(f"✓ Retrieved {len(df)} price points for {ticker}")
                return df["close"]

            logger.warning(f"No price data found for {ticker}")
            return None
        except Exception as e:
            logger.error(f"Failed to get price series for {ticker}: {e}")
            return None

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            query = "SELECT 1"
            result = self.execute_query(query)
            if result is not None:
                logger.info("✓ PostgreSQL connection successful")
                return True
            else:
                logger.error("✗ PostgreSQL connection test failed")
                return False
        except Exception as e:
            logger.error(f"✗ PostgreSQL connection failed: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("✓ Closed PostgreSQL connection")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test the client
    try:
        with PostgresClient() as db:
            print("Testing PostgreSQL connection...")
            if db.test_connection():
                print("✓ Connection successful!\n")

                # Try to fetch some data
                print("Sample data fetch:")
                for ticker in ["SPY", "QQQ"]:
                    close = db.get_latest_close(ticker)
                    if close:
                        print(f"  {ticker}: ${close:.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")
