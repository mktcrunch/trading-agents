"""
Alpaca Trading API client
Supports both paper trading accounts (baseline and internal)
"""
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
import json

import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOrdersRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from src import config
from src.logger import setup_logger
from src.models.position import Position
from src.models.order import Order, OrderType, OrderStatus as OrderStatusEnum, OrderTimeInForce

logger = setup_logger(__name__)


def round_limit_price(price: float) -> float:
    """
    Round to a valid Alpaca limit price increment.
    Stocks >= $1 use penny ticks; stocks < $1 use sub-penny ticks.
    """
    if price >= 1.0:
        return round(price, 2)
    return round(price, 4)


class AlpacaClient:
    """
    Wrapper for Alpaca Trading API
    Handles both paper trading accounts
    """

    def __init__(self, system: str = "baseline"):
        """
        Initialize Alpaca client

        Args:
            system: "baseline" or "internal"
        """
        if not ALPACA_AVAILABLE:
            logger.error("Alpaca SDK not installed. Run: pip install alpaca-trade-api")
            raise ImportError("alpaca-trade-api not installed")

        self.system = system

        if system == "baseline":
            api_key = config.ALPACA_API_KEY_BASELINE
            secret_key = config.ALPACA_SECRET_KEY_BASELINE
        elif system == "internal":
            api_key = config.ALPACA_API_KEY_INTERNAL
            secret_key = config.ALPACA_SECRET_KEY_INTERNAL
        else:
            raise ValueError(f"Invalid system: {system}")

        self.client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True  # Always use paper trading
        )
        self._api_key = api_key
        self._secret_key = secret_key
        self._data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

        logger.info(f"✓ Initialized Alpaca client for {system} system")

    def get_account(self) -> Optional[Dict[str, Any]]:
        """Get account information"""
        try:
            account = self.client.get_account()
            return {
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "status": getattr(account, 'account_status', getattr(account, 'status', 'unknown')),
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    def get_positions(self) -> Dict[str, Position]:
        """
        Get all open positions

        Returns:
            Dict mapping ticker -> Position object
        """
        try:
            positions_data = self.client.get_all_positions()
            positions = {}

            for pos in positions_data:
                position = Position(
                    ticker=pos.symbol,
                    qty=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    entry_date=datetime.now(),  # Alpaca doesn't always provide this
                )
                positions[pos.symbol] = position

            logger.info(f"Retrieved {len(positions)} open positions")
            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get a specific position"""
        try:
            pos = self.client.get_open_position(ticker)
            if pos:
                return Position(
                    ticker=pos.symbol,
                    qty=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    entry_date=datetime.now(),
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get position for {ticker}: {e}")
            return None

    def place_market_order(
        self,
        ticker: str,
        qty: float,
        side: str = "buy",  # "buy" or "sell"
        time_in_force: str = "day"
    ) -> Optional[str]:
        """
        Place a market order

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            time_in_force_enum = TimeInForce(time_in_force)

            order_request = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=time_in_force_enum
            )

            order = self.client.submit_order(order_request)
            logger.info(f"✓ Placed market {side} order: {qty} {ticker} (ID: {order.id})")
            return str(order.id)

        except Exception as e:
            logger.error(f"Failed to place market order for {ticker}: {e}")
            return None

    def place_limit_order(
        self,
        ticker: str,
        qty: float,
        limit_price: float,
        side: str = "buy",  # "buy" or "sell"
        time_in_force: str = "day"
    ) -> Optional[str]:
        """
        Place a limit order

        Args:
            ticker: Stock symbol
            qty: Quantity
            limit_price: Limit price
            side: "buy" or "sell"
            time_in_force: "day", "gtc", "opg", "cls"

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            time_in_force_enum = TimeInForce(time_in_force)
            limit_price = round_limit_price(limit_price)

            order_request = LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                limit_price=limit_price,
                time_in_force=time_in_force_enum
            )

            order = self.client.submit_order(order_request)
            logger.info(
                f"✓ Placed limit {side} order: {qty} {ticker} @ ${limit_price:.2f} "
                f"(ID: {order.id}, TIF: {time_in_force})"
            )
            return str(order.id)

        except Exception as e:
            logger.error(f"Failed to place limit order for {ticker}: {e}")
            return None

    def get_orders(self, status: str = "open") -> List[Any]:
        """
        Get orders

        Args:
            status: "open", "closed", "all"

        Returns:
            List of orders
        """
        try:
            request = GetOrdersRequest(status=status, limit=100)
            orders = self.client.get_orders(request)
            logger.info(f"Retrieved {len(orders)} {status} orders")
            return orders

        except Exception as e:
            logger.error(f"Failed to get {status} orders: {e}")
            return []

    def get_order(self, order_id: str) -> Optional[Any]:
        """Get a specific order"""
        try:
            order = self.client.get_order_by_id(order_id)
            return order
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"✓ Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get the latest trade price for a ticker."""
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            request = StockLatestTradeRequest(symbol_or_symbols=ticker)
            res = self._data_client.get_stock_latest_trade(request)
            if ticker in res:
                return float(res[ticker].price)
            return None
        except Exception as e:
            logger.error(f"Failed to get latest price for {ticker}: {e}")
            return None

    def get_recent_volatility(self, ticker: str, minutes: int = 30) -> Optional[Dict[str, float]]:
        """
        Calculate price volatility metrics for a ticker over the last N minutes.
        
        Returns a dict with:
            - 'std_dev_pct': standard deviation of 1-minute returns (as %)
            - 'range_pct': total high-low range over the period (as % of average price)
            - 'sample_count': number of 1-minute bars retrieved
        """
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            import numpy as np

            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=minutes + 5) # Add buffer to ensure enough bars
            
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            barset = self._data_client.get_stock_bars(request)
            bars = barset.data.get(ticker, [])
            if not bars or len(bars) < 5:
                return None
                
            df = self._bars_to_dataframe(bars)
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Calculate 1-minute returns
            returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([])
            std_dev_pct = float(np.std(returns) * 100) if len(returns) > 0 else 0.0
            
            # Calculate high-low range percentage
            max_high = float(np.max(highs))
            min_low = float(np.min(lows))
            avg_price = float(np.mean(closes))
            range_pct = float((max_high - min_low) / avg_price * 100) if avg_price > 0 else 0.0
            
            return {
                "std_dev_pct": std_dev_pct,
                "range_pct": range_pct,
                "sample_count": len(closes)
            }
        except Exception as e:
            logger.error(f"Failed to calculate recent volatility for {ticker}: {e}")
            return None

    def _bars_to_dataframe(self, bars: List[Any]) -> pd.DataFrame:
        """Convert Alpaca Bar objects to an OHLCV DataFrame."""
        return pd.DataFrame(
            [
                {
                    "date": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
        )

    def get_historical_bars(
        self,
        ticker: str,
        lookback_days: int = 90,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV bars for a single ticker."""
        result = self.get_historical_bars_batch([ticker], lookback_days=lookback_days)
        return result.get(ticker)

    def get_historical_bars_batch(
        self,
        tickers: List[str],
        lookback_days: int = 90,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch daily OHLCV bars for multiple tickers."""
        if not tickers:
            return {}

        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=lookback_days)
            request = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            barset = self._data_client.get_stock_bars(request)
            price_data = {}

            for ticker in tickers:
                bars = barset.data.get(ticker, [])
                if bars:
                    price_data[ticker] = self._bars_to_dataframe(bars)

            logger.info(f"Retrieved Alpaca bars for {len(price_data)}/{len(tickers)} tickers")
            return price_data
        except Exception as e:
            logger.error(f"Failed to get historical bars from Alpaca: {e}")
            return {}

    def get_portfolio_history_series(self, since_hours: int = 168) -> List[Dict[str, Any]]:
        """Return equity history for the dashboard chart (sourced from Alpaca)."""
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        from src.agents.competition_context import STARTING_EQUITY

        if since_hours <= 168:
            period, timeframe = "1W", "1H"
        elif since_hours <= 720:
            period, timeframe = "1M", "1D"
        else:
            period, timeframe = "3M", "1D"

        try:
            hist = self.client.get_portfolio_history(
                GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
            )
        except Exception as e:
            logger.error(f"Failed to get portfolio history for {self.system}: {e}")
            return []

        points: List[Dict[str, Any]] = []
        for ts, equity in zip(hist.timestamp, hist.equity):
            if not equity or float(equity) <= 0:
                continue
            pv = float(equity)
            pnl_usd = pv - STARTING_EQUITY
            pnl_pct = (pnl_usd / STARTING_EQUITY * 100) if STARTING_EQUITY else 0.0
            points.append({
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "portfolio_value": pv,
                "pnl_pct": round(pnl_pct, 2),
                "pnl_usd": round(pnl_usd, 2),
                "source": "alpaca",
            })
        return points

    def get_calendar(self, filters: Any = None) -> List[Any]:
        """Return Alpaca exchange calendar entries for the requested date range."""
        try:
            return self.client.get_calendar(filters)
        except Exception as e:
            logger.error(f"Failed to get market calendar: {e}")
            return []

    def get_clock(self) -> Optional[Dict[str, Any]]:
        """Get market clock"""
        try:
            clock = self.client.get_clock()
            return {
                "timestamp": clock.timestamp,
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
            }
        except Exception as e:
            logger.error(f"Failed to get market clock: {e}")
            return None

    def test_connection(self) -> bool:
        """Test Alpaca API connection"""
        try:
            account = self.get_account()
            if account:
                logger.info(f"✓ Alpaca connection successful ({self.system})")
                logger.info(f"  Cash: ${account['cash']:,.2f}")
                logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
                logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
                return True
            else:
                logger.error("✗ Failed to get account info")
                return False
        except Exception as e:
            logger.error(f"✗ Alpaca connection failed: {e}")
            return False


def test_both_accounts():
    """Test both Alpaca accounts"""
    print("Testing Alpaca connections...\n")

    for system in ["baseline", "internal"]:
        print(f"Testing {system} account...")
        try:
            client = AlpacaClient(system=system)
            if client.test_connection():
                positions = client.get_positions()
                print(f"  Positions: {len(positions)}")
                if positions:
                    for ticker, pos in list(positions.items())[:3]:
                        print(f"    {ticker}: {pos.qty} @ ${pos.avg_entry_price:.2f}")
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}\n")


if __name__ == "__main__":
    test_both_accounts()
