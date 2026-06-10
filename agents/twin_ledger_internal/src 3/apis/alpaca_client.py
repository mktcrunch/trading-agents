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
            return order.id

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
            return order.id

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
