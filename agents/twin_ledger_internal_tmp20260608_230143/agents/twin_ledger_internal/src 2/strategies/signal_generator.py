"""
Signal Generator (SHARED)
Calculates technical indicators:
- Bollinger Bands (20-period, 2 std dev)
- MACD (12/26/9)
- RSI (14-period)
"""
from typing import Dict, Optional
import pandas as pd
import numpy as np
from src.logger import setup_logger

logger = setup_logger(__name__)


class SignalGenerator:
    """
    Generates technical indicators from price data
    Used by both baseline and internal systems
    """

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands

        Args:
            prices: Close price series
            period: SMA period (default 20)
            num_std: Number of standard deviations (default 2)

        Returns:
            Dict with 'upper', 'middle', 'lower', 'zscore'
        """
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = sma + (num_std * std)
            lower = sma - (num_std * std)

            # Z-score: how many std devs from middle?
            zscore = (prices - sma) / std

            return {
                'upper': upper,
                'middle': sma,
                'lower': lower,
                'zscore': zscore
            }
        except Exception as e:
            logger.error(f"Failed to calculate Bollinger Bands: {e}")
            return {}

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD indicator

        Args:
            prices: Close price series
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            Dict with 'macd', 'signal', 'histogram'
        """
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line

            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"Failed to calculate MACD: {e}")
            return {}

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index

        Args:
            prices: Close price series
            period: RSI period (default 14)

        Returns:
            RSI series
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return pd.Series()

    @staticmethod
    def generate_technical_indicators(df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate all technical indicators from OHLCV data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with latest values of each indicator
        """
        if df is None or len(df) < 26:
            logger.warning("Insufficient data for technical indicators")
            return {}

        close = df['close']

        # Calculate indicators
        bb = SignalGenerator.calculate_bollinger_bands(close)
        macd = SignalGenerator.calculate_macd(close)
        rsi = SignalGenerator.calculate_rsi(close)

        # Get latest values
        latest = {
            'bollinger_zscore': bb.get('zscore', pd.Series()).iloc[-1] if len(bb.get('zscore', [])) > 0 else None,
            'bollinger_upper': bb.get('upper', pd.Series()).iloc[-1] if len(bb.get('upper', [])) > 0 else None,
            'bollinger_lower': bb.get('lower', pd.Series()).iloc[-1] if len(bb.get('lower', [])) > 0 else None,
            'bollinger_middle': bb.get('middle', pd.Series()).iloc[-1] if len(bb.get('middle', [])) > 0 else None,
            'macd': macd.get('macd', pd.Series()).iloc[-1] if len(macd.get('macd', [])) > 0 else None,
            'macd_signal': macd.get('signal', pd.Series()).iloc[-1] if len(macd.get('signal', [])) > 0 else None,
            'macd_histogram': macd.get('histogram', pd.Series()).iloc[-1] if len(macd.get('histogram', [])) > 0 else None,
            'rsi_14': rsi.iloc[-1] if len(rsi) > 0 else None,
        }

        return latest

    @staticmethod
    def build_technical_data(price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Build per-ticker technical features from Alpaca OHLCV bars.

        Computes RSI, MACD, Bollinger locally and includes latest close.
        """
        technical_data = {}
        for ticker, df in price_data.items():
            indicators = SignalGenerator.generate_technical_indicators(df)
            if df is not None and not df.empty:
                indicators["close"] = float(df["close"].iloc[-1])
            technical_data[ticker] = indicators
        return technical_data
