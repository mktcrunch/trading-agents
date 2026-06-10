#!/usr/bin/env python3
"""
Test all API connections
Run this to verify everything is set up correctly

Usage:
    python test_connections.py
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.logger import logger
from src import config


def test_config():
    """Test configuration loading"""
    print("\n" + "=" * 80)
    print("TESTING CONFIGURATION")
    print("=" * 80)

    try:
        config.print_config()
        logger.info("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration error: {e}")
        return False


def test_marketcrunch():
    """Test MarketCrunch API"""
    print("\n" + "=" * 80)
    print("TESTING MARKETCRUNCH API")
    print("=" * 80)

    try:
        from src.apis.marketcrunch_client import MarketCrunchClient

        client = MarketCrunchClient()
        success = client.test_connection()

        if success:
            # Try to fetch a complete analysis
            analysis = client.get_ai_estimates("AAPL")
            if analysis:
                print(f"\nComplete Analysis (AAPL):")
                try:
                    ai_est = analysis.get('ai_estimate', {})
                    technical = analysis.get('technical', {})
                    factors = analysis.get('factors', {})
                    weekly = analysis.get('weekly_range', {})

                    # AI Estimates
                    print(f"  AI Prediction:")
                    print(f"    Target Price: {ai_est.get('target_price', 'N/A')}")

                    # Parse target delta
                    target_delta_str = ai_est.get('target_delta_pct', 'N/A')
                    target_delta_num = ai_est.get('target_delta_numeric', 'N/A')
                    print(f"    Target Delta: {target_delta_num}% (raw: {target_delta_str})")
                    print(f"    Confidence: {ai_est.get('confidence', 'N/A')}")

                    # Technical Scores
                    scores = technical.get('scores', {})
                    if scores:
                        print(f"  Technical Scores:")
                        print(f"    Trend: {scores.get('trend', 'N/A')}/100")
                        print(f"    Momentum: {scores.get('momentum', 'N/A')}/100")
                        print(f"    Volatility: {scores.get('volatility', 'N/A')}/100")

                    # Positive Factors
                    pos_factors = factors.get('positive', [])
                    if pos_factors:
                        print(f"  Positive Factors: {len(pos_factors)} factors identified")
                        for f in pos_factors[:2]:
                            print(f"    • {f.get('key', 'N/A')}")

                except (ValueError, TypeError) as e:
                    print(f"  (Error parsing: {e})")
        return success

    except Exception as e:
        logger.error(f"✗ MarketCrunch test failed: {e}")
        return False


def test_alpaca():
    """Test both Alpaca accounts"""
    print("\n" + "=" * 80)
    print("TESTING ALPACA (Both Accounts)")
    print("=" * 80)

    results = {}

    for system in ["baseline", "internal"]:
        print(f"\n--- Account: {system.upper()} ---")
        try:
            from src.apis.alpaca_client import AlpacaClient

            client = AlpacaClient(system=system)
            success = client.test_connection()

            if success:
                positions = client.get_positions()
                print(f"Open positions: {len(positions)}")
                if positions:
                    for ticker in list(positions.keys())[:3]:
                        print(f"  {ticker}")

            results[system] = success

        except Exception as e:
            logger.error(f"✗ Alpaca {system} test failed: {e}")
            results[system] = False

    return all(results.values())


def test_postgres():
    """Test PostgreSQL database"""
    print("\n" + "=" * 80)
    print("TESTING POSTGRESQL")
    print("=" * 80)

    try:
        from src.apis.db_client import PostgresClient

        with PostgresClient() as db:
            success = db.test_connection()

            if success:
                # Try to fetch latest close for a ticker
                close = db.get_latest_close("SPY")
                if close:
                    print(f"\nSample data (SPY latest close): ${close:.2f}")

            return success

    except Exception as e:
        logger.error(f"✗ PostgreSQL test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  MarketCrunch Trading Agents - Connection Tests".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    results = {
        "Configuration": test_config(),
        "MarketCrunch API": test_marketcrunch(),
        "Alpaca Accounts": test_alpaca(),
        "PostgreSQL": test_postgres(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:>8}  {name}")

    print("=" * 80)

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ All tests passed! You're ready to start building.\n")
        return 0
    else:
        print("\n✗ Some tests failed. Check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
