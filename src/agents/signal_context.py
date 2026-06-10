"""Shared context helpers for signal generation (news, etc.)."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from src.logger import setup_logger

logger = setup_logger(__name__)


def fetch_signal_news(
    tickers: List[str],
    max_articles_per_ticker: int = 3,
) -> Dict[str, Any]:
    """Fetch recent headlines for the tradable universe (Alpaca primary, Yahoo fallback)."""
    from src.adk.tools.alpaca_tools import get_recent_news

    tickers_str = ",".join(tickers)
    result = get_recent_news(tickers=tickers_str, max_articles_per_ticker=max_articles_per_ticker)
    news = result.get("news") or {}
    article_count = sum(
        len(articles) for articles in news.values() if isinstance(articles, list)
    )
    tickers_with_news = sum(
        1
        for articles in news.values()
        if isinstance(articles, list)
        and articles
        and not (len(articles) == 1 and articles[0].get("error"))
    )
    logger.info(
        f"Signal news: {article_count} articles across "
        f"{tickers_with_news}/{len(tickers)} tickers"
    )
    return result


def format_news_block(news_data: Dict[str, Any]) -> str:
    """Compact news section for ledger prompts."""
    if not news_data:
        return "No recent news available."
    return json.dumps(news_data, indent=2)
