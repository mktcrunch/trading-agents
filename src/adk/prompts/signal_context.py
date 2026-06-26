"""Runtime context appended to ADK signal instructions (direct + workflow parity)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.agents.signal_context import format_news_block

SIGNAL_JSON_APPENDIX = """
Return ONLY a JSON object with this shape:
{
  "decisions": [ ...trade objects... ],
  "no_action_rationale": "2-4 sentences — REQUIRED when decisions is empty"
}

Put trade objects in "decisions" only when action is not HOLD. Each trade object must have:
- action: BUY | SELL | CLOSE | SHORT | COVER
- ticker: symbol from universe
- size_pct: number
- confidence: 0.0–1.0
- rationale: why this trade grows portfolio value with acceptable risk
- invalidation: what would make you reverse this decision
- competitive_note: how this helps you stay #1 while compounding (not defensive cash when ahead)

When you recommend no trades (decisions = []), you MUST fill no_action_rationale with a clear explanation.
"""


def build_runtime_signal_prompt(
    instruction: str,
    *,
    competition: Dict[str, Any],
    market_lines: List[str],
    news_data: Optional[Dict[str, Any]] = None,
    learning_block: str = "",
    extra_sections: str = "",
) -> str:
    """ADK static instruction + nightly JSON context (same text production and fallback paths)."""
    parts = [instruction.strip()]
    if learning_block.strip():
        parts.append(learning_block.strip())
    if extra_sections.strip():
        parts.append(extra_sections.strip())
    parts.extend([
        "Competition context:",
        json.dumps(competition, indent=2),
        "",
        "Market data & indicators:",
        "\n".join(market_lines) if market_lines else "- (no market data)",
        "",
        "Recent news (Alpaca / fallback):",
        format_news_block(news_data or {}),
        SIGNAL_JSON_APPENDIX.strip(),
    ])
    return "\n\n".join(parts)
