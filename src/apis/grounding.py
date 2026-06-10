"""Gemini grounding helpers (Google Search, etc.)."""
from __future__ import annotations

from google.genai import types


def google_search_grounding_config() -> types.GenerateContentConfig:
    """Enable Grounding with Google Search on generate_content calls."""
    return types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
