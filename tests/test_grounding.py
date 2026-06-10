"""Google Search grounding helper."""
from google.genai import types

from src.apis.grounding import google_search_grounding_config


def test_google_search_grounding_config_has_tool():
    cfg = google_search_grounding_config()
    assert cfg.tools
    assert isinstance(cfg.tools[0], types.Tool)
    assert cfg.tools[0].google_search is not None
