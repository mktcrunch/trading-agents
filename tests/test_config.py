"""Smoke tests for core configuration (no live API calls)."""
from src import config


def test_gemini_model_id_is_35_flash():
    assert config.GEMINI_FLASH_MODEL == "gemini-3.5-flash"


def test_vertex_gemini_location_defaults_global():
    assert config.GEMINI_VERTEX_LOCATION == "global"


def test_gcp_region_defaults_us_central1():
    assert config.GCP_REGION == "us-central1"


def test_learning_enabled_by_default():
    assert config.LEARNING_ENABLED is True


def test_baseline_google_search_grounding_default():
    assert config.BASELINE_GOOGLE_SEARCH_GROUNDING is True
