"""ADK / Gemini model configuration (API key or Vertex AI)."""
import os

from src import config


def configure_genai_env() -> None:
    """Set environment variables ADK uses for Gemini routing."""
    if config.USE_VERTEX_AI:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
        if config.GCP_PROJECT:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", config.GCP_PROJECT)
        if config.GCP_REGION:
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", config.GCP_REGION)
    elif config.GEMINI_API_KEY:
        os.environ.setdefault("GOOGLE_API_KEY", config.GEMINI_API_KEY)


def adk_model() -> str:
    """Model string for LlmAgent definitions."""
    return config.GEMINI_FLASH_MODEL
