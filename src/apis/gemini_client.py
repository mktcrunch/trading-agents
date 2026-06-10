"""Shared Gemini client — Google AI API key or Vertex AI (ADC)."""
import google.genai as genai

from src import config
from src.adk.model import configure_genai_env


def get_genai_client() -> genai.Client:
    """Return a genai Client configured for API key or Vertex AI."""
    configure_genai_env()
    if config.USE_VERTEX_AI:
        return genai.Client(
            vertexai=True,
            project=config.GCP_PROJECT,
            location=config.GEMINI_VERTEX_LOCATION or config.GCP_REGION,
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)
