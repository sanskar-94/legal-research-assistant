"""Generation module — LLM interaction, prompt building, and response parsing."""

from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import CitedResponse, CitedSource, ResponseParser

__all__ = [
    "GeminiClient",
    "GeminiClientError",
    "PromptBuilder",
    "ResponseParser",
    "CitedResponse",
    "CitedSource",
]


def __getattr__(name: str):
    """Lazy import for GeminiClient to avoid requiring google-generativeai at import time."""
    if name in ("GeminiClient", "GeminiClientError"):
        from src.generation.gemini_client import GeminiClient, GeminiClientError
        return GeminiClient if name == "GeminiClient" else GeminiClientError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
