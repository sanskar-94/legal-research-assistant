"""Generation module: LLM client, prompt building and response parsing."""

from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import CitedResponse, CitedSource, ResponseParser

__all__ = [
    "LLMClient",
    "LLMClientError",
    "PromptBuilder",
    "ResponseParser",
    "CitedResponse",
    "CitedSource",
]


def __getattr__(name: str):
    """Import the LLM client lazily so the openai package is only needed at call time."""
    if name in ("LLMClient", "LLMClientError"):
        from src.generation.llm_client import LLMClient, LLMClientError
        return LLMClient if name == "LLMClient" else LLMClientError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
