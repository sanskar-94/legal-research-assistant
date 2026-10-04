"""LLM API wrapper with rate limiting and retry logic.

Provides a thin, immutable client around the OpenAI-compatible API
(NVIDIA NIM / DeepSeek) with token-bucket rate limiting and
exponential-backoff retries via tenacity.
"""

from __future__ import annotations

import logging
import os
import re
import time
from threading import Lock
from typing import Optional

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import (
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL,
    GEMINI_RPM_LIMIT,
    GEMINI_TEMPERATURE,
    LLM_BASE_URL,
)

load_dotenv()

logger = logging.getLogger(__name__)

_THINKING_BLOCK_PATTERN = re.compile(
    r"<think\b[^>]*>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)
_VISIBLE_REASONING_PATTERN = re.compile(
    r"^\s*(?:reasoning|thinking|thought process|chain of thought)\s*:\s*.*?"
    r"(?=\n\s*(?:answer|final answer|response)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:answer|final answer|response)\s*:\s*",
    re.IGNORECASE,
)


class GeminiClientError(Exception):
    """Raised when LLM API interaction fails after all retries."""


class _RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Tracks timestamps of recent requests and sleeps when the
    window is full, keeping throughput at or below `max_rpm`.
    """

    def __init__(self, max_rpm: int) -> None:
        self._max_rpm = max_rpm
        self._window_seconds = 60.0
        self._timestamps: list[float] = []
        self._lock = Lock()

    def wait_if_needed(self) -> None:
        """Block the caller until a request slot is available."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window_seconds
            self._timestamps = [ts for ts in self._timestamps if ts > cutoff]

            if len(self._timestamps) >= self._max_rpm:
                sleep_duration = self._timestamps[0] - cutoff
                if sleep_duration > 0:
                    logger.info(
                        "Rate limit reached (%d RPM). Sleeping %.2f seconds.",
                        self._max_rpm,
                        sleep_duration,
                    )
                    time.sleep(sleep_duration)

                now = time.monotonic()
                cutoff = now - self._window_seconds
                self._timestamps = [ts for ts in self._timestamps if ts > cutoff]

            self._timestamps.append(time.monotonic())


class GeminiClient:
    """Immutable wrapper around the OpenAI-compatible LLM API (NVIDIA NIM).

    Args:
        api_key:  API key. Falls back to ``NVIDIA_API_KEY`` then
                  ``GEMINI_API_KEY`` environment variables when *None*.
        model:    Model identifier. Defaults to the value in
                  ``config.settings.GEMINI_MODEL``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        resolved_key = (
            api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise GeminiClientError(
                "No API key provided. Set OPENROUTER_API_KEY in your "
                "environment or pass api_key explicitly."
            )

        from openai import OpenAI

        self._client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=resolved_key,
        )

        self._model_name: str = model or GEMINI_MODEL
        self._rate_limiter = _RateLimiter(max_rpm=GEMINI_RPM_LIMIT)

        logger.info(
            "GeminiClient initialised — model=%s, rpm_limit=%d, base_url=%s",
            self._model_name,
            GEMINI_RPM_LIMIT,
            LLM_BASE_URL,
        )

    # ── Public API ────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt:        The user-facing prompt text.
            system_prompt: Optional system-level instruction prepended
                           to the conversation.

        Returns:
            The generated text as a plain string.

        Raises:
            GeminiClientError: If generation fails after all retries.
        """
        if not prompt:
            raise GeminiClientError("Prompt must not be empty.")

        self._wait_for_rate_limit()
        return self._call_api(prompt, system_prompt)

    # ── Internal helpers ──────────────────────────────────────────────

    def _wait_for_rate_limit(self) -> None:
        self._rate_limiter.wait_if_needed()

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
    ) -> str:
        """Execute the LLM API call with retries."""
        try:
            messages: list[dict[str, str]] = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            completion = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=GEMINI_TEMPERATURE,
                max_tokens=GEMINI_MAX_TOKENS,
                top_p=0.95,
            )

            response_text = completion.choices[0].message.content

            if not response_text:
                raise GeminiClientError("LLM returned an empty response.")

            response_text = _strip_model_thinking(response_text)
            if not response_text:
                raise GeminiClientError("LLM returned only hidden reasoning.")

            logger.debug("LLM response received — %d characters.", len(response_text))
            return response_text

        except GeminiClientError:
            raise
        except Exception as exc:
            logger.warning("LLM API call failed (will retry): %s", exc)
            raise


def _strip_model_thinking(text: str) -> str:
    """Remove model reasoning wrappers and return only the user-facing answer."""
    cleaned = _THINKING_BLOCK_PATTERN.sub("", text).strip()
    cleaned = _VISIBLE_REASONING_PATTERN.sub("", cleaned).strip()
    cleaned = _ANSWER_PREFIX_PATTERN.sub("", cleaned).strip()
    return cleaned
