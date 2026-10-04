"""Client for the answer-generation LLM.

Talks to any OpenAI-compatible chat endpoint. The project default is
DeepSeek V3.2 on NVIDIA NIM (see ``config.settings``). Requests are rate
limited with a sliding window and retried with exponential backoff.
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
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RPM_LIMIT,
    LLM_TEMPERATURE,
)

load_dotenv()

# Checked in order when no key is passed explicitly.
_API_KEY_ENV_VARS = ("NVIDIA_API_KEY", "LLM_API_KEY")

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


class LLMClientError(Exception):
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


class LLMClient:
    """Wrapper around an OpenAI-compatible chat completions API.

    Args:
        api_key:  API key. Falls back to ``NVIDIA_API_KEY`` then
                  ``LLM_API_KEY`` from the environment when *None*.
        model:    Model identifier. Defaults to ``config.settings.LLM_MODEL``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or next(
            (os.getenv(name) for name in _API_KEY_ENV_VARS if os.getenv(name)),
            None,
        )
        if not resolved_key:
            raise LLMClientError(
                "No API key provided. Set NVIDIA_API_KEY in .env "
                "or pass api_key explicitly."
            )

        from openai import OpenAI

        self._client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=resolved_key,
        )

        self._model_name: str = model or LLM_MODEL
        self._rate_limiter = _RateLimiter(max_rpm=LLM_RPM_LIMIT)

        logger.info(
            "LLMClient ready: model=%s, rpm_limit=%d, base_url=%s",
            self._model_name,
            LLM_RPM_LIMIT,
            LLM_BASE_URL,
        )

    # Public API

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
            LLMClientError: If generation fails after all retries.
        """
        if not prompt:
            raise LLMClientError("Prompt must not be empty.")

        self._wait_for_rate_limit()
        return self._call_api(prompt, system_prompt)

    # Internal helpers

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
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                top_p=0.95,
            )

            response_text = completion.choices[0].message.content

            if not response_text:
                raise LLMClientError("LLM returned an empty response.")

            response_text = _strip_model_thinking(response_text)
            if not response_text:
                raise LLMClientError("LLM returned only hidden reasoning.")

            logger.debug("LLM response received - %d characters.", len(response_text))
            return response_text

        except LLMClientError:
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
