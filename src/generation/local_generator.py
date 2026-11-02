"""Nyaya-Gen: the locally fine-tuned answer writer (SmolLM2-135M-Instruct).

Drop-in replacement for ``LLMClient``: same ``generate(prompt, system_prompt)``
call, but the model runs on this machine instead of an API. Select it with
``LLM_BACKEND=local`` in ``.env``. Training: ``scripts/nyaya_gen/train_generator.py``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from config.settings import (
    LOCAL_GENERATOR_MAX_NEW_TOKENS,
    LOCAL_GENERATOR_PATH,
    LOCAL_GENERATOR_SOURCE_WORDS,
)

logger = logging.getLogger(__name__)

# Text of one source inside the user prompt (see SOURCE_FORMAT_TEMPLATE).
_SOURCE_BODY = re.compile(r"(---\n)(.*?)(\n---\n)", re.DOTALL)


class LocalGeneratorError(Exception):
    """Raised when the local model is missing or generation fails."""


def shorten_sources(prompt: str, max_words: int) -> str:
    """Cut every source passage to *max_words* words, as in the training data."""

    def _cut(match: re.Match) -> str:
        words = match.group(2).split()
        body = match.group(2) if len(words) <= max_words else " ".join(words[:max_words]) + " ..."
        return match.group(1) + body + match.group(3)

    return _SOURCE_BODY.sub(_cut, prompt)


class LocalGenerator:
    """Runs Nyaya-Gen with Hugging Face ``transformers`` on MPS or CPU."""

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None) -> None:
        self.model_path = Path(model_path or LOCAL_GENERATOR_PATH)
        if not (self.model_path / "config.json").exists():
            raise LocalGeneratorError(
                f"No trained model at {self.model_path}. Run scripts/nyaya_gen/train_generator.py first."
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device).eval()
        self.model_name = f"Nyaya-Gen ({self.model_path.name})"
        logger.info("Local generator loaded from %s on %s", self.model_path, self.device)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not prompt:
            raise LocalGeneratorError("Prompt must not be empty.")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": shorten_sources(prompt, LOCAL_GENERATOR_SOURCE_WORDS)})
        text = self._tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self._tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=LOCAL_GENERATOR_MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=self._tok.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=self._tok.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self._tok.decode(new_tokens, skip_special_tokens=True).strip()
