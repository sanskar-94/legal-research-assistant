"""Tests for the Nyaya-Gen local answer writer (no model download needed)."""

import pytest

from config.prompts import SOURCE_FORMAT_TEMPLATE
from src.generation.local_generator import LocalGenerator, LocalGeneratorError, shorten_sources


def _source(index: int, text: str) -> str:
    return SOURCE_FORMAT_TEMPLATE.format(index=index, doc_type="statute", metadata_line="Act: IPC", text=text)


def test_shorten_sources_cuts_each_passage():
    long_text = " ".join(f"w{i}" for i in range(200))
    prompt = _source(1, long_text) + _source(2, "short passage here") + "\nQUESTION: q\n"
    out = shorten_sources(prompt, 10)
    assert "w9 ..." in out and "w10" not in out
    assert "short passage here" in out
    assert out.count("[SOURCE_") == 2 and "QUESTION: q" in out


def test_shorten_sources_leaves_short_prompt_unchanged():
    prompt = _source(1, "only five words in here")
    assert shorten_sources(prompt, 90) == prompt


def test_missing_model_raises(tmp_path):
    with pytest.raises(LocalGeneratorError):
        LocalGenerator(model_path=tmp_path)
