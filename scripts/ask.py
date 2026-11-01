"""Ask the legal research assistant a question from the command line.

Prints the answer with its [SOURCE_N] citations, the numbered sources, the
IPC/BNS correspondence (if any) and the confidence score. Without a
question it starts an interactive prompt, so the models load only once.

Usage:
    python scripts/ask.py "What is Section 302 IPC and its BNS equivalent?"
    python scripts/ask.py --json "What does Article 21 guarantee?"
    python scripts/ask.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
if not _VERBOSE:
    # Keep model-loading progress bars and library notices out of the answer.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")

from src.generation.prompt_builder import source_label  # noqa: E402
from src.pipeline import LegalResearchPipeline, PipelineResult  # noqa: E402

_SNIPPET_CHARS = 240
_WIDTH = 100


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _SNIPPET_CHARS else flat[:_SNIPPET_CHARS].rstrip() + "..."


def render_text(result: PipelineResult) -> str:
    """Format a pipeline result for the terminal."""
    lines: list[str] = []

    if result.answer_text:
        for paragraph in result.answer_text.split("\n"):
            lines.append(textwrap.fill(paragraph, _WIDTH) if paragraph.strip() else "")
        lines.append("")

    if result.error:
        lines += [f"Note: {result.error}", ""]

    if result.invalid_citations:
        lines += [
            "Citations that point to no source: " + ", ".join(result.invalid_citations),
            "",
        ]

    if result.ipc_bns_mappings:
        lines.append("IPC -> BNS")
        for row in result.ipc_bns_mappings:
            bns = ", ".join(row.get("bns_sections", [])) or "no equivalent"
            lines.append(f"  IPC {row['ipc_section']} ({row['ipc_title']}) -> BNS {bns}")
        lines.append("")

    if result.sources:
        cited = {source.source_index for source in result.cited_sources}
        lines.append("Sources (* = cited in the answer)")
        for index, chunk in enumerate(result.sources, start=1):
            marker = "*" if index in cited else " "
            lines.append(f"{marker} [SOURCE_{index}] {source_label(chunk)}")
            lines.append(textwrap.indent(textwrap.fill(_snippet(chunk.get("text", "")), _WIDTH - 6), "      "))
        lines.append("")

    if result.confidence:
        breakdown = ", ".join(
            f"{name.replace('_', ' ')} {value:.2f}"
            for name, value in result.confidence.breakdown.items()
        )
        lines.append(
            f"Confidence: {result.confidence.overall_score:.2f} "
            f"({result.confidence.level}) - {breakdown}"
        )

    return "\n".join(lines).rstrip()


def render_json(result: PipelineResult) -> str:
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="show pipeline logs")
    args = parser.parse_args()

    # Without -v, logging below ERROR stays disabled (see the top of this
    # file); a missing API key and similar problems are part of the result.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    render = render_json if args.json else render_text

    try:
        pipeline = LegalResearchPipeline.from_settings()
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")

    question = " ".join(args.question).strip()
    if question:
        result = pipeline.query(question)
        print(render(result))
        sys.exit(1 if result.error and not result.sources else 0)

    print("Ask a question about Indian law (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        print()
        print(render(pipeline.query(question)))


if __name__ == "__main__":
    main()
