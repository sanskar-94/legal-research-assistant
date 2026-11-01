"""Shared fixtures and path bootstrapping for the test suite.

Adds the project root to sys.path so that ``config`` and ``src`` packages
are importable regardless of the working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

# Path bootstrap
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Stub python-dotenv and the openai SDK so the suite runs without them
# installed and without any network access. The LLM client tests patch
# the OpenAI object directly.
if "dotenv" not in sys.modules:
    _dotenv = ModuleType("dotenv")
    _dotenv.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["dotenv"] = _dotenv

if "openai" not in sys.modules:
    _openai = ModuleType("openai")
    _openai.OpenAI = MagicMock  # type: ignore[attr-defined]
    sys.modules["openai"] = _openai


# Sample texts

SAMPLE_JUDGEMENT_TEXT = """\
State of Maharashtra v. Prakash

Supreme Court of India
Date of Judgment: 15 March 2020
CORAM: Justice A.K. Sharma, Justice B.N. Rao

1. The appellant filed an appeal against the order of the High Court of Bombay.
The main issue before this Court relates to the applicability of Section 302 of IPC.

2. The prosecution case is that the accused committed murder on 10 January 2019.
Multiple witnesses were examined during the trial. The evidence clearly shows
that the accused had premeditated intent. This was supported by forensic analysis
and witness testimony from four different individuals who were present at the scene.

3. The learned counsel for the appellant argued that the evidence was circumstantial
and did not establish guilt beyond reasonable doubt. He relied on the judgment in
Sharad Birdhichand Sarda v State of Maharashtra (1984) 4 SCC 116.

4. We have carefully considered the submissions. The High Court rightly observed
that the chain of circumstances was complete. The evidence of motive, preparation,
and conduct leaves no doubt about the guilt of the accused.

5. In conclusion, the appeal is dismissed. The conviction under Section 302 IPC
is upheld. The sentence of life imprisonment shall stand.
"""

SAMPLE_STATUTE_TEXT = """\
Indian Penal Code, 1860

Section 299. Culpable homicide:
Whoever causes death by doing an act with the intention of causing death,
or with the intention of causing such bodily injury as is likely to cause death,
or with the knowledge that he is likely by such act to cause death,
commits the offence of culpable homicide.

Section 300. Murder:
Except in the cases hereinafter excepted, culpable homicide is murder,
if the act by which the death is caused is done with the intention of
causing death, or if the act is done with the intention of causing such
bodily injury as the offender knows to be likely to cause the death of
the person to whom the harm is caused, or if it is done with the
intention of causing bodily injury to any person and the bodily injury
intended to be inflicted is sufficient in the ordinary course of nature
to cause death, or if the person committing the act knows that it is
so imminently dangerous that it must in all probability cause death or
such bodily injury as is likely to cause death, and commits such act
without any excuse for incurring the risk of causing death or such
injury as aforesaid.

Section 302. Punishment for murder:
Whoever commits murder shall be punished with death, or imprisonment
for life, and shall also be liable to fine.

Section 304. Punishment for culpable homicide not amounting to murder:
Whoever commits culpable homicide not amounting to murder shall be
punished with imprisonment for life, or imprisonment for a term which
may extend to ten years, and shall also be liable to fine.
"""

SAMPLE_CONSTITUTION_TEXT = """\
Constitution of India

Article 14. Equality before law:
The State shall not deny to any person equality before the law or the
equal protection of the laws within the territory of India.
Explanation: This article guarantees both formal and substantive equality.

Article 19. Protection of certain rights regarding freedom of speech:
(1) All citizens shall have the right to freedom of speech and expression;
to assemble peaceably and without arms; to form associations or unions;
to move freely throughout the territory of India; to reside and settle
in any part of the territory of India; and to practise any profession,
or to carry on any occupation, trade or business.
(2) Nothing in sub clause (a) of clause (1) shall affect the operation
of any existing law.

Article 21. Protection of life and personal liberty:
No person shall be deprived of his life or personal liberty except
according to procedure established by law.
Explanation: This article has been interpreted expansively to include
the right to live with dignity, right to livelihood, right to privacy,
and other fundamental rights essential to a meaningful existence.

Article 32. Remedies for enforcement of rights conferred by this Part:
(1) The right to move the Supreme Court by appropriate proceedings for
the enforcement of the rights conferred by this Part is guaranteed.
(2) The Supreme Court shall have power to issue directions or orders or
writs for the enforcement of any of the rights conferred by this Part.
"""


# Chunk metadata factory

def make_chunk_metadata(
    doc_id: str = "test-doc-001",
    doc_type: str = "judgement",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a metadata dict suitable for chunking and retrieval tests."""
    base: dict[str, Any] = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "case_name": None,
        "citation": None,
        "court": None,
        "date": None,
        "judges": None,
        "section_refs": [],
        "page_numbers": [],
        "paragraph_numbers": [],
        "chunk_index": 0,
        "authority_score": 0.0,
        "act_name": None,
        "section_number": None,
    }
    return {**base, **overrides}


# Pytest fixtures

@pytest.fixture()
def judgement_text() -> str:
    return SAMPLE_JUDGEMENT_TEXT


@pytest.fixture()
def statute_text() -> str:
    return SAMPLE_STATUTE_TEXT


@pytest.fixture()
def constitution_text() -> str:
    return SAMPLE_CONSTITUTION_TEXT


@pytest.fixture()
def judgement_metadata() -> dict[str, Any]:
    return make_chunk_metadata(
        doc_type="judgement",
        case_name="State of Maharashtra v. Prakash",
        court="Supreme Court of India",
        date="2020-03-15",
    )


@pytest.fixture()
def statute_metadata() -> dict[str, Any]:
    return make_chunk_metadata(
        doc_type="statute",
        act_name="Indian Penal Code, 1860",
    )


@pytest.fixture()
def constitution_metadata() -> dict[str, Any]:
    return make_chunk_metadata(
        doc_type="constitution",
        act_name="Constitution of India",
    )


@pytest.fixture()
def mock_vectorstore() -> MagicMock:
    """A MagicMock that mimics a ChromaDB collection interface.

    ``query`` answers both the semantic search and the filtered exact-match
    searches with the same three chunks.
    """
    store = MagicMock()
    store.query.return_value = {
        "ids": [["chunk_1", "chunk_2", "chunk_3"]],
        "documents": [
            [
                "Section 302 IPC deals with punishment for murder.",
                "Murder is defined under Section 300 of IPC.",
                "The accused was convicted under Section 302 read with 34 IPC.",
            ]
        ],
        "metadatas": [
            [
                {"doc_type": "statute", "section_number": "302", "act_name": "IPC"},
                {"doc_type": "statute", "section_number": "300", "act_name": "IPC"},
                {"doc_type": "judgement", "court": "Supreme Court of India"},
            ]
        ],
        "distances": [[0.1, 0.3, 0.5]],
    }
    return store


@pytest.fixture()
def mock_embedding_service() -> MagicMock:
    """A MagicMock that returns a fixed-length embedding vector."""
    service = MagicMock()
    service.embed_query.return_value = [0.1] * 384
    return service


@pytest.fixture()
def sample_retrieved_chunks() -> list[dict[str, Any]]:
    """A list of chunk dicts as returned by the retrieval pipeline."""
    return [
        {
            "chunk_id": "chunk_1",
            "text": "Section 302 IPC provides punishment for murder: death or life imprisonment.",
            "doc_type": "statute",
            "act_name": "Indian Penal Code",
            "section_number": "302",
            "semantic_score": 0.9,
            "rrf_score": 0.015,
        },
        {
            "chunk_id": "chunk_2",
            "text": "In State of Maharashtra v. Prakash, the Supreme Court upheld the conviction under Section 302.",
            "doc_type": "judgement",
            "case_name": "State of Maharashtra v. Prakash",
            "court": "Supreme Court of India",
            "citation": "AIR 2020 SC 1234",
            "date": "2020-03-15",
            "semantic_score": 0.8,
            "rrf_score": 0.012,
        },
        {
            "chunk_id": "chunk_3",
            "text": "Article 21 guarantees the right to life and personal liberty.",
            "doc_type": "constitution",
            "section_number": "21",
            "semantic_score": 0.5,
            "rrf_score": 0.008,
        },
    ]
