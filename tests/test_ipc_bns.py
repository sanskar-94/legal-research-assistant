"""Tests for the IPC/BNS lookup in src/retrieval/ipc_bns.py."""

from __future__ import annotations

import pytest

from src.retrieval.ipc_bns import MAPPING_CHUNK_ID, IpcBnsMapper


@pytest.fixture(scope="module")
def mapper() -> IpcBnsMapper:
    return IpcBnsMapper()


def _ipc_sections(rows: list[dict]) -> list[str]:
    return [row["ipc_section"] for row in rows]


class TestFind:

    @pytest.mark.parametrize(
        "query",
        [
            "What is Section 302 IPC?",
            "What is Section 302 of IPC and its BNS equivalent?",
            "Explain section 302 of the Indian Penal Code",
            "IPC 302",
            "IPC Section 302 punishment",
            "302 IPC",
        ],
    )
    def test_ipc_reference_forms(self, mapper: IpcBnsMapper, query: str) -> None:
        rows = mapper.find(query)
        assert _ipc_sections(rows) == ["302"]
        assert rows[0]["bns_sections"] == ["103"]
        assert rows[0]["matched_on"] == "ipc"

    def test_lettered_section(self, mapper: IpcBnsMapper) -> None:
        rows = mapper.find("Define Section 498A of IPC")
        assert _ipc_sections(rows) == ["498A"]
        assert rows[0]["bns_sections"] == ["85"]

    def test_bns_to_ipc(self, mapper: IpcBnsMapper) -> None:
        rows = mapper.find("Explain BNS Section 103")
        assert _ipc_sections(rows) == ["302"]
        assert rows[0]["matched_on"] == "bns"

    def test_bns_subsection(self, mapper: IpcBnsMapper) -> None:
        rows = mapper.find("Punishment under Section 318(4) BNS")
        assert _ipc_sections(rows) == ["420"]

    def test_several_sections(self, mapper: IpcBnsMapper) -> None:
        rows = mapper.find("Difference between Section 302 and Section 307 IPC")
        assert set(_ipc_sections(rows)) == {"302", "307"}

    def test_topic_match_when_both_codes_named(self, mapper: IpcBnsMapper) -> None:
        rows = mapper.find("Compare IPC and BNS provisions on murder")
        assert "302" in _ipc_sections(rows)
        assert all(row["matched_on"] == "topic" for row in rows)

    def test_no_match_for_constitution(self, mapper: IpcBnsMapper) -> None:
        assert mapper.find("What does Article 21 say?") == []

    def test_bare_section_ignored_without_act(self, mapper: IpcBnsMapper) -> None:
        assert mapper.find("What is Section 302?") == []

    def test_unknown_section(self, mapper: IpcBnsMapper) -> None:
        assert mapper.find("Section 999 IPC") == []

    def test_omitted_offence(self, mapper: IpcBnsMapper) -> None:
        rows = mapper.find("Is Section 497 IPC still valid?")
        assert rows[0]["bns_sections"] == []


class TestCorrectedEntries:
    """Rows that were wrong in the original table."""

    @pytest.mark.parametrize(
        ("ipc", "bns"),
        [
            ("376", "64"),
            ("364", "140(1)"),
            ("320", "116"),
            ("186", "221"),
            ("170", "204"),
            ("166A", "199"),
            ("489A", "178"),
        ],
    )
    def test_mapping(self, mapper: IpcBnsMapper, ipc: str, bns: str) -> None:
        assert mapper.ipc_to_bns(ipc)["bns_sections"] == [bns]


class TestExpansionAndSource:

    def test_expand_query_adds_counterpart(self, mapper: IpcBnsMapper) -> None:
        query = "What is Section 302 IPC?"
        expanded = mapper.expand_query(query, mapper.find(query))
        assert expanded.startswith(query)
        assert "Section 103 BNS" in expanded

    def test_expand_query_without_matches(self, mapper: IpcBnsMapper) -> None:
        assert mapper.expand_query("Article 21", []) == "Article 21"

    def test_source_chunk_shape(self, mapper: IpcBnsMapper) -> None:
        chunk = mapper.as_source_chunk(mapper.find("Section 420 IPC"))
        assert chunk["chunk_id"] == MAPPING_CHUNK_ID
        assert chunk["metadata"]["doc_type"] == "mapping"
        assert "IPC Section 420" in chunk["text"]
        assert "BNS Section 318(4)" in chunk["text"]

    def test_source_chunk_for_omitted_offence(self, mapper: IpcBnsMapper) -> None:
        chunk = mapper.as_source_chunk(mapper.find("Section 377 IPC"))
        assert "no BNS equivalent" in chunk["text"]

    def test_no_source_chunk_without_matches(self, mapper: IpcBnsMapper) -> None:
        assert mapper.as_source_chunk([]) is None
