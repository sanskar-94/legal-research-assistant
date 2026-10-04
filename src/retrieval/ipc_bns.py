"""IPC to BNS section lookup used at query time.

The Bharatiya Nyaya Sanhita, 2023 replaced the Indian Penal Code, 1860 from
1 July 2024, so users still ask about IPC numbers while newer material uses
BNS numbers. This module reads ``config/ipc_bns_mapping.json`` and:

* finds IPC or BNS section references in a query,
* expands the retrieval query with the corresponding section on the other side,
* turns the matched rows into a source chunk the LLM can cite.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from config.settings import IPC_BNS_MAPPING_PATH

logger = logging.getLogger(__name__)

MAPPING_CHUNK_ID = "ipc_bns_mapping"

_SECTION_NO = r"(\d{1,3}[A-Z]{0,2}(?:\(\d+\))?)"
_IPC_NAMES = r"(?:IPC|I\.P\.C\.?|Indian\s+Penal\s+Code)"
_BNS_NAMES = r"(?:BNS|B\.N\.S\.?|Bharatiya\s+Nyaya\s+Sanhita)"
_SEC_WORD = r"(?:Section|Sec\.?|S\.)\s*"

# "Section 302 IPC", "Section 302 of the IPC", "s. 302 of Indian Penal Code"
_SECTION_THEN_ACT = r"{sec}{no}\s+(?:of\s+(?:the\s+)?)?{act}"
# "IPC Section 302", "IPC 302"
_ACT_THEN_SECTION = r"{act}\s+(?:{sec})?{no}"

_IPC_PATTERNS = (
    re.compile(_SECTION_THEN_ACT.format(sec=_SEC_WORD, no=_SECTION_NO, act=_IPC_NAMES), re.I),
    re.compile(r"\b" + _SECTION_NO + r"\s+(?:of\s+(?:the\s+)?)?" + _IPC_NAMES, re.I),
    re.compile(_ACT_THEN_SECTION.format(sec=_SEC_WORD, no=_SECTION_NO, act=_IPC_NAMES), re.I),
)
_BNS_PATTERNS = (
    re.compile(_SECTION_THEN_ACT.format(sec=_SEC_WORD, no=_SECTION_NO, act=_BNS_NAMES), re.I),
    re.compile(r"\b" + _SECTION_NO + r"\s+(?:of\s+(?:the\s+)?)?" + _BNS_NAMES, re.I),
    re.compile(_ACT_THEN_SECTION.format(sec=_SEC_WORD, no=_SECTION_NO, act=_BNS_NAMES), re.I),
)
_BARE_SECTION = re.compile(_SEC_WORD + _SECTION_NO, re.I)
_MENTIONS_IPC = re.compile(r"\b" + _IPC_NAMES + r"\b", re.I)
_MENTIONS_BNS = re.compile(r"\b" + _BNS_NAMES + r"\b", re.I)

_TOPIC_STOPWORDS = {
    "what", "which", "where", "when", "explain", "compare", "comparison", "between",
    "under", "with", "from", "into", "about", "provisions", "provision", "section",
    "sections", "equivalent", "difference", "differences", "does", "the", "and",
    "law", "laws", "code", "penal", "indian", "bharatiya", "nyaya", "sanhita", "new", "old",
}


def _base_section(section: str) -> str:
    """Strip a sub-section suffix: ``"318(4)"`` -> ``"318"``."""
    return section.split("(", 1)[0].upper()


class IpcBnsMapper:
    """Bidirectional lookup over the IPC/BNS correspondence table."""

    def __init__(self, mapping_path: Optional[Path] = None) -> None:
        path = Path(mapping_path or IPC_BNS_MAPPING_PATH)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._rows: list[dict[str, Any]] = data.get("mappings", [])
        self._ipc_act: str = data.get("metadata", {}).get("ipc_act", "Indian Penal Code, 1860")
        self._bns_act: str = data.get("metadata", {}).get("bns_act", "Bharatiya Nyaya Sanhita, 2023")

        self._by_ipc: dict[str, dict[str, Any]] = {
            row["ipc_section"].upper(): row for row in self._rows
        }
        self._by_bns: dict[str, list[dict[str, Any]]] = {}
        for row in self._rows:
            for bns in row.get("bns_sections", []):
                for key in {bns.upper(), _base_section(bns)}:
                    self._by_bns.setdefault(key, []).append(row)

        logger.info("Loaded %d IPC/BNS mapping rows from %s", len(self._rows), path)

    def __len__(self) -> int:
        return len(self._rows)

    # Lookups

    def ipc_to_bns(self, ipc_section: str) -> Optional[dict[str, Any]]:
        """Return the mapping row for an IPC section, or ``None``."""
        key = ipc_section.strip().upper()
        return self._by_ipc.get(key) or self._by_ipc.get(_base_section(key))

    def bns_to_ipc(self, bns_section: str) -> list[dict[str, Any]]:
        """Return all mapping rows whose BNS side matches *bns_section*."""
        key = bns_section.strip().upper()
        return list(self._by_bns.get(key) or self._by_bns.get(_base_section(key), []))

    def find(self, query: str) -> list[dict[str, Any]]:
        """Return mapping rows relevant to *query*.

        Each returned dict is a copy of the table row plus ``matched_on``
        (``"ipc"``, ``"bns"`` or ``"topic"``).
        """
        if not query:
            return []

        ipc_refs = _collect(query, _IPC_PATTERNS)
        bns_refs = _collect(query, _BNS_PATTERNS)

        # A bare "Section 302" counts for whichever act the query names,
        # but only when it names exactly one of them.
        mentions_ipc = bool(_MENTIONS_IPC.search(query))
        mentions_bns = bool(_MENTIONS_BNS.search(query))
        if mentions_ipc != mentions_bns:
            bare = [m.group(1) for m in _BARE_SECTION.finditer(query)]
            (ipc_refs if mentions_ipc else bns_refs).extend(bare)

        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _add(row: dict[str, Any], matched_on: str) -> None:
            if row["ipc_section"] not in seen:
                seen.add(row["ipc_section"])
                results.append({**row, "matched_on": matched_on})

        for ref in ipc_refs:
            row = self.ipc_to_bns(ref)
            if row:
                _add(row, "ipc")
        for ref in bns_refs:
            for row in self.bns_to_ipc(ref):
                _add(row, "bns")

        # "Compare IPC and BNS provisions on murder": no section number given,
        # so match on the offence name instead.
        if not results and mentions_ipc and mentions_bns:
            for row in self._match_topic(query):
                _add(row, "topic")

        return results

    def _match_topic(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        words = {
            w for w in re.findall(r"[a-z]+", query.lower())
            if len(w) > 3 and w not in _TOPIC_STOPWORDS
        }
        if not words:
            return []
        scored = []
        for row in self._rows:
            title_words = set(re.findall(r"[a-z]+", f"{row['ipc_title']} {row['bns_title']}".lower()))
            hits = len(words & title_words)
            if hits:
                scored.append((hits, row))
        scored.sort(key=lambda pair: -pair[0])
        return [row for _, row in scored[:limit]]

    # Query expansion and prompt source

    @staticmethod
    def expand_query(query: str, matches: list[dict[str, Any]]) -> str:
        """Append the counterpart section numbers so retrieval finds both texts."""
        extra: list[str] = []
        for row in matches:
            if row["matched_on"] in ("ipc", "topic"):
                extra.extend(f"Section {_base_section(s)} BNS" for s in row.get("bns_sections", []))
            if row["matched_on"] in ("bns", "topic"):
                extra.append(f"Section {row['ipc_section']} IPC")
            extra.append(row["bns_title"] if row.get("bns_sections") else row["ipc_title"])
        if not extra:
            return query
        return f"{query} ({'; '.join(dict.fromkeys(extra))})"

    def as_source_chunk(self, matches: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Render matched rows as a chunk dict shaped like a retrieval result."""
        if not matches:
            return None

        lines = [
            "Correspondence between the Indian Penal Code, 1860 and the "
            "Bharatiya Nyaya Sanhita, 2023 (in force from 1 July 2024):",
        ]
        for row in matches:
            bns = ", ".join(row.get("bns_sections", []))
            if bns:
                lines.append(
                    f"- IPC Section {row['ipc_section']} ({row['ipc_title']}) corresponds to "
                    f"BNS Section {bns} ({row['bns_title']})."
                )
            else:
                lines.append(
                    f"- IPC Section {row['ipc_section']} ({row['ipc_title']}) has no BNS "
                    f"equivalent: {row['bns_title']}."
                )

        return {
            "chunk_id": MAPPING_CHUNK_ID,
            "text": "\n".join(lines),
            "metadata": {
                "doc_id": MAPPING_CHUNK_ID,
                "doc_type": "mapping",
                "act_name": f"{self._ipc_act} / {self._bns_act}",
                "authority_score": 1.0,
            },
            "semantic_score": 0.0,
            "rrf_score": 0.0,
            "authority_score": 1.0,
            "final_score": 0.0,
        }


def _collect(query: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    refs: list[str] = []
    for pattern in patterns:
        refs.extend(m.group(1) for m in pattern.finditer(query))
    return refs
