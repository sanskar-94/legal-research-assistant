"""Hybrid search combining semantic and keyword retrieval with query classification."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from config.settings import RRF_K, SEMANTIC_TOP_K

logger = logging.getLogger(__name__)

# A party name: capitalised words, optionally joined by "of" / "and" / "&",
# e.g. "Maneka Gandhi", "K.S. Puttaswamy", "State of Kerala".
_PARTY = r"[A-Z][\w.'&-]*(?:\s+(?:(?:of|and|&)\s+)?[A-Z][\w.'&-]*)*"
# "Maneka Gandhi v Union of India", "X vs. Y", "X versus Y"
_CASE_NAME_PATTERN = re.compile(
    rf"(?P<petitioner>{_PARTY})\s+(?:v|vs|versus)\.?\s+(?P<respondent>{_PARTY})"
)
# Capitalised words that start a question rather than a party name
_LEADING_WORDS = frozenset({
    "what", "explain", "summarize", "summarise", "describe", "discuss", "define",
    "compare", "how", "why", "which", "who", "when", "where", "tell", "give", "list",
    "is", "was", "in", "the",
})
# Parties that name no particular case, and statutes compared with "vs"
_GENERIC_PARTY = re.compile(
    r"^(?:the\s+)?(?:union|state|government|govt|commissioner|collector"
    r"|ipc|bns|bnss|crpc|cpc|constitution)\b",
    re.IGNORECASE,
)

# Query classification patterns
_SECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bSection\s+\d+", re.IGNORECASE),
    re.compile(r"\bArticle\s+\d+", re.IGNORECASE),
    re.compile(r"\bAct[,\s]", re.IGNORECASE),
    re.compile(r"\bCode\s+of\b", re.IGNORECASE),
    re.compile(r"\bIPC\b"),
    re.compile(r"\bCrPC\b"),
    re.compile(r"\bCPC\b"),
    re.compile(r"\bBNS\b"),
)

# A query naming a case (see ``_case_party_names``) is also a case lookup.
_CITATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bAIR\s+\d{4}\b"),
    re.compile(r"\bSCC\s+\d+\b", re.IGNORECASE),
    re.compile(r"\(\d{4}\)\s+\d+\s+SCC\b"),
)

# Nearest exact matches fetched per term spelling. ``$contains`` also
# matches longer identifiers ("Article 14" in "Article 141"), which are
# dropped afterwards, so this is well above the number finally kept.
_KEYWORD_POOL_SIZE = 100

# Weight presets per classification  (semantic_weight, keyword_weight)
_CLASSIFICATION_WEIGHTS: dict[str, tuple[float, float]] = {
    "section_lookup": (0.3, 0.7),
    "case_lookup": (0.2, 0.8),
    "conceptual": (0.8, 0.2),
    "mixed": (0.5, 0.5),
}


def _case_party_names(query: str) -> list[str]:
    """Return the distinctive party names of any case named in *query*.

    "What was held in Maneka Gandhi v Union of India?" gives
    ``["Maneka Gandhi"]``: generic parties such as "Union of India" or
    "State of Kerala" occur in thousands of judgments and are skipped.
    """
    names: list[str] = []
    for match in _CASE_NAME_PATTERN.finditer(query):
        for party in (match.group("petitioner"), match.group("respondent")):
            words = party.split()
            while words and words[0].lower() in _LEADING_WORDS:
                words = words[1:]
            name = " ".join(words).rstrip(".,'&-")
            if not name or _GENERIC_PARTY.match(name):
                continue
            names.append(name)
            # Judgments write initials with dots: "MC Mehta" -> "M.C. Mehta"
            initials = re.match(r"([A-Z]{2,3})\s+(?=[A-Z])", name)
            if initials:
                dotted = "".join(f"{letter}." for letter in initials.group(1))
                names.append(f"{dotted} {name[initials.end():]}")
    return names


def _extract_keyword_terms(query: str) -> list[str]:
    """Pull out specific legal identifiers for keyword matching.

    Returns a list of distinct search phrases extracted from the query,
    such as section numbers, article numbers, case party names, act
    names, and case citation fragments.
    """
    terms: list[str] = _case_party_names(query)

    # Section / Article numbers, normalised to "Section 304A", "Article 21A"
    for match in re.finditer(r"\b(Section|Article)\s+(\d+[A-Za-z]*)", query, re.IGNORECASE):
        terms.append(f"{match.group(1).capitalize()} {match.group(2).upper()}")

    # AIR / SCC citations
    for match in re.finditer(r"(AIR\s+\d{4}\s+\w+\s+\d+)", query):
        terms.append(match.group(0))
    for match in re.finditer(r"(\(\d{4}\)\s+\d+\s+SCC\s+\d+)", query):
        terms.append(match.group(0))
    for match in re.finditer(r"(SCC\s+\d+)", query, re.IGNORECASE):
        terms.append(match.group(0))

    # A bare act abbreviation matches a large part of the corpus, so it is
    # only used when the query names no specific section or citation.
    if not terms:
        for abbr in ("IPC", "CrPC", "CPC", "BNS"):
            if re.search(rf"\b{abbr}\b", query):
                terms.append(abbr)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        normalised = term.strip().lower()
        if normalised not in seen:
            seen.add(normalised)
            unique.append(term.strip())
    return unique


def _case_variants(term: str) -> list[str]:
    """ChromaDB ``$contains`` is case-sensitive, so try the usual spellings.

    Judgments write both "Section 302" and "section 302"; party names,
    citations and acronyms are searched as written.
    """
    if term.startswith(("Section ", "Article ")):
        return [term, term[:1].lower() + term[1:]]
    return [term]


def _whole_term_pattern(term: str) -> re.Pattern[str]:
    """Match *term* as a whole identifier: "Article 21" but not "Article 210"."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _hits_from_query_result(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a ChromaDB ``query`` response for one query embedding."""
    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    hits: list[dict[str, Any]] = []
    for idx, chunk_id in enumerate(ids):
        # The collection uses cosine space, so distance = 1 - cosine similarity.
        distance = distances[idx] if idx < len(distances) else 1.0
        hits.append({
            "chunk_id": chunk_id,
            "text": documents[idx] if idx < len(documents) else "",
            "metadata": metadatas[idx] if idx < len(metadatas) else {},
            "semantic_score": max(0.0, 1.0 - distance),
        })
    return hits


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """Compute RRF scores from multiple ranked result lists.

    Each list contains chunk_ids ordered by descending relevance.
    Returns a mapping of chunk_id -> rrf_score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)


def fuse_hits(
    semantic_hits: list[dict[str, Any]],
    keyword_hits: list[dict[str, Any]],
    sem_weight: float,
    kw_weight: float,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Weighted RRF of the dense and exact-match lists (eq. 3 and 4).

    ``R(c) = alpha * RRF_sem(c) + beta * RRF_kw(c)`` with
    ``RRF_L(c) = 1 / (k + r_L(c))`` and ``RRF_L(c) = 0`` when ``c`` is not in
    ``L``. Every chunk of either list is kept (a chunk found by both keeps
    the dense hit's fields); the result is sorted by ``rrf_score``,
    descending, with ties left in dense-then-exact order.
    """
    semantic_ranked = [h["chunk_id"] for h in semantic_hits]
    keyword_ranked = [h["chunk_id"] for h in keyword_hits]

    # Collect every hit by chunk_id (prefer semantic result for score)
    hits_by_id: dict[str, dict[str, Any]] = {}
    for hit in semantic_hits:
        hits_by_id.setdefault(hit["chunk_id"], {**hit})
    for hit in keyword_hits:
        if hit["chunk_id"] not in hits_by_id:
            hits_by_id[hit["chunk_id"]] = {**hit}

    semantic_rrf = _reciprocal_rank_fusion([semantic_ranked], k=k)
    keyword_rrf = _reciprocal_rank_fusion([keyword_ranked], k=k)

    merged: list[dict[str, Any]] = [
        {
            **hit,
            "rrf_score": (
                sem_weight * semantic_rrf.get(chunk_id, 0.0)
                + kw_weight * keyword_rrf.get(chunk_id, 0.0)
            ),
        }
        for chunk_id, hit in hits_by_id.items()
    ]
    merged.sort(key=lambda r: r["rrf_score"], reverse=True)
    return merged


class HybridSearch:
    """Dual-path retrieval combining semantic and keyword search with
    query-adaptive weighting and Reciprocal Rank Fusion."""

    def __init__(self, vectorstore: Any, embedding_service: Any) -> None:
        """Initialise hybrid search.

        Parameters
        ----------
        vectorstore:
            A ChromaDB collection (or compatible wrapper) exposing
            ``query`` and ``get`` methods.
        embedding_service:
            An object with an ``embed_query(text) -> list[float]`` method.
        """
        self._vectorstore = vectorstore
        self._embedding_service = embedding_service

    # Query classification

    @staticmethod
    def classify_query(query: str) -> str:
        """Classify *query* into a retrieval strategy bucket.

        Returns one of:
            ``"section_lookup"`` | ``"case_lookup"`` |
            ``"conceptual"``     | ``"mixed"``
        """
        has_section = any(p.search(query) for p in _SECTION_PATTERNS)
        has_case = bool(_case_party_names(query)) or any(
            p.search(query) for p in _CITATION_PATTERNS
        )

        if has_section and has_case:
            return "mixed"
        if has_section:
            return "section_lookup"
        if has_case:
            return "case_lookup"
        return "conceptual"

    # Semantic search

    def _semantic_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Retrieve nearest neighbours of *query_embedding* from ChromaDB."""
        return _hits_from_query_result(self._vectorstore.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        ))

    # Keyword search

    def _keyword_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Exact-match retrieval of legal identifiers via ``where_document``.

        Only identifiers such as "Section 302", "Article 21", a case party
        name or an AIR/SCC citation are searched; a conceptual query with
        none of these skips the keyword path. Among the chunks containing a
        term, the ones nearest to the query embedding are taken, so common
        identifiers that occur in thousands of judgments still return the
        most relevant matches.
        """
        terms = _extract_keyword_terms(query)
        if not terms:
            return []

        all_hits: dict[str, dict[str, Any]] = {}
        for term in terms:
            whole_term = _whole_term_pattern(term)
            for variant in _case_variants(term):
                try:
                    results = self._vectorstore.query(
                        query_embeddings=[query_embedding],
                        where_document={"$contains": variant},
                        n_results=max(top_k, _KEYWORD_POOL_SIZE),
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception:
                    logger.debug("Keyword search failed for '%s'", variant, exc_info=True)
                    continue

                for hit in _hits_from_query_result(results):
                    if hit["chunk_id"] not in all_hits and whole_term.search(hit["text"]):
                        all_hits[hit["chunk_id"]] = hit

        hits = sorted(all_hits.values(), key=lambda h: h["semantic_score"], reverse=True)
        return hits[:top_k]

    # Public search methods

    def search_paths(self, query: str, top_k: int = SEMANTIC_TOP_K) -> dict[str, Any]:
        """Run both retrieval paths once, without fusing them.

        Returns ``{"classification", "semantic_hits", "keyword_hits"}``:
        the query type, the ``top_k`` nearest chunks, and the ``top_k``
        nearest chunks that contain one of the query's legal identifiers.
        ``search`` fuses these; ``scripts/evaluation/retrieval_experiments.py`` caches
        them so every fusion and re-ranking setting sees the same candidates.
        """
        classification = self.classify_query(query)
        query_embedding: list[float] = self._embedding_service.embed_query(query)
        semantic_hits = self._semantic_search(query_embedding, top_k)
        keyword_hits = self._keyword_search(query, query_embedding, top_k)

        logger.debug(
            "Semantic hits: %d, keyword hits: %d",
            len(semantic_hits), len(keyword_hits),
        )
        return {
            "classification": classification,
            "semantic_hits": semantic_hits,
            "keyword_hits": keyword_hits,
        }

    def search(self, query: str, top_k: int = SEMANTIC_TOP_K) -> list[dict[str, Any]]:
        """Execute hybrid search and return merged results sorted by RRF score.

        Parameters
        ----------
        query:
            Natural-language legal query.
        top_k:
            Maximum number of candidates to retrieve from *each* path
            before fusion.

        Returns
        -------
        list[dict]
            Merged result dicts containing ``chunk_id``, ``text``,
            ``metadata``, ``semantic_score``, and ``rrf_score``, sorted
            descending by weighted RRF score.
        """
        classification = self.classify_query(query)
        sem_weight, kw_weight = _CLASSIFICATION_WEIGHTS[classification]
        logger.info(
            "Query classified as '%s' - semantic=%.1f, keyword=%.1f",
            classification, sem_weight, kw_weight,
        )

        paths = self.search_paths(query, top_k)
        merged = fuse_hits(
            paths["semantic_hits"], paths["keyword_hits"], sem_weight, kw_weight, k=RRF_K,
        )

        logger.info("Hybrid search returned %d merged candidates", len(merged))
        return merged
