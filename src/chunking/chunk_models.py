"""Pydantic v2 models for document chunks and their metadata."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk for retrieval and ranking."""

    doc_id: str
    doc_type: str  # "judgement", "statute", "constitution"
    case_name: Optional[str] = None
    citation: Optional[str] = None
    court: Optional[str] = None
    date: Optional[str] = None
    judges: Optional[list[str]] = None
    section_refs: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    paragraph_numbers: list[int] = Field(default_factory=list)
    chunk_index: int = 0
    authority_score: float = 0.0
    act_name: Optional[str] = None
    section_number: Optional[str] = None


class Chunk(BaseModel):
    """A single text chunk with its associated metadata.

    When no ``chunk_id`` is given it is derived from the document id and
    chunk index, so rebuilding the corpus upserts the same ids instead of
    adding duplicates to the vector store.
    """

    chunk_id: str
    text: str
    metadata: ChunkMetadata

    @model_validator(mode="before")
    @classmethod
    def _default_chunk_id(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("chunk_id"):
            return data
        meta = data.get("metadata")
        if isinstance(meta, ChunkMetadata):
            doc_id, index = meta.doc_id, meta.chunk_index
        elif isinstance(meta, dict):
            doc_id, index = meta.get("doc_id", "doc"), meta.get("chunk_index", 0)
        else:
            return data
        return {**data, "chunk_id": f"{doc_id}::{index}"}
