"""Pydantic v2 models for document chunks and their metadata."""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field


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
    """A single text chunk with its associated metadata."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: ChunkMetadata
