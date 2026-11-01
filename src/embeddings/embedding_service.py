"""Sentence-transformers wrapper for generating text embeddings."""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thin wrapper around a sentence-transformers model.

    The model is loaded when the service is created, so the first query
    is not slowed down by it.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name or EMBEDDING_MODEL_NAME
        logger.info("Loading embedding model '%s' ...", self._model_name)
        self._model = SentenceTransformer(self._model_name)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text string and return its vector."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text.")
        return self._model.encode(text, show_progress_bar=False).tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> list[list[float]]:
        """Embed a list of texts, one vector per input text."""
        if not texts:
            return []
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        ).tolist()
