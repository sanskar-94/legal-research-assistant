"""Sentence-transformers wrapper for generating text embeddings."""

from __future__ import annotations

import logging
import math
from typing import Optional

from config.settings import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thin wrapper around a sentence-transformers model.

    Loads the model lazily on first use and exposes single-text and
    batch embedding helpers with progress logging.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or EMBEDDING_MODEL_NAME
        self._model = None
        logger.info("EmbeddingService configured with model '%s'.", self._model_name)
        # Eagerly load the model at init time so first query isn't penalised
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the sentence-transformers model, raising on failure."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s' ...", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded successfully.")
        except ImportError as exc:
            logger.error(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers package is required for EmbeddingService"
            ) from exc
        except Exception as exc:
            logger.error(
                "Failed to load embedding model '%s': %s",
                self._model_name,
                exc,
            )
            raise RuntimeError(
                f"Could not load embedding model '{self._model_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Alias for :meth:`embed_text` — used by the retrieval pipeline."""
        return self.embed_text(text)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string and return its vector.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        list[float]
            The embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text.")

        self._load_model()
        embedding = self._model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> list[list[float]]:
        """Embed a list of texts in batches with progress logging.

        Parameters
        ----------
        texts:
            The input strings to embed.
        batch_size:
            Number of texts to encode per batch.

        Returns
        -------
        list[list[float]]
            A list of embedding vectors, one per input text.
        """
        if not texts:
            logger.warning("embed_batch called with an empty list.")
            return []

        self._load_model()

        total = len(texts)
        num_batches = math.ceil(total / batch_size)
        logger.info(
            "Embedding %d texts in %d batch(es) of up to %d ...",
            total,
            num_batches,
            batch_size,
        )

        all_embeddings: list[list[float]] = []

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            logger.info(
                "  Batch %d/%d  (texts %d-%d of %d)",
                batch_idx + 1,
                num_batches,
                start + 1,
                end,
                total,
            )

            embeddings = self._model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            all_embeddings.extend(embeddings.tolist())

        logger.info("Embedding complete. Produced %d vectors.", len(all_embeddings))
        return all_embeddings
