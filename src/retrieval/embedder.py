# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Biomedical document chunker and text embedder.
"""

from __future__ import annotations

import re
import numpy as np
from loguru import logger

HAS_SENTENCE_TRANSFORMERS = None  # resolved lazily on first use


def _try_import_sentence_transformers():
    global HAS_SENTENCE_TRANSFORMERS
    if HAS_SENTENCE_TRANSFORMERS is not None:
        return HAS_SENTENCE_TRANSFORMERS
    try:
        import sentence_transformers  # noqa: F401
        HAS_SENTENCE_TRANSFORMERS = True
    except Exception:
        HAS_SENTENCE_TRANSFORMERS = False
        logger.warning("sentence-transformers unavailable. Embedder will use fallback bag-of-words mode.")
    return HAS_SENTENCE_TRANSFORMERS


class BiomedicalEmbedder:
    """Manages document chunking and maps biomedical text to dense vector embeddings."""

    def __init__(self, model_name: str = "NLP4Science/pubmedbert-base-embeddings", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = 768
        
        if _try_import_sentence_transformers():
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading sentence transformer model: {}...", model_name)
                self.model = SentenceTransformer(model_name, device=device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            except Exception as exc:
                logger.error("Failed to load sentence transformer model: {}. Falling back to bag-of-words.", exc)
                self.model = None

    def embed_texts(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Embed a list of text strings into dense vector representations.

        Args:
            texts: List of text inputs.
            batch_size: Mini-batch size.
            show_progress: Toggle progress indicator.

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.zeros((0, self.embedding_dim))
            
        if _try_import_sentence_transformers() and self.model is not None:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
            
        # Fallback bag-of-words/hashing vectorizer proxy
        logger.info("Computing fallback bag-of-words document embeddings ({} dim)...", self.embedding_dim)
        vectors = []
        for text in texts:
            # Generate deterministic vector based on word hashes
            words = re.findall(r"\w+", text.lower())
            v = np.zeros(self.embedding_dim)
            if words:
                for w in words:
                    # Hash word to index
                    h = hash(w) % self.embedding_dim
                    v[h] += 1.0
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            vectors.append(v)
            
        return np.array(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate vector embedding for a single text query string."""
        return self.embed_texts([query], show_progress=False)[0]

    def chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 40) -> list[str]:
        """Split a long document into overlapping chunks by word counts.

        Args:
            text: Input string.
            chunk_size: Target words count per chunk.
            overlap: Overlapping words count between adjacent chunks.

        Returns:
            List of chunked strings.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
            
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
            if i + chunk_size >= len(words):
                break
                
        return chunks
