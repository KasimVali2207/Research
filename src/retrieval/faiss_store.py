# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FAISS Vector Store and document retriever.

Provides local document vector indexing. Falls back to pure-numpy cosine similarity
if the compiled faiss-cpu package is missing.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from loguru import logger

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss is not installed. Vector store will run in fallback NumPy mode.")


class FAISSVectorStore:
    """Indexes and retrieves documents using dense vector representations."""

    def __init__(self, embedding_dim: int = 768, index_type: str = "flat") -> None:
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # In-memory document storage
        self.documents: list[dict] = []
        
        # FAISS index
        self.index = None
        self.numpy_embeddings = []  # Fallback store
        
        if HAS_FAISS:
            if index_type == "flat":
                # L2 distance index (L2 distance on normalized vectors = cosine distance)
                self.index = faiss.IndexFlatL2(embedding_dim)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatL2(embedding_dim)
                # 100 clusters
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100, faiss.METRIC_L2)
                # Training needed for IVF index, we'll initialize or fallback to Flat
                self.index.set_direct_map_type(faiss.DirectMap.Hashtable)
        else:
            self.index = None

    def add_documents(self, documents: list[dict], embeddings: np.ndarray) -> None:
        """Add documents and their pre-computed dense embeddings to the index.

        Args:
            documents: List of dictionary records containing document fields.
            embeddings: Float numpy array of shape (len(documents), embedding_dim)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")
            
        if len(documents) == 0:
            return
            
        # Ensure float32 format
        embeddings_f32 = embeddings.astype(np.float32)
        
        # Add to list
        self.documents.extend(documents)
        
        if HAS_FAISS and self.index is not None:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_f32)
            
            # If IVF index, we need to train first
            if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
                logger.info("Training FAISS IVF index on {} vectors...", len(embeddings_f32))
                self.index.train(embeddings_f32)
                
            self.index.add(embeddings_f32)
        else:
            # Fallback: store raw normalized embeddings
            norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normed = embeddings_f32 / norms
            if len(self.numpy_embeddings) == 0:
                self.numpy_embeddings = normed
            else:
                self.numpy_embeddings = np.vstack([self.numpy_embeddings, normed])
                
        logger.info("Added {} documents to the vector store.", len(documents))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Query vector database and return top K matching documents.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of matching document dicts with score, pmid, and title.
        """
        if not self.documents:
            return []
            
        query_np = query_embedding.astype(np.float32).reshape(1, -1)
        
        if HAS_FAISS and self.index is not None:
            faiss.normalize_L2(query_np)
            # Search L2 distance
            distances, indices = self.index.search(query_np, min(top_k, len(self.documents)))
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.documents):
                    continue
                doc = self.documents[idx].copy()
                # Cosine similarity = 1 - (L2_distance^2 / 2)
                doc["score"] = float(1.0 - (dist / 2.0))
                doc["rank"] = i + 1
                results.append(doc)
            return results
            
        else:
            # Fallback: numpy cosine similarity
            norm = np.linalg.norm(query_np)
            norm = 1.0 if norm == 0 else norm
            q_normed = query_np / norm
            
            # dot product of normalized vectors
            scores = np.dot(self.numpy_embeddings, q_normed.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                doc = self.documents[idx].copy()
                doc["score"] = float(scores[idx])
                doc["rank"] = i + 1
                results.append(doc)
            return results

    def save(self, path: str) -> None:
        """Serialize FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save metadata and numpy arrays
        meta = {
            "documents": self.documents,
            "numpy_embeddings": self.numpy_embeddings,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type
        }
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump(meta, f)
            
        # Save FAISS index
        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, f"{path}.index")
            
        logger.info("Saved vector store to {}", path)

    @classmethod
    def load(cls, path: str) -> FAISSVectorStore:
        """Load serialized FAISS index and metadata from disk."""
        with open(f"{path}.meta.pkl", "rb") as f:
            meta = pickle.load(f)
            
        store = cls(embedding_dim=meta["embedding_dim"], index_type=meta["index_type"])
        store.documents = meta["documents"]
        store.numpy_embeddings = meta["numpy_embeddings"]
        
        if HAS_FAISS and os.path.exists(f"{path}.index"):
            store.index = faiss.read_index(f"{path}.index")
            
        logger.info("Loaded vector store from {}", path)
        return store
