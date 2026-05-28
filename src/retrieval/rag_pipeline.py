# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified RAG (Retrieval-Augmented Generation) pipeline coordinating search and embeddings.
"""

from __future__ import annotations

import json
import os
from loguru import logger

from src.retrieval.embedder import BiomedicalEmbedder
from src.retrieval.faiss_store import FAISSVectorStore


class RAGPipeline:
    """Orchestrates document embedding, index creation, and runtime query retrieval."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.index_path = cfg.get("retrieval", {}).get("faiss_index_path", "data/processed/faiss_index")
        
        # Load embedding model
        model_name = cfg.get("retrieval", {}).get("embedding_model", "NLP4Science/pubmedbert-base-embeddings")
        self.embedder = BiomedicalEmbedder(model_name=model_name)
        
        # Initialize vector store
        self.store = None
        if os.path.exists(f"{self.index_path}.meta.pkl"):
            try:
                self.store = FAISSVectorStore.load(self.index_path)
            except Exception as exc:
                logger.error("Failed to load vector store from disk: {}. Re-initializing.", exc)
                
        if self.store is None:
            self.store = FAISSVectorStore(embedding_dim=self.embedder.embedding_dim)

    def build_index(self, kb_path: str) -> None:
        """Parse, chunk, embed, and index a raw JSONL literature knowledgebase.

        Args:
            kb_path: Path to pubmed_kb.jsonl file.
        """
        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"Knowledgebase file not found at {kb_path}")
            
        logger.info("Building vector index from literature database: {}...", kb_path)
        
        documents = []
        texts = []
        
        chunk_size = self.cfg.get("retrieval", {}).get("chunk_size", 256)
        chunk_overlap = self.cfg.get("retrieval", {}).get("chunk_overlap", 40)
        
        with open(kb_path, "r") as f:
            for line in f:
                paper = json.loads(line)
                text = paper.get("text", "")
                
                # Chunk document
                chunks = self.embedder.chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
                
                for idx, chunk in enumerate(chunks):
                    # Add index tag
                    doc_meta = {
                        "pmid": paper.get("pmid", "N/A"),
                        "title": paper.get("title", "Unknown"),
                        "journal": paper.get("journal", "Unknown"),
                        "year": paper.get("year", "N/A"),
                        "chunk_idx": idx
                    }
                    
                    documents.append({
                        "text": chunk,
                        "metadata": doc_meta
                    })
                    texts.append(chunk)
                    
        if not texts:
            logger.warning("No texts found to index in knowledgebase.")
            return
            
        # Compute embeddings
        logger.info("Computing dense embeddings for {} chunks...", len(texts))
        embeddings = self.embedder.embed_texts(texts)
        
        # Add to FAISS index
        self.store.add_documents(documents, embeddings)
        
        # Save index to disk
        self.store.save(self.index_path)
        logger.info("RAG vector index built and saved to {}", self.index_path)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Convert query string to embedding and search the index.

        Args:
            query: Clinical query string.
            top_k: Number of abstracts to retrieve.

        Returns:
            List of matching documents.
        """
        if top_k is None:
            top_k = self.cfg.get("retrieval", {}).get("top_k", 5)
            
        query_emb = self.embedder.embed_query(query)
        results = self.store.search(query_emb, top_k=top_k)
        
        # Format results for the grounding agent
        formatted = []
        for r in results:
            formatted.append({
                "text": r["text"],
                "metadata": r["metadata"],
                "score": r["score"],
                "rank": r["rank"]
            })
            
        return formatted
