from __future__ import annotations

from typing import List

import faiss
import numpy as np

from .models import RetrievalResult, SemanticChunk


class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[SemanticChunk] = []

    def add(self, chunks: List[SemanticChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunk count and embedding count mismatch.")
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[RetrievalResult]:
        if self.index.ntotal == 0:
            return []
        q = np.expand_dims(query_embedding.astype(np.float32), axis=0)
        scores, ids = self.index.search(q, top_k)
        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            results.append(RetrievalResult(chunk=self.chunks[int(idx)], score=float(score)))
        return results
