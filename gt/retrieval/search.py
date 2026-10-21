"""Hybrid retrieval: vector search (ChromaDB) + BM25, fused via Reciprocal
Rank Fusion (architecture mitigation 3: wide recall, then narrow down).

Why hybrid: pure vector search misses exact terminology (acronyms, program
names, species); pure BM25 misses paraphrase. Running both over the same
corpus and fusing with RRF — score(chunk) = sum over lists of
1/(RRF_K + rank) — needs no calibration between incomparable score scales
and rewards chunks BOTH methods rank highly. All tie-breaks are by chunk id:
same corpus + same query -> identical fused order, every run (req 19).
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from gt.retrieval.index import Chunk, PublicationIndex

RRF_K = 60  # standard RRF constant: dampens the difference between rank 1 and 2


def tokenize(text: str) -> list[str]:
    """Simple lexical tokenizer shared by BM25 indexing and queries."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridSearcher:
    """Vector + BM25 over the same chunk corpus. The embedder stays OUTSIDE
    (the pipeline computes the query embedding once and passes it in), so
    this class is trivially testable with any vectors."""

    def __init__(self, index: PublicationIndex, chunks: list[Chunk]) -> None:
        self._index = index
        self._chunks = list(chunks)
        self._by_id = {c.id: c for c in self._chunks}
        tokenized = [tokenize(c.text) for c in self._chunks]
        # BM25Okapi raises on an empty corpus — fall back to vector-only.
        self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None

    def search(
        self, query_text: str, query_embedding: np.ndarray, k: int
    ) -> list[tuple[Chunk, float]]:
        """Top-k chunks by RRF fusion of vector and BM25 rankings."""
        if not self._chunks or k <= 0:
            return []

        # Vector ranking: nearest chunk ids from ChromaDB (order = rank).
        vector_ids = self._index.query(query_embedding, k)
        vector_rank = {cid: rank for rank, cid in enumerate(vector_ids, start=1)}

        # BM25 ranking: corpus positions of the top-scoring chunks.
        bm25_rank: dict[str, int] = {}
        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokenize(query_text))
            # Only chunks with a positive score are "retrieved" by BM25;
            # ties break by chunk id so the ranking is deterministic.
            matching = [
                i for i in np.argsort(-scores) if scores[i] > 0
            ]
            for rank, i in enumerate(matching[:k], start=1):
                bm25_rank[self._chunks[i].id] = rank

        # Reciprocal Rank Fusion over the union of both lists.
        fused: list[tuple[Chunk, float]] = []
        for cid in set(vector_rank) | set(bm25_rank):
            if cid not in self._by_id:
                continue  # defensive: BM25 corpus and index share ids by construction
            score = 0.0
            if cid in vector_rank:
                score += 1.0 / (RRF_K + vector_rank[cid])
            if cid in bm25_rank:
                score += 1.0 / (RRF_K + bm25_rank[cid])
            fused.append((self._by_id[cid], score))
        fused.sort(key=lambda item: (-item[1], item[0].id))
        return fused[:k]
