"""Retrieval pipeline: query -> hybrid search -> author MMR -> deterministic
scoring -> evidence pack. This is the fixed workflow the architecture doc
describes — every step runs in code order; the LLM (Stage 4) never chooses
tools or weights.

Division of labor:
- MMR decides POOL MEMBERSHIP (relevance + author diversity, mitigation 5);
- the deterministic RosterScorer decides ORDER (req 12: 0.6 semantic +
  0.25 H5 + 0.15 citations — weights from config, never the LLM);
- EvidenceChunks ride along for grounding (req 18) and citation (req 27).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gt.config import load_settings
from gt.faculty.roster import Roster
from gt.faculty.scoring import RosterScorer, ScoredCandidate
from gt.faculty.source import FacultySource, get_faculty_source
from gt.retrieval.evidence import EvidenceChunk, build_evidence
from gt.retrieval.index import (
    CHROMA_DIR,
    Chunk,
    PublicationIndex,
    SentenceTransformerEmbedder,
    build_chunks,
    corpus_fingerprint,
)
from gt.retrieval.rerank import build_author_candidates, mmr_select
from gt.retrieval.search import HybridSearcher


@dataclass
class RetrievalResult:
    """Everything Stage 4 needs to propose and verify a team."""

    query: str
    ranked: list[ScoredCandidate] = field(default_factory=list)  # score order
    evidence_by_id: dict[str, list[EvidenceChunk]] = field(default_factory=dict)
    semantic_by_id: dict[str, float] = field(default_factory=dict)


class RetrievalPipeline:
    def __init__(
        self,
        roster: Roster,
        chunks: list[Chunk],
        index: PublicationIndex,
        searcher: HybridSearcher,
        embedder: SentenceTransformerEmbedder,
        settings: dict,
    ) -> None:
        self._roster = roster
        self._chunks = chunks
        self._index = index
        self._searcher = searcher
        self._embedder = embedder
        self._settings = settings
        retrieval = settings["retrieval"]
        self._candidate_k = retrieval["candidate_k"]
        self._author_pool_k = retrieval["author_pool_k"]
        self._select_k = retrieval["select_k"]
        self._evidence_per_author = retrieval["evidence_per_author"]
        self._mmr_lambda = retrieval["mmr_lambda"]
        self._scorer = RosterScorer(roster, settings)  # fitted once

    # Read-only access for the LLM layer (Stage 4): the PI-widening fallback
    # re-uses this exact scorer — a second instance would re-fit
    # normalization bounds and could diverge (req 19).
    @property
    def roster(self) -> Roster:
        return self._roster

    @property
    def scorer(self) -> RosterScorer:
        return self._scorer

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def corpus_hash(self) -> str:
        """Content fingerprint used in LLM cache keys (Stage 6)."""
        return corpus_fingerprint(self._chunks)

    def retrieve(self, query: str) -> RetrievalResult:
        """One retrieval pass. Deterministic: same corpus + same query ->
        identical ranked ids, scores, and evidence (req 19)."""
        query = (query or "").strip()
        if not query:
            return RetrievalResult(query=query)

        # 1) Hybrid recall over chunks (vector + BM25 -> RRF).
        query_embedding = self._embedder.encode_query(query)
        fused = self._searcher.search(query, query_embedding, self._candidate_k)
        if not fused:
            return RetrievalResult(query=query)

        # 2) Group by author, MMR-select the diverse candidate pool.
        embeddings = self._index.get_embeddings([c.id for c, _ in fused])
        candidates = build_author_candidates(
            fused, embeddings, query_embedding, self._author_pool_k
        )
        selected = mmr_select(candidates, self._select_k, self._mmr_lambda)
        if not selected:
            return RetrievalResult(query=query)

        # 3) Deterministic ranking over the selected pool (req 12).
        semantic_by_id = {c.scopus_id: c.relevance for c in selected}
        scored = self._scorer.score(semantic_by_id)
        selected_ids = set(semantic_by_id)
        ranked = [c for c in scored if c.faculty.scopus_id in selected_ids][
            : self._select_k
        ]

        # 4) Evidence pack: each author's top chunks, verbatim (reqs 18/27).
        evidence_by_id = {
            c.scopus_id: build_evidence(c.chunks[: self._evidence_per_author])
            for c in selected
        }
        return RetrievalResult(
            query=query,
            ranked=ranked,
            evidence_by_id=evidence_by_id,
            semantic_by_id=semantic_by_id,
        )


def build_pipeline(
    settings: dict | None = None,
    source: FacultySource | None = None,
    embedder: SentenceTransformerEmbedder | None = None,
    chroma_path=None,
) -> RetrievalPipeline:
    """Factory: wire roster + corpus + index + searcher.

    Auto-builds the (expensive) embedding index only when it is empty or out
    of sync with the corpus (manifest mismatch) — deliberate rebuilds belong
    to scripts/build_index.py, never to a chat turn.
    """
    cfg = settings or load_settings()
    source = source or get_faculty_source(cfg)
    embedder = embedder or SentenceTransformerEmbedder(
        cfg["retrieval"]["embedding_model"]
    )

    roster = source.load_roster()
    chunks = build_chunks(source.load_publications())

    index = PublicationIndex(chroma_path or CHROMA_DIR)
    if index.needs_build(chunks):
        index.build(chunks, embedder)
    searcher = HybridSearcher(index, chunks)
    return RetrievalPipeline(roster, chunks, index, searcher, embedder, cfg)
