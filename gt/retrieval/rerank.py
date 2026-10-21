"""Author-level aggregation + MMR diversity selection.

Architecture mitigation 5 (and flag D6): one prolific author with many
similar publications can flood chunk-level results. The fix is three steps —
1. group fused chunks by author (every CoS author on a publication owns it),
2. score each AUTHOR (max cosine over their chunks = "their most relevant
   publication" — an explainable, citable notion), and
3. select authors with MMR so the pool is relevant AND diverse (a team of
   five people doing identical work is not a team).

Determinism (req 19): every sort/tie-break is (-score, scopus_id); MMR picks
via min((-mmr, scopus_id)). Same inputs -> identical selection, every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gt.retrieval.index import Chunk


def cosine_clamped(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity clamped to [0, 1] — the scorer's contract (req 12)
    expects semantic inputs in the unit interval; negative similarities
    carry no useful signal for team building."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


@dataclass
class AuthorCandidate:
    """One author in the candidate pool, with their matched evidence."""

    scopus_id: str
    relevance: float  # max cosine(query, chunk) over their chunks, in [0, 1]
    chunks: list[tuple[Chunk, float]] = field(default_factory=list)  # (chunk, rrf)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(1))


def build_author_candidates(
    fused: list[tuple[Chunk, float]],
    embeddings_by_id: dict[str, np.ndarray],
    query_embedding: np.ndarray,
    author_pool_k: int,
) -> list[AuthorCandidate]:
    """Group fused chunks by author -> the top author_pool_k candidates.

    An author's relevance is the MAX cosine over their chunks (best single
    publication, not an average diluted by their unrelated work). The
    centroid (mean of their matched chunks' embeddings) is MMR's notion of
    "what this person's matched work looks like".
    """
    by_author: dict[str, list[tuple[Chunk, float]]] = {}
    for chunk, rrf_score in fused:  # fused is already rrf-descending
        embedding = embeddings_by_id.get(chunk.id)
        if embedding is None:
            continue  # defensive: skip chunks without a stored embedding
        for author_id in chunk.pub.cos_author_ids:
            by_author.setdefault(author_id, []).append((chunk, rrf_score))

    candidates: list[AuthorCandidate] = []
    for author_id, owned in by_author.items():
        vectors = [embeddings_by_id[c.id] for c, _ in owned]
        relevance = max(cosine_clamped(query_embedding, v) for v in vectors)
        centroid = np.mean(vectors, axis=0)
        candidates.append(
            AuthorCandidate(
                scopus_id=author_id, relevance=relevance, chunks=owned, centroid=centroid
            )
        )
    candidates.sort(key=lambda c: (-c.relevance, c.scopus_id))
    return candidates[:author_pool_k]


def mmr_select(
    candidates: list[AuthorCandidate], select_k: int, mmr_lambda: float
) -> list[AuthorCandidate]:
    """Maximal Marginal Relevance over author centroids.

    Iteratively pick the candidate maximizing
        lambda * relevance - (1 - lambda) * max cosine to already-selected
    (pure relevance for the first pick). High lambda favors relevance, low
    favors diversity. Ties break by scopus_id — deterministic (req 19).
    """
    pool = sorted(candidates, key=lambda c: (-c.relevance, c.scopus_id))
    selected: list[AuthorCandidate] = []
    while pool and len(selected) < select_k:
        def mmr(cand: AuthorCandidate) -> float:
            if not selected:
                return cand.relevance
            penalty = max(cosine_clamped(cand.centroid, s.centroid) for s in selected)
            return mmr_lambda * cand.relevance - (1.0 - mmr_lambda) * penalty

        best = min(pool, key=lambda c: (-mmr(c), c.scopus_id))
        pool.remove(best)
        selected.append(best)
    return selected
