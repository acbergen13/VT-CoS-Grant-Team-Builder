"""Evidence records: verbatim source text attached to retrieved authors.

Every factual claim the LLM makes in Stage 4 must trace back to one of
these chunks (chain-of-verification, req 18) — so an EvidenceChunk carries
the exact indexed text (`verbatim`) plus the keys needed to check and cite
it: EID for matching, Link for the UI's traceability hyperlink (req 27).
"""

from __future__ import annotations

from dataclasses import dataclass

from gt.retrieval.index import Chunk


@dataclass
class EvidenceChunk:
    """One publication chunk offered as grounding for a candidate."""

    eid: str  # canonical publication key (D9) — verification matches on this
    title: str
    year: int | None
    source_title: str
    doi: str
    link: str  # Scopus URL — shown in the UI next to each claim (req 27)
    doc_type: str
    verbatim: str  # the exact indexed text — the grounding truth
    score: float  # fused RRF score (ranking context for the LLM)
    author_ids: list[str]  # CoS authors on this publication


def build_evidence(chunk_scores: list[tuple[Chunk, float]]) -> list[EvidenceChunk]:
    """(Chunk, rrf) pairs -> EvidenceChunk records, order preserved."""
    return [
        EvidenceChunk(
            eid=chunk.id,
            title=chunk.pub.title,
            year=chunk.pub.year,
            source_title=chunk.pub.source_title,
            doi=chunk.pub.doi,
            link=chunk.pub.link,
            doc_type=chunk.pub.doc_type,
            verbatim=chunk.text,
            score=rrf_score,
            author_ids=list(chunk.pub.cos_author_ids),
        )
        for chunk, rrf_score in chunk_scores
    ]
