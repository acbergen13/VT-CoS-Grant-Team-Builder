"""Retrieval layer tests — fake embedder + tmp Chroma, no network, no model.

FakeEmbedder maps tokens to deterministic md5-hashed dimensions, so texts
sharing words get similar (dot-product) vectors. The corpus below:

  E1 (author 1) lake water quality modeling     -> climate-ish
  E2 (author 1) climate temperature trends      -> climate-ish
  E3 (author 2) neural network training methods -> ML-ish
  E4 (author 3) machine learning for climate    -> both (the bridge)
  E5 (author 4) quantum computing gates         -> neither
"""

import hashlib
import re

import numpy as np
import pytest

from gt.faculty.publications import Publication
from gt.faculty.roster import Faculty, Roster
from gt.faculty.scoring import RosterScorer  # noqa: F401 (sanity import)
from gt.retrieval.evidence import build_evidence
from gt.retrieval.index import (
    PublicationIndex,
    SentenceTransformerEmbedder,
    build_chunk_text,
    build_chunks,
)
from gt.retrieval.pipeline import RetrievalPipeline, build_pipeline
from gt.retrieval.rerank import (
    AuthorCandidate,
    build_author_candidates,
    cosine_clamped,
    mmr_select,
)
from gt.retrieval.search import HybridSearcher, tokenize

DIM = 32


class FakeEmbedder(SentenceTransformerEmbedder):
    """Deterministic stand-in: token -> md5-hashed dimension, normalized."""

    def __init__(self, dim: int = DIM) -> None:  # deliberately no super().__init__
        self._dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def pub(eid, author_ids, title, abstract=""):
    return Publication(
        eid=eid,
        title=title,
        abstract=abstract,
        year=2024,
        source_title="Journal of Tests",
        doc_type="Article",
        cited_by=0,
        doi=f"10.1/{eid}",
        link=f"https://scopus.com/{eid}",
        author_keywords="",
        index_keywords="",
        cos_author_ids=list(author_ids),
    )


PUBS = [
    pub("E1", ["1"], "Lake water quality modeling", "Climate change drives lake modeling"),
    pub("E2", ["1"], "Climate temperature trends", "Long term climate warming in lakes"),
    pub("E3", ["2"], "Neural network training methods", "Machine learning optimization for networks"),
    pub("E4", ["3"], "Machine learning for climate", "Climate prediction with machine learning models"),
    pub("E5", ["4"], "Quantum computing gates", "Qubit entanglement in quantum computing"),
]


class FakeSource:
    def __init__(self, pubs=PUBS):
        self._pubs = pubs

    def load_roster(self):
        return Roster(
            faculty=[
                Faculty(scopus_id="1", name="Climate Person", h5=10, citations=100),
                Faculty(scopus_id="2", name="ML Person", h5=20, citations=10000),
                Faculty(scopus_id="3", name="Bridge Person", h5=15, citations=5000),
                Faculty(scopus_id="4", name="Quantum Person", h5=25, citations=20000),
            ]
        )

    def load_publications(self):
        return list(self._pubs)


def settings(**overrides):
    retrieval = {
        "embedding_model": "fake",
        "exclude_doc_types": [],
        "candidate_k": 5,
        "author_pool_k": 3,
        "select_k": 2,
        "evidence_per_author": 2,
        "mmr_lambda": 0.7,
    }
    retrieval.update(overrides)
    return {
        "data_source": "csv",
        "scoring": {
            "weights": {"semantic": 0.6, "h5": 0.25, "citations": 0.15},
            "h5_normalization": "minmax",
            "citations_normalization": "log1p_minmax",
            "missing_metric_impute": 0,
        },
        "team": {"pi_eligible_ranks": ["Professor"]},
        "retrieval": retrieval,
    }


def make_pipeline(tmp_path, **overrides):
    return build_pipeline(
        settings=settings(**overrides),
        source=FakeSource(),
        embedder=FakeEmbedder(),
        chroma_path=tmp_path / "chroma",
    )


# -- Chunking -------------------------------------------------------------------


def test_chunk_text_is_verbatim_concatenation():
    p = pub("E1", ["1"], "Title", abstract="Abstract")
    p.author_keywords = "kw1"
    assert build_chunk_text(p) == "Title. Abstract. kw1"
    assert build_chunk_text(pub("E9", ["1"], "Only title")) == "Only title"


def test_build_chunks_skips_textless_publications():
    chunks = build_chunks([pub("E1", ["1"], "Has text"), pub("E2", ["1"], "")])
    assert [c.id for c in chunks] == ["E1"]


# -- Index ------------------------------------------------------------------------


def test_index_build_is_idempotent_and_prunes_stale(tmp_path):
    index = PublicationIndex(tmp_path / "chroma")
    chunks = build_chunks(PUBS)
    assert index.build(chunks, FakeEmbedder()) == 5
    assert index.needs_build(chunks) is False

    shrunk = build_chunks(PUBS[:4])  # E5 leaves the corpus
    assert index.needs_build(shrunk) is True
    assert index.build(shrunk, FakeEmbedder()) == 4
    assert index.needs_build(shrunk) is False
    assert "E5" not in index.indexed_ids()


def test_index_rebuilds_when_eids_match_but_text_changes(tmp_path):
    index = PublicationIndex(tmp_path / "chroma")
    original = build_chunks(PUBS)
    index.build(original, FakeEmbedder())
    assert index.needs_build(original) is False

    changed_pubs = [pub("E1", ["1"], "Completely different title")] + PUBS[1:]
    changed = build_chunks(changed_pubs)
    assert {c.id for c in changed} == {c.id for c in original}
    assert index.needs_build(changed) is True  # CSV -> Elsevier-style change

    index.build(changed, FakeEmbedder())
    assert index.needs_build(changed) is False


def test_index_query_returns_nearest_ids(tmp_path):
    index = PublicationIndex(tmp_path / "chroma")
    index.build(build_chunks(PUBS), FakeEmbedder())
    embedder = FakeEmbedder()
    ids = index.query(embedder.encode_query("climate machine learning"), k=3)
    assert len(ids) == 3
    assert set(ids) <= {c.id for c in build_chunks(PUBS)}


# -- Hybrid search + RRF ------------------------------------------------------------


def test_search_ranks_bridge_chunk_first(tmp_path):
    index = PublicationIndex(tmp_path / "chroma")
    index.build(build_chunks(PUBS), FakeEmbedder())
    searcher = HybridSearcher(index, build_chunks(PUBS))
    embedder = FakeEmbedder()

    fused = searcher.search("machine learning climate", embedder.encode_query("machine learning climate"), k=5)
    assert fused[0][0].id == "E4"  # shares tokens with BOTH query terms
    assert all(score > 0 for _, score in fused)
    # Deterministic order: rerunning yields the identical list (req 19).
    again = searcher.search("machine learning climate", embedder.encode_query("machine learning climate"), k=5)
    assert [(c.id, s) for c, s in fused] == [(c.id, s) for c, s in again]


def test_rrf_rewards_chunks_ranked_by_both_methods():
    # A chunk at vector rank 3 AND bm25 rank 3 beats one at rank 1 in a
    # single list:  2/(60+3) > 1/(60+1).
    both = 1 / (60 + 3) + 1 / (60 + 3)
    single = 1 / (60 + 1)
    assert both > single


def test_search_on_empty_corpus_returns_empty(tmp_path):
    index = PublicationIndex(tmp_path / "chroma")
    searcher = HybridSearcher(index, [])
    assert searcher.search("anything", np.zeros(DIM), k=5) == []


# -- Author aggregation + MMR ---------------------------------------------------------


def make_fused():
    chunks = {c.id: c for c in build_chunks(PUBS)}
    embedder = FakeEmbedder()
    query_emb = embedder.encode_query("machine learning climate")
    # Embeddings for every chunk, as get_embeddings would return.
    embs = {c.id: embedder.encode_query(c.text) for c in chunks.values()}
    fused = [(chunks[eid], 1.0 / (60 + i)) for i, eid in enumerate(["E4", "E1", "E2", "E3", "E5"], start=1)]
    return fused, embs, query_emb


def test_author_grouping_and_max_relevance():
    fused, embs, q = make_fused()
    candidates = build_author_candidates(fused, embs, q, author_pool_k=10)
    by_id = {c.scopus_id: c for c in candidates}
    # Author 1 owns E1 and E2 — grouped, not duplicated.
    assert "1" in by_id["1"].chunks[0][0].pub.cos_author_ids
    assert len(by_id["1"].chunks) == 2
    # Author 3 (bridge, E4) has the highest relevance.
    assert candidates[0].scopus_id == "3"
    assert 0.0 <= candidates[0].relevance <= 1.0
    # Author pool cap respected.
    assert build_author_candidates(fused, embs, q, author_pool_k=2).__len__() == 2


def test_mmr_lambda_extremes():
    fused, embs, q = make_fused()
    candidates = build_author_candidates(fused, embs, q, author_pool_k=10)

    # lambda=1 -> pure relevance: bridge author (3) first.
    pure = mmr_select(candidates, select_k=2, mmr_lambda=1.0)
    assert pure[0].scopus_id == "3"

    # lambda=0 with equal relevance -> first pick by tie-break, then the
    # LEAST similar centroid to it (diversity dominates).
    equal = [
        AuthorCandidate(scopus_id="a", relevance=0.5, centroid=np.array([1.0, 0.0])),
        AuthorCandidate(scopus_id="b", relevance=0.5, centroid=np.array([0.0, 1.0])),
        AuthorCandidate(scopus_id="c", relevance=0.5, centroid=np.array([0.99, 0.05])),
    ]
    diverse = mmr_select(equal, select_k=2, mmr_lambda=0.0)
    assert [c.scopus_id for c in diverse] == ["a", "b"]  # c ~= a, so b wins


def test_cosine_clamped_guards():
    assert cosine_clamped(np.zeros(3), np.ones(3)) == 0.0
    assert cosine_clamped(np.array([1, 0]), np.array([1, 0])) == pytest.approx(1.0)
    assert cosine_clamped(np.array([1, 0]), np.array([-1, 0])) == 0.0  # clamped


# -- Evidence -------------------------------------------------------------------------


def test_evidence_carries_verbatim_text_and_keys():
    chunks = build_chunks(PUBS)
    evidence = build_evidence([(chunks[3], 0.031)])
    assert evidence[0].eid == "E4"
    assert evidence[0].verbatim == chunks[3].text
    assert evidence[0].link == "https://scopus.com/E4"
    assert evidence[0].author_ids == ["3"]


# -- Pipeline end-to-end -----------------------------------------------------------------


def test_pipeline_end_to_end_selects_relevant_diverse_authors(tmp_path):
    pipeline = make_pipeline(tmp_path)
    result = pipeline.retrieve("machine learning climate")

    assert result.query == "machine learning climate"
    selected_ids = [c.faculty.scopus_id for c in result.ranked]
    assert len(selected_ids) == 2  # select_k
    assert "4" not in selected_ids  # quantum author is irrelevant -> not in pool
    assert "3" in selected_ids  # bridge author retrieved

    # Order comes from the deterministic scorer, not retrieval order:
    # author 3 has the best semantic score (bridge chunk E4).
    assert selected_ids[0] == "3"

    # Every ranked member carries score components (req 27) and evidence.
    for candidate in result.ranked:
        assert set(candidate.components) == {"semantic", "h5", "citations"}
        evidence = result.evidence_by_id[candidate.faculty.scopus_id]
        assert 1 <= len(evidence) <= 2  # evidence_per_author cap
        for chunk in evidence:
            assert chunk.verbatim  # non-empty grounding text
            assert chunk.eid  # verification key present

    # Author 1 appears at most once despite owning two matched chunks (D6).
    pool_sizes = [len(result.evidence_by_id[i]) for i in selected_ids]
    assert all(size <= 2 for size in pool_sizes)


def test_pipeline_is_deterministic(tmp_path):
    pipeline = make_pipeline(tmp_path)
    first = pipeline.retrieve("machine learning climate")
    second = pipeline.retrieve("machine learning climate")
    assert [(c.faculty.scopus_id, c.score) for c in first.ranked] == [
        (c.faculty.scopus_id, c.score) for c in second.ranked
    ]
    assert first.evidence_by_id == second.evidence_by_id


def test_pipeline_empty_query_returns_empty(tmp_path):
    result = make_pipeline(tmp_path).retrieve("   ")
    assert result.ranked == []
    assert result.evidence_by_id == {}


def test_pipeline_empty_corpus_returns_empty(tmp_path):
    pipeline = build_pipeline(
        settings=settings(),
        source=FakeSource(pubs=[]),
        embedder=FakeEmbedder(),
        chroma_path=tmp_path / "chroma",
    )
    assert pipeline.retrieve("climate").ranked == []


def test_factory_auto_builds_on_manifest_mismatch(tmp_path):
    # Build once with 5 pubs...
    make_pipeline(tmp_path)
    index = PublicationIndex(tmp_path / "chroma")
    assert index.count() == 5
    # ...then open a pipeline against a SHRUNKER corpus: the manifest
    # mismatch must trigger an automatic rebuild (no stale E5 chunks).
    build_pipeline(
        settings=settings(),
        source=FakeSource(pubs=PUBS[:4]),
        embedder=FakeEmbedder(),
        chroma_path=tmp_path / "chroma",
    )
    assert PublicationIndex(tmp_path / "chroma").count() == 4
