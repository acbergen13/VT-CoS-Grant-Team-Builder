"""ChromaDB vector index over publication chunks + the local embedding model.

Chunking (design flag D5): ONE chunk per publication — title + abstract +
author/index keywords. That concatenation is verbatim source text: it is what
gets embedded AND what Stage 4's chain-of-verification quotes against
(req 18), so evidence and retrieval always agree.

Embeddings (flag D10): local sentence-transformers model from
config retrieval.embedding_model (default BAAI/bge-small-en-v1.5 — 512-token
context, 384-dim; the median abstract is 156 words so nothing meaningful is
truncated). Local keeps the app runnable without extra endpoint setup; the
ARC endpoint's Qwen3-Embedding-4B is the documented fallback. First build
downloads the model from HuggingFace once (~130MB).

Storage: persistent ChromaDB under data/chroma/ (gitignored). Embedding
~19.5k chunks is the expensive step (one-time, minutes); queries are
milliseconds. Rebuilds are idempotent: chunk ids are EIDs (deterministic,
D9) so upserts overwrite in place, and a manifest.json tracks indexed ids so
chunks whose publications left the corpus are pruned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from gt.config import REPO_ROOT
from gt.faculty.publications import Publication

CHROMA_DIR = REPO_ROOT / "data" / "chroma"
COLLECTION_NAME = "publications"
MANIFEST_NAME = "manifest.json"
UPSERT_BATCH = 512

# BGE retrieval models are trained with this instruction prefixed to queries
# (not to indexed passages) — skipping it measurably hurts recall.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Chunk:
    """One indexed publication. id is the EID (canonical key, D9)."""

    id: str
    text: str  # verbatim source text (embedding input AND evidence)
    pub: Publication


def build_chunk_text(pub: Publication) -> str:
    """Title + abstract + keywords, skipping absent parts (934 CoS-linked
    rows have no abstract — title+keywords still match queries)."""
    parts = [pub.title, pub.abstract, pub.author_keywords, pub.index_keywords]
    return ". ".join(p for p in parts if p).strip()


def build_chunks(publications: list[Publication]) -> list[Chunk]:
    """Publication list -> indexable chunks. Publications with no text at all
    are skipped (nothing to embed, nothing to ground a claim in)."""
    chunks: list[Chunk] = []
    for pub in publications:
        text = build_chunk_text(pub)
        if text:
            chunks.append(Chunk(id=pub.eid, text=text, pub=pub))
    return chunks


class SentenceTransformerEmbedder:
    """Lazy local embedder. Inject a fake in tests — no network, no 130MB.

    All embeddings are L2-normalized, so cosine similarity is a dot product.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # loaded on first use (keeps import/boot cheap)

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def encode(
        self, texts: list[str], batch_size: int = 64, show_progress_bar: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        prefixed = (
            BGE_QUERY_PREFIX + text if "bge" in self._model_name.lower() else text
        )
        return self.encode([prefixed])[0]


class PublicationIndex:
    """Persistent ChromaDB wrapper over the chunk corpus.

    The chunk registry (Chunk objects) lives in memory next to this index —
    Chroma answers "which ids are nearest", the registry supplies the
    verbatim text and publication fields.
    """

    def __init__(self, persist_dir: Path | str | None = None) -> None:
        self._dir = Path(persist_dir) if persist_dir else CHROMA_DIR
        self._manifest_path = self._dir / MANIFEST_NAME
        # hnsw:space=cosine — distances are 1 - cosine similarity.
        self._client = chromadb.PersistentClient(path=str(self._dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    # -- Build ---------------------------------------------------------------

    def indexed_ids(self) -> set[str]:
        """Ids recorded in the manifest (empty set if never built)."""
        if not self._manifest_path.exists():
            return set()
        with open(self._manifest_path, encoding="utf-8") as f:
            return set(json.load(f))

    def build(
        self,
        chunks: list[Chunk],
        embedder: SentenceTransformerEmbedder,
        show_progress_bar: bool = False,
    ) -> int:
        """Embed every chunk and upsert (idempotent); prune stale ids.

        Embedding is recomputed for all chunks on every build — that is the
        cost of correctness (upserted embeddings always match current text).
        Run scripts/build_index.py deliberately; the pipeline auto-builds
        only when the manifest disagrees with the corpus.
        """
        embeddings = embedder.encode(
            [c.text for c in chunks], show_progress_bar=show_progress_bar
        )
        new_ids = {c.id for c in chunks}
        for start in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[start : start + UPSERT_BATCH]
            self._collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embeddings[start : start + UPSERT_BATCH].tolist(),
                documents=[c.text for c in batch],
            )
        stale = self.indexed_ids() - new_ids
        if stale:
            self._collection.delete(ids=sorted(stale))
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(sorted(new_ids), f)
        return self._collection.count()

    def needs_build(self, chunks: list[Chunk]) -> bool:
        """True when the index is empty or out of sync with the corpus."""
        return self.indexed_ids() != {c.id for c in chunks}

    # -- Query ---------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def query(self, embedding: np.ndarray, k: int) -> list[str]:
        """Top-k nearest chunk ids for one query embedding."""
        n = min(k, self.count())
        if n <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding.tolist()], n_results=n
        )
        return list(result["ids"][0])

    def get_embeddings(self, ids: list[str]) -> dict[str, np.ndarray]:
        """Fetch stored embeddings by id (for centroid + cosine computation
        on chunks that BM25, not the vector search, surfaced)."""
        if not ids:
            return {}
        result = self._collection.get(ids=ids, include=["embeddings"])
        vectors = np.asarray(result["embeddings"], dtype=np.float32)
        return dict(zip(result["ids"], vectors))
