"""CLI: embed publications and (re)build the ChromaDB index.

Usage (from the repo root, in the ads_5984 env):
    python scripts/build_index.py

First run downloads the local embedding model from HuggingFace (~130MB);
embedding ~19.5k chunks takes a few minutes. Later runs re-embed everything
(idempotent upsert) — run this after the CSVs change, not routinely.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as "python scripts/build_index.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gt.config import load_settings  # noqa: E402
from gt.faculty.source import get_faculty_source  # noqa: E402
from gt.retrieval.index import (  # noqa: E402
    PublicationIndex,
    SentenceTransformerEmbedder,
    build_chunks,
)


def main() -> int:
    settings = load_settings()
    model_name = settings["retrieval"]["embedding_model"]

    print(f"Loading publications (data_source: {settings['data_source']})...")
    source = get_faculty_source(settings)
    roster = source.load_roster()
    publications = source.load_publications()
    chunks = build_chunks(publications)
    print(f"roster: {len(roster)} faculty; corpus: {len(chunks)} indexable chunks")

    print(f"Embedding with {model_name} (first run downloads the model)...")
    t0 = time.time()
    embedder = SentenceTransformerEmbedder(model_name)
    index = PublicationIndex()
    count = index.build(chunks, embedder, show_progress_bar=True)
    elapsed = time.time() - t0

    print(f"indexed chunks : {count}")
    print(f"elapsed        : {elapsed:.0f}s")
    print(f"index location : {index._dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
