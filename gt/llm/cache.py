"""Disk cache for LLM build plans and finished builds (reqs 19, 21).

Why: the ARC endpoint's thinking models are not sampling-reproducible, and
the optimized query shapes the whole candidate pool. Caching the plan per
solicitation pins the query + roles so every re-run of the same solicitation
works from identical inputs, and skips the LLM calls that produced them.

Keys are content-hashed (opportunity id + model + solicitation title/synopsis
excerpt), so a materially different solicitation or model invalidates the
entry naturally. Files live under data/llm_cache/ (gitignored, ships in the
zip like the other data artifacts). JSON only — no pickle, nothing executable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gt.config import REPO_ROOT

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "llm_cache"

# Bumped whenever the serialized plan/build format changes meaning (v2:
# canonical-query rework; v3: deterministic base_roles structure; v4:
# corpus/source fingerprint in the key — Oct 22 Stage 6): old-format entries
# must never be read back under an unchanged key.
CACHE_FORMAT = "v4"


def plan_cache_key(
    opp_id: str,
    model: str,
    title: str,
    synopsis: str,
    extra: str = "",
    corpus: str = "",
) -> str:
    """Stable key for one solicitation's build plan under one model.
    `extra` covers user build instructions that shape the role plan;
    `corpus` is the data-source/corpus fingerprint so CSV-built plans are
    never served after switching to Elsevier (or after a corpus rebuild)."""
    content = (
        f"{CACHE_FORMAT}|{opp_id}|{model}|{title}|{(synopsis or '')[:500]}"
        f"|{extra.strip()[:200]}|{corpus}"
    )
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:24]


def build_cache_key(
    opp_id: str,
    model: str,
    revision: int,
    anchor: str,
    extra: str = "",
    corpus: str = "",
) -> str:
    """Stable key for one finished build (a revision of a team). `anchor` is
    the query or critique excerpt the build was produced from; `corpus`
    pins the data source/corpus content."""
    content = (
        f"{CACHE_FORMAT}|{opp_id}|{model}|{revision}|{anchor[:500]}"
        f"|{extra.strip()[:200]}|{corpus}"
    )
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:24]


class BuildCache:
    """Tiny JSON file store: one file per key under cache_dir."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self._dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    def get(self, key: str) -> dict | None:
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # corrupt entry -> treat as a miss

    def put(self, key: str, payload: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
