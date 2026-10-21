"""Deterministic faculty ranking (req 12) — the LLM never touches this.

    score = 0.6 x semantic + 0.25 x norm(H5) + 0.15 x norm(citations)

- Weights come from config/settings.yaml scoring.weights (pinned to
  0.6/0.25/0.15 by tests) and must never be set or changed by the LLM.
- semantic: similarity in [0, 1] from the retrieval layer (Stage 3), keyed by
  Scopus ID; missing authors default to 0.0; values are clamped defensively.
- norm(H5): plain min-max over the roster (H5 is bounded, 1-26 in this data).
- norm(citations): log1p then min-max — Citation Count is extremely
  right-skewed (max ~49.6k vs median 526), so raw min-max would map almost
  everyone to ~0 (design flag D3).

Determinism (req 19): normalizers are fitted once on the roster's imputed
values, and ranking is a total order (-score, scopus_id). Same roster + same
semantic scores -> bit-identical ranking, every run.

Explainability (req 27): every ScoredCandidate carries its component
breakdown so the UI can show WHY a faculty member ranked where they did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from gt.config import load_settings
from gt.faculty.roster import Faculty, Roster


@dataclass
class ScoredCandidate:
    """One ranked faculty member with the deterministic score breakdown."""

    faculty: Faculty
    score: float
    components: dict  # {"semantic": x, "h5": y, "citations": z} in [0, 1]


class MetricNormalizer:
    """Fitted min-max normalizer; "log1p_minmax" log-transforms first (D3).

    Fitting happens exactly once per roster (in RosterScorer.__init__), on
    the imputed metric values, so bounds are stable across runs.
    """

    def __init__(self, values: Sequence[float], mode: str) -> None:
        if mode not in ("minmax", "log1p_minmax"):
            raise ValueError(f"unknown normalization mode {mode!r}")
        self._mode = mode
        transformed = [self._transform(v) for v in values] if len(values) else [0.0]
        self._lo = min(transformed)
        self._hi = max(transformed)

    def _transform(self, value: float) -> float:
        return math.log1p(value) if self._mode == "log1p_minmax" else float(value)

    def __call__(self, value: float) -> float:
        v = self._transform(value)
        if self._hi <= self._lo:
            return 0.0  # degenerate roster (single member / constant metric)
        normalized = (v - self._lo) / (self._hi - self._lo)
        return max(0.0, min(1.0, normalized))  # clamp stragglers into [0, 1]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class RosterScorer:
    """Scores the whole roster against one set of semantic similarities."""

    def __init__(self, roster: Roster, settings: dict | None = None) -> None:
        cfg = (settings or load_settings())["scoring"]
        self._roster = roster
        self.weights: dict = cfg["weights"]  # {semantic, h5, citations}
        # Fit normalization bounds ONCE, on imputed roster values (D2/D3).
        self._h5_norm = MetricNormalizer([f.h5 for f in roster], cfg["h5_normalization"])
        self._citations_norm = MetricNormalizer(
            [f.citations for f in roster], cfg["citations_normalization"]
        )

    def score(self, semantic_by_id: Mapping[str, float]) -> list[ScoredCandidate]:
        """Rank every roster member. Returns a NEW list sorted by
        (-score, scopus_id) — a total order, so reruns are bit-identical."""
        w_sem = self.weights["semantic"]
        w_h5 = self.weights["h5"]
        w_cit = self.weights["citations"]

        candidates: list[ScoredCandidate] = []
        for member in self._roster:
            semantic = _clamp01(semantic_by_id.get(member.scopus_id, 0.0))
            h5 = self._h5_norm(member.h5)
            citations = self._citations_norm(member.citations)
            total = w_sem * semantic + w_h5 * h5 + w_cit * citations
            candidates.append(
                ScoredCandidate(
                    faculty=member,
                    score=total,
                    components={"semantic": semantic, "h5": h5, "citations": citations},
                )
            )
        candidates.sort(key=lambda c: (-c.score, c.faculty.scopus_id))
        return candidates
