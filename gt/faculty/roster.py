"""Roster loading: authors.csv -> clean, deduplicated, PI-annotated faculty.

Data-quality rules (design flags in plan.md):
- D1: one Scopus ID (56940337900) is shared by two different people. The
  deterministic rule is keep-FIRST by file order; the drop is recorded in
  Roster.dropped_duplicates and surfaced in the README. No heuristic guessing
  which row is "right" — same input, same output, every run.
- D2: missing H5 Index (15% of rows) and H-Index impute to the configured
  constant (0) BEFORE normalization. The scorer never sees NaN and the LLM
  never sees or fills these numbers (req 12).
- D7: Top Journal Quartile % is the only author-level prestige signal. A
  missing value stays None (imputing 0 would claim "bottom-quartile
  journals"); it feeds evidence packs in Stage 4, not the score.
- D8: PI eligibility is a deterministic rank lookup against
  settings.yaml team.pi_eligible_ranks (professorial ranks only) — never an
  LLM decision.
- D11: authors.csv starts with a UTF-8 BOM. Parse with encoding="utf-8-sig"
  or the first column name silently keeps the BOM and lookups break.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gt.config import REPO_ROOT, load_settings

logger = logging.getLogger(__name__)

AUTHORS_CSV = REPO_ROOT / "data" / "authors.csv"


@dataclass
class Faculty:
    """One roster member. Metrics are floats (imputed per D2); raw keeps the
    original CSV row for verification/traceability (Stage 4)."""

    scopus_id: str
    name: str  # canonical "Last, First" display name from the CSV
    department: str = ""
    rank: str = ""
    tenure: str = ""
    h_index: float = 0.0
    h5: float = 0.0
    citations: float = 0.0
    scholarly_output: float = 0.0
    top_journal_quartile: float | None = None  # None = unknown (D7)
    pi_eligible: bool = False  # D8: rank-based, deterministic
    raw: dict = field(default_factory=dict)


@dataclass
class Roster:
    """The clean faculty list plus records of what loading changed."""

    faculty: list[Faculty]
    dropped_duplicates: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.by_id: dict[str, Faculty] = {f.scopus_id: f for f in self.faculty}

    def __len__(self) -> int:
        return len(self.faculty)

    def __iter__(self):
        return iter(self.faculty)

    def __contains__(self, scopus_id: str) -> bool:
        return scopus_id in self.by_id


def _to_float(value, impute: float) -> float:
    """Numeric CSV field -> float, imputing NaN/unparseable values (D2)."""
    num = pd.to_numeric(value, errors="coerce")
    return float(impute) if pd.isna(num) else float(num)


def _clean_str(value) -> str:
    """CSV cell -> stripped string. dtype=str still yields float NaN for
    empty cells, and `NaN or ''` doesn't fall through (NaN is truthy) —
    without this, a missing Tenure would become the literal string 'nan'."""
    return "" if pd.isna(value) else str(value).strip()


def load_roster(
    csv_path: Path | str = AUTHORS_CSV, settings: dict | None = None
) -> Roster:
    """Load and clean the faculty roster.

    Kept uncached deliberately: 501 rows is cheap to re-read, and an
    lru_cache keyed on injectable paths would leak state between tests.
    Streamlit-level caching is a Stage 5 concern.
    """
    cfg = settings or load_settings()
    impute = float(cfg["scoring"]["missing_metric_impute"])
    pi_ranks = set(cfg["team"]["pi_eligible_ranks"])

    # dtype=str keeps Scopus IDs as exact strings (no float mangling) and
    # preserves "Last, First" names verbatim; utf-8-sig strips the BOM (D11).
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)

    required = {"Author", "Scopus ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"authors CSV is missing required columns: {sorted(missing)}")

    faculty: list[Faculty] = []
    dropped: list[dict] = []
    seen: dict[str, Faculty] = {}
    for _, row in df.iterrows():
        scopus_id = str(row["Scopus ID"]).strip()
        if not scopus_id or scopus_id.lower() == "nan":
            logger.warning("skipping roster row with blank Scopus ID: %s", row.get("Author"))
            continue
        name = _clean_str(row.get("Author"))

        if scopus_id in seen:  # D1: shared Scopus ID — keep first, record drop
            dropped.append(
                {
                    "scopus_id": scopus_id,
                    "kept": seen[scopus_id].name,
                    "dropped": name,
                }
            )
            logger.warning(
                "duplicate Scopus ID %s: keeping %r, dropping %r",
                scopus_id, seen[scopus_id].name, name,
            )
            continue

        # D7: Top Journal Quartile % — None when unknown, never zero-filled.
        tq = pd.to_numeric(row.get("Top Journal Quartile %"), errors="coerce")
        top_quartile = None if pd.isna(tq) else float(tq)

        member = Faculty(
            scopus_id=scopus_id,
            name=_clean_str(row.get("Author")),
            department=_clean_str(row.get("Department")),
            rank=_clean_str(row.get("Rank")),
            tenure=_clean_str(row.get("Tenure")),
            h_index=_to_float(row.get("H-Index"), impute),
            h5=_to_float(row.get("H5 Index"), impute),
            citations=_to_float(row.get("Citation Count"), impute),
            scholarly_output=_to_float(row.get("Scholarly Output"), impute),
            top_journal_quartile=top_quartile,
            pi_eligible=_clean_str(row.get("Rank")) in pi_ranks,
            raw={k: v for k, v in row.items()},
        )
        faculty.append(member)
        seen[scopus_id] = member

    return Roster(faculty=faculty, dropped_duplicates=dropped)
