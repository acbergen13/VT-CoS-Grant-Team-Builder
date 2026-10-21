"""Publications loading: publications.csv -> CoS-linked publication records.

Join + filtering rules (design flags in plan.md):
- D4: Author(s) ID is a ";"-separated Scopus ID list. We split, strip, and
  INTERSECT with the roster — the ~50k external co-author IDs and the 1,425
  rows with no CoS author never enter the candidate pool or author grouping.
- D5: noise document types (Erratum, Editorial, Note, Letter — e.g. the
  "This corrects the article DOI: ..." placeholder abstracts) are excluded
  before anything indexes them. The list lives in settings.yaml.
- D9: EID is the canonical publication key (0 duplicates in the real data);
  Link is kept verbatim for the claim-traceability UI (reqs 18/27).
- D11: UTF-8 BOM -> parse with encoding="utf-8-sig".

raw rows are NOT retained (20k rows x full dicts would balloon memory);
every field the verification stage needs is extracted onto the dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gt.config import REPO_ROOT, load_settings
from gt.faculty.roster import Roster

logger = logging.getLogger(__name__)

PUBLICATIONS_CSV = REPO_ROOT / "data" / "publications.csv"


@dataclass
class Publication:
    """One CoS-linked publication (the unit of retrieval chunks in Stage 3)."""

    eid: str  # canonical Scopus key (D9)
    title: str
    year: int | None
    abstract: str
    source_title: str
    doc_type: str
    cited_by: int
    doi: str
    link: str  # Scopus URL — claim traceability in the UI (req 18/27)
    author_keywords: str
    index_keywords: str
    cos_author_ids: list[str] = field(default_factory=list)  # roster-filtered (D4)


def parse_author_ids(raw: str) -> set[str]:
    """'id1; id2; id3' -> {id1, id2, id3}. Tolerates blank/NaN input."""
    if not isinstance(raw, str):
        return set()
    return {part.strip() for part in raw.split(";") if part.strip()}


def _clean_str(value) -> str:
    """CSV cell -> stripped string. With dtype=str, pandas still yields float
    NaN for empty cells — and `NaN or ''` does NOT fall through (NaN is
    truthy), so a plain str() would produce the literal string 'nan'."""
    return "" if pd.isna(value) else str(value).strip()


def load_publications(
    roster: Roster,
    csv_path: Path | str = PUBLICATIONS_CSV,
    settings: dict | None = None,
) -> list[Publication]:
    """Load publications, keeping only rows with >=1 roster author (D4) and a
    document type worth indexing (D5). Input order is preserved."""
    cfg = settings or load_settings()
    excluded = set(cfg["retrieval"]["exclude_doc_types"])

    # dtype=str: IDs/DOIs/links stay verbatim strings; utf-8-sig for the BOM.
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)

    required = {"Author(s) ID", "Title", "EID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"publications CSV is missing required columns: {sorted(missing)}")

    publications: list[Publication] = []
    seen_eids: set[str] = set()
    for _, row in df.iterrows():
        eid = str(row.get("EID") or "").strip()
        if not eid or eid.lower() == "nan":
            continue  # no canonical key -> cannot be verified later (D9)
        if eid in seen_eids:  # defensive: real data has 0 duplicate EIDs
            logger.warning("duplicate EID %s skipped (keeping first)", eid)
            continue

        cos_author_ids = parse_author_ids(row.get("Author(s) ID")) & set(roster.by_id)
        if not cos_author_ids:
            continue  # D4: no CoS author -> not a candidate-team data point

        doc_type = str(row.get("Document Type") or "").strip()
        if doc_type in excluded:
            continue  # D5: errata/editorials/notes/letters are noise

        year = pd.to_numeric(row.get("Year"), errors="coerce")
        cited_by = pd.to_numeric(row.get("Cited by"), errors="coerce")
        publications.append(
            Publication(
                eid=eid,
                title=_clean_str(row.get("Title")),
                year=None if pd.isna(year) else int(year),
                abstract=_clean_str(row.get("Abstract")),
                source_title=_clean_str(row.get("Source title")),
                doc_type=doc_type,
                cited_by=0 if pd.isna(cited_by) else int(cited_by),
                doi=_clean_str(row.get("DOI")),
                link=_clean_str(row.get("Link")),
                author_keywords=_clean_str(row.get("Author Keywords")),
                index_keywords=_clean_str(row.get("Index Keywords")),
                cos_author_ids=sorted(cos_author_ids),  # sorted -> deterministic
            )
        )
        seen_eids.add(eid)

    return publications


def index_by_author(publications: list[Publication]) -> dict[str, list[Publication]]:
    """Scopus ID -> that author's publications (roster-filtered already)."""
    index: dict[str, list[Publication]] = {}
    for pub in publications:
        for author_id in pub.cos_author_ids:
            index.setdefault(author_id, []).append(pub)
    return index
