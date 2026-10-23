"""FacultySource: one retrieval interface for dev (CSV) and prod (Elsevier).

Req 5 / req 26: development reads the local CSVs; the production version
retrieves publication data directly from the Elsevier API behind the SAME
retrieval interface, with local caching to limit repeat calls. This module is
the seam/factory; the production implementation lives in
``gt.faculty.elsevier`` and is re-exported here for callers and tests.

Callers should always go through get_faculty_source() so the data_source
switch in settings.yaml controls which implementation runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from gt.config import load_settings
from gt.faculty.elsevier import ElsevierFacultySource
from gt.faculty.publications import PUBLICATIONS_CSV, Publication, load_publications
from gt.faculty.roster import AUTHORS_CSV, Roster, load_roster


class FacultySource(Protocol):
    """The retrieval seam. Both implementations must return roster-filtered,
    cleaned data — callers never see raw CSV/API shapes."""

    def load_roster(self) -> Roster: ...

    def load_publications(self) -> list[Publication]: ...


class CsvFacultySource:
    """Development source: the two local CSVs (authors.csv, publications.csv)."""

    def __init__(
        self,
        authors_path: Path | str | None = None,
        publications_path: Path | str | None = None,
        settings: dict | None = None,
    ) -> None:
        self._authors_path = Path(authors_path) if authors_path else AUTHORS_CSV
        self._publications_path = (
            Path(publications_path) if publications_path else PUBLICATIONS_CSV
        )
        self._settings = settings

    def load_roster(self) -> Roster:
        return load_roster(self._authors_path, self._settings)

    def load_publications(self) -> list[Publication]:
        # The CoS-author filter (D4) needs the roster's ID set, so load it
        # first — one extra cheap read of a 501-row file.
        roster = self.load_roster()
        return load_publications(roster, self._publications_path, self._settings)


def get_faculty_source(settings: dict | None = None) -> FacultySource:
    """Factory keyed on settings.yaml data_source (csv | elsevier)."""
    cfg = settings or load_settings()
    mode = cfg.get("data_source")
    if mode == "csv":
        return CsvFacultySource(settings=cfg)
    if mode == "elsevier":
        return ElsevierFacultySource(settings=cfg)
    raise ValueError(
        f"unknown data_source {mode!r} in config/settings.yaml (expected csv|elsevier)"
    )
