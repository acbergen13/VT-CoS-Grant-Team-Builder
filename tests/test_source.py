"""FacultySource factory + CSV implementation end-to-end on synthetic files."""

import pandas as pd
import pytest

from gt.faculty.source import CsvFacultySource, get_faculty_source

AUTHORS_COLUMNS = ["Author", "Scopus ID", "H-Index", "H5 Index", "Scholarly Output",
                   "Citation Count", "Top Journal Quartile %", "Department", "Rank", "Tenure"]
PUBS_COLUMNS = ["Authors", "Author full names", "Author(s) ID", "Title", "Year",
                "Source title", "Cited by", "DOI", "Link", "Abstract",
                "Author Keywords", "Index Keywords", "Document Type", "EID"]


def write_fixtures(tmp_path):
    authors = tmp_path / "authors.csv"
    pubs = tmp_path / "publications.csv"
    pd.DataFrame(
        [
            {"Author": "Alpha, Ann", "Scopus ID": "1", "Rank": "Professor"},
            {"Author": "Beta, Bob", "Scopus ID": "2", "Rank": "Postdoctoral Associate"},
        ],
        columns=AUTHORS_COLUMNS,
    ).to_csv(authors, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"Author(s) ID": "1; 999", "Title": "Roster paper", "EID": "E1", "Year": "2024"},
            {"Author(s) ID": "999", "Title": "External only", "EID": "E2", "Year": "2024"},
        ],
        columns=PUBS_COLUMNS,
    ).to_csv(pubs, index=False, encoding="utf-8-sig")
    return authors, pubs


def csv_settings():
    return {
        "data_source": "csv",
        "scoring": {"missing_metric_impute": 0, "weights": {}},
        "team": {"pi_eligible_ranks": ["Professor"]},
        "retrieval": {"exclude_doc_types": ["Erratum"]},
    }


def test_csv_source_end_to_end(tmp_path):
    authors, pubs = write_fixtures(tmp_path)
    source = CsvFacultySource(authors, pubs, csv_settings())
    roster = source.load_roster()
    assert len(roster) == 2
    assert roster.by_id["1"].pi_eligible is True
    publications = source.load_publications()
    assert [p.eid for p in publications] == ["E1"]  # external-only row dropped (D4)
    assert publications[0].cos_author_ids == ["1"]


def test_factory_returns_csv_source_for_csv_mode():
    source = get_faculty_source(csv_settings())
    assert isinstance(source, CsvFacultySource)


def test_factory_returns_elsevier_source(monkeypatch, tmp_path):
    monkeypatch.setenv("ELSEVIER_API_KEY", "test-key")
    monkeypatch.setenv("ELSEVIER_BASE_URL", "https://api.elsevier.test")
    from gt.faculty.elsevier import ElsevierFacultySource

    source = get_faculty_source(
        {
            **csv_settings(),
            "data_source": "elsevier",
            "elsevier": {"cache_dir": str(tmp_path / "cache")},
        }
    )
    assert isinstance(source, ElsevierFacultySource)


def test_factory_rejects_unknown_mode():
    with pytest.raises(ValueError, match="data_source"):
        get_faculty_source({"data_source": "telepathy"})
