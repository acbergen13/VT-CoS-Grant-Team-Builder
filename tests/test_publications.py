"""Publications loading tests — synthetic CSVs, roster filtering (D4/D5)."""

import pandas as pd

from gt.faculty.publications import index_by_author, load_publications, parse_author_ids
from gt.faculty.roster import Faculty, Roster

COLUMNS = [
    "Authors", "Author full names", "Author(s) ID", "Title", "Year",
    "Source title", "Cited by", "DOI", "Link", "Abstract",
    "Author Keywords", "Index Keywords", "Document Type", "EID",
]


def write_pubs(path, rows):
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def pub(eid, author_ids, title="A paper", **kw):
    base = {
        "Authors": "Doe J.", "Author full names": "Doe, Jane (1)",
        "Author(s) ID": author_ids, "Title": title, "Year": "2024",
        "Source title": "Journal of Tests", "Cited by": "7", "DOI": "10.1/x",
        "Link": f"https://scopus.com/eid/{eid}", "Abstract": "An abstract.",
        "Author Keywords": "testing", "Index Keywords": "", "Document Type": "Article",
        "EID": eid,
    }
    base.update(kw)
    return base


def make_roster(*ids):
    return Roster(faculty=[Faculty(scopus_id=i, name=f"Name {i}") for i in ids])


def settings(exclude=None):
    return {"retrieval": {"exclude_doc_types": exclude or ["Erratum", "Editorial", "Note", "Letter"]}}


def test_parse_author_ids_splits_strips_and_skips_junk():
    assert parse_author_ids("1; 2; 3") == {"1", "2", "3"}
    assert parse_author_ids("1;;  ;2") == {"1", "2"}
    assert parse_author_ids(None) == set()


def test_load_filters_to_cos_authors(tmp_path):
    path = tmp_path / "pubs.csv"
    write_pubs(
        path,
        [
            pub("E1", "1; 999; 2"),      # roster + external co-authors
            pub("E2", "888; 999"),       # external only -> dropped (D4)
            pub("E3", ""),               # no authors -> dropped
        ],
    )
    pubs = load_publications(make_roster("1", "2"), path, settings())
    assert [p.eid for p in pubs] == ["E1"]
    assert pubs[0].cos_author_ids == ["1", "2"]  # external 999 filtered, sorted
    assert pubs[0].year == 2024
    assert pubs[0].cited_by == 7
    assert pubs[0].link == "https://scopus.com/eid/E1"


def test_load_excludes_noise_document_types(tmp_path):
    path = tmp_path / "pubs.csv"
    write_pubs(
        path,
        [
            pub("E1", "1", Document_Type="Article" if False else "Article"),
            pub("E2", "1", **{"Document Type": "Erratum"}),
            pub("E3", "1", **{"Document Type": "Editorial"}),
            pub("E4", "1", **{"Document Type": "Review"}),  # Reviews are kept
        ],
    )
    pubs = load_publications(make_roster("1"), path, settings())
    assert [p.eid for p in pubs] == ["E1", "E4"]  # D5


def test_missing_eid_skipped_and_duplicate_eid_keeps_first(tmp_path):
    path = tmp_path / "pubs.csv"
    write_pubs(
        path,
        [
            pub("", "1"),               # no canonical key (D9) -> skipped
            pub("E1", "1", title="First"),
            pub("E1", "1", title="Second"),  # defensive dedupe
        ],
    )
    pubs = load_publications(make_roster("1"), path, settings())
    assert len(pubs) == 1
    assert pubs[0].title == "First"


def test_bad_year_and_missing_abstract_are_tolerated(tmp_path):
    path = tmp_path / "pubs.csv"
    write_pubs(
        path,
        [
            pub("E1", "1", Year="not-a-year", Abstract=None, **{"Cited by": None}),
        ],
    )
    pubs = load_publications(make_roster("1"), path, settings())
    assert pubs[0].year is None
    assert pubs[0].abstract == ""
    assert pubs[0].cited_by == 0


def test_index_by_author_maps_every_cos_author(tmp_path):
    path = tmp_path / "pubs.csv"
    write_pubs(path, [pub("E1", "1; 2"), pub("E2", "2"), pub("E3", "1")])
    pubs = load_publications(make_roster("1", "2"), path, settings())
    index = index_by_author(pubs)
    assert [p.eid for p in index["1"]] == ["E1", "E3"]
    assert [p.eid for p in index["2"]] == ["E1", "E2"]
