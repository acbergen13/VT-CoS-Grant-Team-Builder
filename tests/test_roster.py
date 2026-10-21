"""Roster loading tests — synthetic CSVs (with BOM) in tmp_path, no real data."""

import pandas as pd

from gt.faculty.roster import load_roster

COLUMNS = [
    "Author", "Scopus ID", "H-Index", "H5 Index", "Scholarly Output",
    "Citation Count", "Top Journal Quartile %", "Department", "Rank", "Tenure",
]


def write_authors(path, rows):
    # utf-8-sig writes the BOM on purpose — every test exercises D11 handling.
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def settings(pi_ranks=None):
    return {
        "scoring": {"missing_metric_impute": 0, "weights": {}},
        "team": {"pi_eligible_ranks": pi_ranks or ["Professor", "Associate Professor"]},
    }


def row(author, sid, **kw):
    base = {
        "Author": author, "Scopus ID": sid, "H-Index": "10", "H5 Index": "5",
        "Scholarly Output": "40", "Citation Count": "500",
        "Top Journal Quartile %": "62.5", "Department": "Statistics",
        "Rank": "Professor", "Tenure": "Academic Tenure",
    }
    base.update(kw)
    return base


def test_basic_load_parses_bom_and_fields(tmp_path):
    path = tmp_path / "authors.csv"
    write_authors(path, [row("Zhu, Hongxiao", "56330169100")])
    roster = load_roster(path, settings())
    assert len(roster) == 1
    f = roster.by_id["56330169100"]  # by_id lookup works -> BOM was stripped
    assert f.name == "Zhu, Hongxiao"
    assert f.department == "Statistics"
    assert f.h5 == 5.0
    assert f.citations == 500.0
    assert f.top_journal_quartile == 62.5
    assert f.raw["Rank"] == "Professor"  # raw row retained for verification


def test_duplicate_scopus_id_keeps_first_and_records_drop(tmp_path):
    path = tmp_path / "authors.csv"
    write_authors(
        path,
        [
            row("Iliescu, Traian", "56940337900", Department="Mathematics"),
            row("Driscoll, Anne R.", "56940337900", Department="Statistics"),
        ],
    )
    roster = load_roster(path, settings())
    assert len(roster) == 1  # D1: one ID, one person
    assert roster.by_id["56940337900"].name == "Iliescu, Traian"  # file order
    assert roster.dropped_duplicates == [
        {"scopus_id": "56940337900", "kept": "Iliescu, Traian", "dropped": "Driscoll, Anne R."}
    ]


def test_missing_metrics_impute_but_quartile_stays_none(tmp_path):
    path = tmp_path / "authors.csv"
    write_authors(
        path,
        [
            row("A, One", "1", **{"H5 Index": None, "H-Index": None}),
            row("B, Two", "2", **{"Top Journal Quartile %": None, "Tenure": None}),
        ],
    )
    roster = load_roster(path, settings())
    assert roster.by_id["1"].h5 == 0.0  # D2: imputed, not NaN
    assert roster.by_id["1"].h_index == 0.0
    assert roster.by_id["2"].top_journal_quartile is None  # D7: unknown != zero
    assert roster.by_id["2"].tenure == ""  # NaN cell -> "", never the string "nan"


def test_pi_eligibility_is_rank_lookup(tmp_path):
    path = tmp_path / "authors.csv"
    write_authors(
        path,
        [
            row("Prof, Eve", "1", Rank="Professor"),
            row("Post, Doc", "2", Rank="Postdoctoral Associate"),
            row("Collegial, Cam", "3", Rank="Collegiate Professor"),
        ],
    )
    roster = load_roster(path, settings())  # eligible = Professor, Associate Professor
    assert roster.by_id["1"].pi_eligible is True
    assert roster.by_id["2"].pi_eligible is False  # postdocs never PI (D8)
    assert roster.by_id["3"].pi_eligible is False  # not in THIS config's list


def test_blank_scopus_id_rows_are_skipped(tmp_path):
    path = tmp_path / "authors.csv"
    write_authors(path, [row("Ghost, None", ""), row("Real, Person", "42")])
    roster = load_roster(path, settings())
    assert len(roster) == 1
    assert "42" in roster
