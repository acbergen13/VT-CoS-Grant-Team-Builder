"""Deterministic scoring tests (req 12/19) — hand-computed expectations.

Roster used below (imputation already applied):
  A: h5=10, citations=100
  B: h5=20, citations=10000
  C: h5=0 (missing -> imputed), citations=0
Bounds: h5 in [0, 20] -> plain min-max; citations log1p in [0, ln(10001)].
Expected scores (weights 0.6 / 0.25 / 0.15):
  A: 0.6*0.8 + 0.25*0.5  + 0.15*ln(101)/ln(10001)  ~= 0.6802
  B: 0.6*0.2 + 0.25*1.0  + 0.15*1.0                == 0.52
  C: 0.6*0.5 + 0.25*0.0  + 0.15*0.0                == 0.30
"""

import math

import pytest

from gt.faculty.roster import Faculty, Roster
from gt.faculty.scoring import MetricNormalizer, RosterScorer

SETTINGS = {
    "scoring": {
        "weights": {"semantic": 0.6, "h5": 0.25, "citations": 0.15},
        "h5_normalization": "minmax",
        "citations_normalization": "log1p_minmax",
    }
}


def make_roster():
    return Roster(
        faculty=[
            Faculty(scopus_id="A", name="A A", h5=10, citations=100),
            Faculty(scopus_id="B", name="B B", h5=20, citations=10000),
            Faculty(scopus_id="C", name="C C", h5=0, citations=0),
        ]
    )


def test_hand_computed_scores_and_order():
    scorer = RosterScorer(make_roster(), SETTINGS)
    ranked = scorer.score({"A": 0.8, "B": 0.2, "C": 0.5})
    assert [c.faculty.scopus_id for c in ranked] == ["A", "B", "C"]
    assert ranked[0].score == pytest.approx(0.6802, abs=1e-3)
    assert ranked[1].score == pytest.approx(0.52)
    assert ranked[2].score == pytest.approx(0.30)
    # Component breakdown is exposed for the explainable UI (req 27).
    assert ranked[0].components["semantic"] == pytest.approx(0.8)
    assert ranked[0].components["h5"] == pytest.approx(0.5)
    assert ranked[0].components["citations"] == pytest.approx(
        math.log1p(100) / math.log1p(10000)
    )


def test_weights_come_from_settings_not_code():
    custom = {
        "scoring": {
            "weights": {"semantic": 1.0, "h5": 0.0, "citations": 0.0},
            "h5_normalization": "minmax",
            "citations_normalization": "log1p_minmax",
        }
    }
    ranked = RosterScorer(make_roster(), custom).score({"B": 1.0})
    assert ranked[0].score == pytest.approx(1.0)  # pure semantic ranking


def test_missing_semantic_defaults_to_zero():
    ranked = RosterScorer(make_roster(), SETTINGS).score({})  # no retrieval yet
    # With no semantic signal, order falls back to metrics: B > A > C.
    assert [c.faculty.scopus_id for c in ranked] == ["B", "A", "C"]
    assert ranked[0].components["semantic"] == 0.0


def test_semantic_is_clamped_into_unit_interval():
    ranked = RosterScorer(make_roster(), SETTINGS).score({"A": 5.0, "B": -3.0})
    by_id = {c.faculty.scopus_id: c for c in ranked}
    assert by_id["A"].components["semantic"] == 1.0
    assert by_id["B"].components["semantic"] == 0.0


def test_reruns_are_bit_identical():
    scorer = RosterScorer(make_roster(), SETTINGS)
    semantic = {"A": 0.8, "B": 0.2, "C": 0.5}
    first = scorer.score(semantic)
    second = scorer.score(semantic)
    assert first == second  # req 19: exact equality, not approx


def test_ties_break_deterministically_by_scopus_id():
    roster = Roster(
        faculty=[
            Faculty(scopus_id="2", name="Tie Two", h5=10, citations=100),
            Faculty(scopus_id="1", name="Tie One", h5=10, citations=100),
        ]
    )
    ranked = RosterScorer(roster, SETTINGS).score({"1": 0.5, "2": 0.5})
    assert [c.faculty.scopus_id for c in ranked] == ["1", "2"]
    assert ranked[0].score == ranked[1].score


def test_degenerate_roster_scores_zero():
    single = Roster(faculty=[Faculty(scopus_id="X", name="Only", h5=7, citations=9)])
    ranked = RosterScorer(single, SETTINGS).score({"X": 0.0})
    assert ranked[0].score == 0.0  # no spread -> all normalized to 0


def test_metric_normalizer_modes():
    mm = MetricNormalizer([0, 10], "minmax")
    assert mm(5) == pytest.approx(0.5)
    lm = MetricNormalizer([0, 99], "log1p_minmax")
    assert lm(0) == 0.0
    assert lm(99) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        MetricNormalizer([1, 2], "median_split")
