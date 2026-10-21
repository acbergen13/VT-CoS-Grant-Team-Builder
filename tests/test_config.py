"""Config invariants: the deterministic scoring contract must hold."""

from gt.config import load_settings


def test_settings_load_and_data_source():
    settings = load_settings()
    assert isinstance(settings, dict)
    assert settings["data_source"] in ("csv", "elsevier")


def test_normalization_modes_are_pinned():
    scoring = load_settings()["scoring"]
    assert scoring["h5_normalization"] == "minmax"  # H5 is bounded (1-26)
    assert scoring["citations_normalization"] == "log1p_minmax"  # skew (D3)
    assert scoring["missing_metric_impute"] == 0


def test_scoring_weights_are_the_fixed_contract():
    weights = load_settings()["scoring"]["weights"]
    assert set(weights) == {"semantic", "h5", "citations"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["semantic"] == 0.6
    assert weights["h5"] == 0.25
    assert weights["citations"] == 0.15


def test_role_list_is_fixed():
    expected = [
        "Principal Investigator",
        "Co-PI",
        "Methods/Technical Lead",
        "Domain Expert",
        "Data/Computational Specialist",
        "Project Coordinator",
    ]
    assert load_settings()["team"]["roles"] == expected


def test_pi_eligibility_and_mmr_bounds():
    settings = load_settings()
    assert settings["team"]["pi_eligible_ranks"], "PI-eligible ranks must not be empty"
    assert 0.0 < settings["retrieval"]["mmr_lambda"] < 1.0
    assert 0 < load_settings()["llm"]["temperature"] <= 0.2
