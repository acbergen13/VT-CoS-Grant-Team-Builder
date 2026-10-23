"""LLM layer tests — scripted fake client, no network, no real model.

Covers the Stage 4 gates under the deterministic-membership architecture
(decision 8, plan.md):
- the retrieval query is code-assembled (anchor + sorted LLM keywords), so
  reply phrasing/order drift cannot reshape the candidate pool (reqs 14, 19)
- membership is assigned greedily in code: same inputs -> bit-identical team
  (req 19), complete and valid even with the LLM fully down (reqs 13, 23)
- batched rationales are chain-of-verified: wrong person, invented EIDs, or
  empty text are rejected and regenerate; template rationales finish the
  rest (reqs 18, 27)
- the plan/build disk cache pins reproducibility and skips LLM calls, but
  never caches degraded fallback output (reqs 19, 21)
- critiques translate into validated slot ops; only affected slots change,
  the replaced person cannot be re-picked for their own slot, and ops
  outside the critique's named scope are dropped in code (req 15, issue 9)
"""

import json

import pytest

from gt.faculty.publications import Publication
from gt.faculty.roster import Faculty, Roster
from gt.faculty.scoring import RosterScorer
from gt.llm.arc import ArcClient, ArcConfigError, ArcUnavailableError
from gt.llm.cache import BuildCache, build_cache_key, plan_cache_key
from gt.llm.optimize import canonical_query, fallback_query, optimize_query
from gt.llm.prompts import parse_json_object
from gt.llm.team import plan_operations
from gt.llm.team_builder import DEFAULT_FOCUS, DEFAULT_ROLES, TeamBuilder
from gt.llm.verify import verify_rationale, verify_team
from gt.retrieval.evidence import build_evidence
from gt.retrieval.index import build_chunks
from gt.retrieval.pipeline import RetrievalResult
from gt.state import Solicitation, Team, TeamMember


# ------------------------------------------------------------------ fixtures


class FakeClient:
    """Scripted stand-in for ArcClient: pops queued replies (or raises them),
    records every prompt, counts calls like the real client."""

    model = "fake-model"  # cache-key input (matches ArcClient.model)

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        self.prompts.append(messages[0]["content"])
        if not self.replies:
            return "{}"
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def llm_settings(team_overrides=None, **llm_overrides):
    llm = {
        "temperature": 0.1,
        "verify_max_retries": 2,
        "seed": 42,
        "timeout_seconds": 30,
        "max_output_tokens": 512,
        "max_input_chars": 2000,
        "query_max_chars": 120,
        "build_cache": False,  # tests opt in with a tmp BuildCache
    }
    llm.update(llm_overrides)
    team = {
        "roles": [
            "Principal Investigator",
            "Co-PI",
            "Methods/Technical Lead",
            "Domain Expert",
            "Data/Computational Specialist",
            "Project Coordinator",
        ],
        # Test default: a 2-role structure so the 3-person stub pool fills
        # every slot. Full-base tests override with DEFAULT_ROLES.
        "base_roles": ["Principal Investigator", "Domain Expert"],
        "min_pi": 1,
        "min_additional_roles": 1,
        "one_role_per_person": True,
        "team_size": 5,
        "pi_eligible_ranks": ["Professor", "Assistant Professor"],
    }
    if team_overrides:
        team.update(team_overrides)
    return {
        "scoring": {
            "weights": {"semantic": 0.6, "h5": 0.25, "citations": 0.15},
            "h5_normalization": "minmax",
            "citations_normalization": "log1p_minmax",
        },
        "team": team,
        "retrieval": {"evidence_per_author": 3},
        "llm": llm,
    }


def pub(eid, author_ids, title, year):
    return Publication(
        eid=eid,
        title=title,
        abstract="",
        year=year,
        source_title="Journal of Tests",
        doc_type="Article",
        cited_by=0,
        doi=f"10.1/{eid}",
        link=f"https://scopus.com/{eid}",
        author_keywords="",
        index_keywords="",
        cos_author_ids=list(author_ids),
    )


PUBS = [
    pub("E1", ["1"], "Lake water quality modeling", 2020),
    pub("E2", ["1"], "Climate temperature trends", 2022),
    pub("E3", ["2"], "Neural network training methods", 2024),
    pub("E4", ["3"], "Machine learning for climate", 2023),
    pub("E5", ["4"], "Quantum computing gates", 2021),
]


class StubPipeline:
    """Just enough of RetrievalPipeline for the builder: a fixed ranked
    pool, evidence packs, the real fitted scorer, and the real chunks.

    Roster: 1 Climate (Professor, PI-eligible), 2 ML (Assistant Professor,
    PI-eligible), 3 Bridge (Postdoc, NOT PI-eligible), 4 Quantum (Professor).
    Default ranked order ("3","2","1") -> greedy PI = 2, next role = 3.
    """

    def __init__(self, settings, ranked_ids=("3", "2", "1"), semantic=None):
        self.settings = settings
        self.roster = Roster(
            faculty=[
                Faculty(scopus_id="1", name="Climate Person", department="Geo",
                        rank="Professor", pi_eligible=True, h5=10, citations=100),
                Faculty(scopus_id="2", name="ML Person", department="Stats",
                        rank="Assistant Professor", pi_eligible=True, h5=20, citations=10000),
                Faculty(scopus_id="3", name="Bridge Person", department="Bio",
                        rank="Postdoc", pi_eligible=False, h5=15, citations=5000),
                Faculty(scopus_id="4", name="Quantum Person", department="Phys",
                        rank="Professor", pi_eligible=True, h5=25, citations=20000),
            ]
        )
        self.chunks = build_chunks(PUBS)
        self.scorer = RosterScorer(self.roster, settings)
        self.semantic = semantic or {"3": 0.9, "1": 0.8, "2": 0.7}
        scored = {c.faculty.scopus_id: c for c in self.scorer.score(self.semantic)}
        self.ranked = [scored[i] for i in ranked_ids]
        by_eid = {c.id: c for c in self.chunks}
        self.evidence_by_id = {
            "1": build_evidence([(by_eid["E1"], 1.0), (by_eid["E2"], 0.9)]),
            "2": build_evidence([(by_eid["E3"], 1.0)]),
            "3": build_evidence([(by_eid["E4"], 1.0)]),
            "4": build_evidence([(by_eid["E5"], 1.0)]),
        }

    def retrieve(self, query):
        return RetrievalResult(
            query=query,
            ranked=list(self.ranked),
            evidence_by_id={k: list(v) for k, v in self.evidence_by_id.items()},
            semantic_by_id=dict(self.semantic),
        )


SOL = Solicitation(
    opp_id="42",
    title="AI for Climate Resilience",
    agency="NSF",
    synopsis="Using machine learning to model climate impacts on ecosystems.",
)

OPT_JSON = json.dumps(
    {"domains": ["machine learning", "climate science"],
     "methodology": ["ecosystem modeling"]}
)
# Canonical query (code-assembled): deterministic anchor + sorted keywords.
CANONICAL = (
    fallback_query(SOL, llm_settings())
    + " climate science ecosystem modeling machine learning"
)
DECOMP_JSON = json.dumps(
    {"roles": [
        {"role": "Principal Investigator", "focus": "Lead the climate AI project"},
        {"role": "Domain Expert", "focus": "Climate science depth"},
    ]}
)
# Greedy assignment from ranked ("3","2","1"): PI = 2 (ML, top PI-eligible),
# Domain Expert = 3 (Bridge, next ranked). One batched call covers both.
RATIONALE_JSON = json.dumps(
    {"members": [
        {"scopus_id": "2", "evidence_eids": ["E3"],
         "rationale": "ML Person leads data-driven climate modeling work."},
        {"scopus_id": "3", "evidence_eids": ["E4"],
         "rationale": "Bridge Person bridges machine learning and climate science."},
    ]}
)
# Full base-structure fixtures (production shape: the DEFAULT_ROLES four).
DECOMP4_JSON = json.dumps(
    {"roles": [
        {"role": "Principal Investigator", "focus": "Lead the climate AI project"},
        {"role": "Methods/Technical Lead", "focus": "Own the ML methodology"},
        {"role": "Domain Expert", "focus": "Climate science depth"},
        {"role": "Data/Computational Specialist", "focus": "Data pipelines"},
    ]}
)
RATIONALE4_JSON = json.dumps(
    {"members": [
        {"scopus_id": "2", "evidence_eids": ["E3"],
         "rationale": "ML Person leads data-driven climate modeling work."},
        {"scopus_id": "3", "evidence_eids": ["E4"],
         "rationale": "Bridge Person bridges machine learning and climate science."},
        {"scopus_id": "1", "evidence_eids": ["E1"],
         "rationale": "Climate Person provides climate science depth."},
        {"scopus_id": "4", "evidence_eids": ["E5"],
         "rationale": "Quantum Person handles data pipelines."},
    ]}
)


def make_builder(replies, settings=None, cache=None, **pipeline_kwargs):
    settings = settings or llm_settings()
    client = FakeClient(replies)
    builder = TeamBuilder(
        client, StubPipeline(settings, **pipeline_kwargs), settings, cache=cache
    )
    return builder, client


# ------------------------------------------------------- prompts / parsing


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go:\n{"a": {"b": "x}y"}}\nThanks!') == {
        "a": {"b": "x}y"}
    }
    with pytest.raises(ValueError):
        parse_json_object("no json here")


# --------------------------------------------------------------- optimize


def test_optimize_query_happy_path():
    client = FakeClient([OPT_JSON])
    result = optimize_query(client, SOL, llm_settings())
    assert result.source == "llm"
    assert result.query == CANONICAL
    assert result.domains == ["climate science", "machine learning"]  # sorted
    assert result.methodology == ["ecosystem modeling"]
    assert not result.note


def test_optimize_query_ignores_order_and_stray_fields():
    # Same keyword SET in a different order (plus a stray legacy "query"
    # field the schema no longer asks for) -> identical canonical query.
    reordered = json.dumps(
        {"query": "some free-text drift the schema no longer requests",
         "domains": ["climate science"],
         "methodology": ["ecosystem modeling", "machine learning"]}
    )
    result = optimize_query(FakeClient([reordered]), SOL, llm_settings())
    assert result.query == CANONICAL


def test_canonical_query_bounds_each_half():
    settings = llm_settings(query_max_chars=40)
    q = canonical_query(SOL, ["zzz " * 30], ["yyy " * 30], settings)
    assert len(q) <= 2 * 40 + 1  # anchor and keyword halves each capped


def test_optimize_query_falls_back_when_llm_down():
    client = FakeClient([ArcUnavailableError("endpoint down")])
    result = optimize_query(client, SOL, llm_settings())
    assert result.source == "fallback"
    assert result.query == fallback_query(SOL, llm_settings())
    assert "deterministic query" in result.note


def test_optimize_query_falls_back_on_junk():
    client = FakeClient(["I would rather not answer."])
    result = optimize_query(client, SOL, llm_settings())
    assert result.source == "fallback"


def test_optimize_query_falls_back_when_no_keywords():
    empty = json.dumps({"domains": [], "methodology": []})
    result = optimize_query(FakeClient([empty]), SOL, llm_settings())
    assert result.source == "fallback"


# ---------------------------------------------------- rationale verification


def rationale_entry(scopus_id, eids, text="Solid match for the role focus."):
    return {"scopus_id": scopus_id, "evidence_eids": eids, "rationale": text}


def test_verify_rationale_rejects_wrong_person():
    check = verify_rationale(
        rationale_entry("999", ["E1"]), "1", "Climate Person", {"E1", "E2"}
    )
    assert not check.ok
    assert any("must justify" in r for r in check.reasons)


def test_verify_rationale_rejects_invented_eids():
    check = verify_rationale(
        rationale_entry("1", ["E1", "E999"]), "1", "Climate Person", {"E1", "E2"}
    )
    assert not check.ok
    assert any("E999" in r for r in check.reasons)


def test_verify_rationale_rejects_empty_text():
    check = verify_rationale(rationale_entry("1", ["E1"], "  "), "1",
                             "Climate Person", {"E1", "E2"})
    assert not check.ok


def test_verify_rationale_accepts_valid_entry():
    check = verify_rationale(
        rationale_entry("1", ["E1", "E2"]), "1", "Climate Person", {"E1", "E2"}
    )
    assert check.ok, check.reasons


def test_verify_team_rejects_injected_fake_author_and_publication():
    chunks = build_chunks(PUBS)
    by_eid = {c.id: c for c in chunks}
    evidence = {"1": build_evidence([(by_eid["E1"], 1.0)])}
    fake = Team(
        solicitation_opp_id="42",
        members=[
            TeamMember(
                scopus_id="999",
                name="Fake Person",
                role="Principal Investigator",
                evidence_eids=["E999"],
                rationale="Invented credentials.",
            )
        ],
    )
    check = verify_team(fake, evidence, llm_settings())
    assert not check.ok
    assert any("not in the verified candidate pool" in r for r in check.reasons)


def test_verify_team_rejects_unmatched_publication_and_empty_rationale():
    chunks = build_chunks(PUBS)
    by_eid = {c.id: c for c in chunks}
    evidence = {"1": build_evidence([(by_eid["E1"], 1.0)])}
    team = Team(
        solicitation_opp_id="42",
        members=[
            TeamMember(
                scopus_id="1",
                name="Climate Person",
                role="Principal Investigator",
                evidence_eids=["E999"],
                rationale="",
            )
        ],
    )
    check = verify_team(team, evidence, llm_settings())
    assert not check.ok
    assert any("unmatched evidence EIDs" in r for r in check.reasons)
    assert any("empty rationale" in r for r in check.reasons)


# ------------------------------------------------------------ team builder


def test_build_happy_path():
    builder, client = make_builder([OPT_JSON, DECOMP_JSON, RATIONALE_JSON])
    result = builder.build(SOL)
    assert result.team is not None
    roles = {m.role: m.scopus_id for m in result.team.members}
    # Greedy deterministic assignment: PI = top PI-eligible (2), then next.
    assert roles == {"Principal Investigator": "2", "Domain Expert": "3"}
    pi = result.team.members[0]
    assert pi.name == "ML Person"
    assert pi.evidence_eids == ["E3"]  # LLM's verified subset replaced defaults
    assert "data-driven" in pi.rationale
    assert pi.score_components.keys() == {"semantic", "h5", "citations"}
    assert result.llm_calls == 3  # optimize + decompose + batched rationales
    assert result.notes == []
    assert result.team.query == CANONICAL
    assert verify_team(result.team, builder._pipeline.evidence_by_id,
                       llm_settings()).ok


def test_build_is_bit_identical_across_runs():
    teams = []
    for _ in range(3):
        builder, _ = make_builder([OPT_JSON, DECOMP_JSON, RATIONALE_JSON])
        teams.append(builder.build(SOL).team)
    assert teams[0] == teams[1] == teams[2]  # req 19: exact equality


def test_build_rejects_rationale_about_wrong_person_then_recovers():
    bad = json.dumps({"members": [rationale_entry("999", ["E1"])] * 2})
    builder, client = make_builder([OPT_JSON, DECOMP_JSON, bad, RATIONALE_JSON])
    result = builder.build(SOL)
    assert result.llm_calls == 4  # one regeneration
    pi = result.team.members[0]
    assert pi.rationale  # recovered: verified rationale applied
    assert "data-driven" in pi.rationale or "bridges" in pi.rationale


def test_build_persistent_rationale_failures_fall_back_to_templates():
    bad = json.dumps({"members": [rationale_entry("999", ["E1"])] * 2})
    builder, _ = make_builder([OPT_JSON, DECOMP_JSON, bad, bad, bad])
    result = builder.build(SOL)
    assert result.team is not None
    for m in result.team.members:
        assert "Highest-ranked remaining candidate" in m.rationale
    assert any("could not be verified" in n for n in result.notes)
    assert verify_team(result.team, builder._pipeline.evidence_by_id,
                       llm_settings()).ok


def test_build_rejects_invented_eids_in_rationale():
    bad = json.dumps({"members": [
        rationale_entry("2", ["E999"]), rationale_entry("3", ["E999"])]})
    builder, _ = make_builder([OPT_JSON, DECOMP_JSON, bad, bad, bad])
    result = builder.build(SOL)
    for m in result.team.members:
        # EIDs fall back to the member's real evidence pack (never invented).
        assert m.evidence_eids
        assert "E999" not in m.evidence_eids


def test_build_with_llm_fully_down_still_returns_valid_team():
    down = ArcUnavailableError("endpoint down")
    settings = llm_settings(team_overrides={"base_roles": [r for r, _ in DEFAULT_ROLES]})
    builder, _ = make_builder([down] * 10, settings)
    result = builder.build(SOL)
    assert result.team is not None  # deterministic fallback everywhere
    assert result.optimized.source == "fallback"
    roles = [m.role for m in result.team.members]
    assert roles[0] == "Principal Investigator"
    # The base structure has 4 slots but the pool only has 3 people -> 3 filled.
    assert len(result.team.members) == 3
    for m in result.team.members:
        assert "Highest-ranked remaining candidate" in m.rationale
    check = verify_team(result.team, builder._pipeline.evidence_by_id, llm_settings())
    assert check.ok, check.reasons
    assert result.notes


def test_build_full_base_roles_greedy_fill():
    # Production shape (decision 9): the config-owned base structure is
    # always filled greedily, in order, regardless of what the LLM says.
    settings = llm_settings(team_overrides={"base_roles": [r for r, _ in DEFAULT_ROLES]})
    builder, _ = make_builder(
        [OPT_JSON, DECOMP4_JSON, RATIONALE4_JSON], settings,
        ranked_ids=("3", "2", "1", "4"),
    )
    result = builder.build(SOL)
    roles = {m.role: m.scopus_id for m in result.team.members}
    assert roles == {
        "Principal Investigator": "2",
        "Methods/Technical Lead": "3",
        "Domain Expert": "1",
        "Data/Computational Specialist": "4",
    }
    assert result.llm_calls == 3


def test_decompose_missing_or_junk_focuses_get_defaults():
    # Partial/junk decomposition replies can never change the structure:
    # unknown roles are ignored; known roles without a focus get the
    # generic default focus.
    settings = llm_settings(team_overrides={"base_roles": [r for r, _ in DEFAULT_ROLES]})
    partial = json.dumps(
        {"roles": [
            {"role": "Principal Investigator", "focus": "Lead it"},
            {"role": "Not A Real Role", "focus": "junk"},
        ]}
    )
    builder, _ = make_builder(
        [OPT_JSON, partial, RATIONALE4_JSON], settings,
        ranked_ids=("3", "2", "1", "4"),
    )
    result = builder.build(SOL)
    assert result.team is not None
    prompt = builder._client.prompts[2]  # the batched rationale prompt
    assert "Role focus: Lead it" in prompt
    assert f"Role focus: {DEFAULT_FOCUS}" in prompt


def test_build_decompose_junk_uses_default_roles():
    rationales = json.dumps(
        {"members": [rationale_entry(str(i), []) for i in (2, 3, 1, 4)]}
    )
    settings = llm_settings(team_overrides={"base_roles": [r for r, _ in DEFAULT_ROLES]})
    builder, _ = make_builder(
        [OPT_JSON, "I cannot do that.", rationales],
        settings,
        ranked_ids=("3", "2", "1", "4"),
    )
    result = builder.build(SOL)
    roles = [m.role for m in result.team.members]
    assert roles == [r for r, _ in DEFAULT_ROLES]


def test_build_empty_retrieval_returns_no_team():
    builder, _ = make_builder([OPT_JSON], ranked_ids=())
    result = builder.build(SOL)
    assert result.team is None
    assert any("No faculty matched" in n for n in result.notes)


def test_build_widens_pi_pool_when_no_pi_eligible_in_rank():
    # Ranked pool has ONLY the postdoc; the PI must come from the
    # deterministic widening (Climate Person, semantic 1.0) with a
    # recency-sorted evidence pack (E2 2022 before E1 2020).
    good = json.dumps({"members": [
        rationale_entry("1", ["E2"]), rationale_entry("3", ["E4"])]})
    builder, _ = make_builder(
        [OPT_JSON, DECOMP_JSON, good],
        ranked_ids=("3",), semantic={"3": 0.9, "1": 1.0},
    )
    result = builder.build(SOL)
    pi = result.team.members[0]
    assert pi.scopus_id == "1" and pi.role == "Principal Investigator"
    assert pi.evidence_eids == ["E2"]  # verified LLM subset of the recency pack
    assert any("widened" in n for n in result.notes)


# ------------------------------------------------------------------ cache


def test_build_cache_roundtrip_and_corruption(tmp_path):
    cache = BuildCache(tmp_path)
    assert cache.get("nope") is None
    cache.put("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert cache.get("bad") is None  # corrupt entry -> miss, no crash


def test_cache_keys_are_stable_and_sensitive():
    k1 = plan_cache_key("42", "m", "T", "S", extra="")
    k2 = plan_cache_key("42", "m", "T", "S", extra="")
    k3 = plan_cache_key("42", "other-model", "T", "S", extra="")
    k4 = build_cache_key("42", "m", 0, "anchor", extra="")
    k5 = build_cache_key("42", "m", 1, "anchor", extra="")
    k6 = plan_cache_key("42", "m", "T", "S", extra="", corpus="elsevier:abc")
    assert k1 == k2 and k1 != k3  # model is part of the key
    assert k4 != k5  # revision is part of the key
    assert k1 != k6  # data source/corpus fingerprint is part of the key


def test_cached_build_skips_llm_entirely(tmp_path):
    settings = llm_settings(build_cache=True)
    cache = BuildCache(tmp_path)
    builder1, client1 = make_builder(
        [OPT_JSON, DECOMP_JSON, RATIONALE_JSON], settings, cache=cache
    )
    first = builder1.build(SOL)
    assert first.llm_calls == 3 and not first.cached

    builder2, client2 = make_builder([], settings, cache=cache)  # no replies
    second = builder2.build(SOL)
    assert second.cached
    assert client2.calls == 0  # zero LLM calls on the cache hit
    assert second.team == first.team  # bit-identical (req 19)


def test_fallback_builds_are_not_cached(tmp_path):
    settings = llm_settings(build_cache=True)
    cache = BuildCache(tmp_path)
    down = ArcUnavailableError("down")
    builder1, _ = make_builder([down, down] + [down] * 5, settings, cache=cache)
    first = builder1.build(SOL)
    assert first.team is not None and first.optimized.source == "fallback"

    # LLM recovered: the next build must re-optimize (no cached fallback).
    builder2, client2 = make_builder(
        [OPT_JSON, DECOMP_JSON, RATIONALE_JSON], settings, cache=cache
    )
    second = builder2.build(SOL)
    assert not second.cached
    assert client2.calls == 3
    assert second.optimized.source == "llm"


# ----------------------------------------------------------------- revise


def built_team(**kwargs):
    builder, _ = make_builder(
        [OPT_JSON, DECOMP_JSON, RATIONALE_JSON], **kwargs
    )
    result = builder.build(SOL)
    return builder, result.team


def test_revise_replaces_only_targeted_slot():
    builder, team = built_team()
    kept_before = team.members[1]  # Domain Expert (Bridge Person)
    ops = json.dumps({"operations": [
        {"action": "replace", "scopus_id": "2",
         "reason": "user wants a professor as PI"}]})
    good = json.dumps({"members": [rationale_entry("1", ["E1"])]})
    builder._client.replies = [ops, good]
    result = builder.revise(team, SOL, "make the PI a full professor")
    assert result.team.revision == 1
    assert result.team.members[1] is kept_before  # SAME object, untouched
    new_pi = result.team.members[0]
    # Replaced person (2) is excluded from their own slot; 1 is next eligible.
    assert new_pi.scopus_id == "1"
    assert new_pi.role == "Principal Investigator"
    assert "professor" in new_pi.rationale or new_pi.rationale


def test_revise_removing_sole_pi_becomes_replacement():
    builder, team = built_team()
    ops = json.dumps({"operations": [
        {"action": "remove", "scopus_id": "2", "reason": "not impressed"}]})
    good = json.dumps({"members": [rationale_entry("1", ["E1"])]})
    builder._client.replies = [ops, good]
    result = builder.revise(team, SOL, "drop the PI")
    roles = [m.role for m in result.team.members]
    assert roles.count("Principal Investigator") == 1  # req 13 guard
    assert result.team.members[0].scopus_id == "1"


def test_revise_ignores_ops_on_unknown_members():
    builder, team = built_team()
    ops = json.dumps({"operations": [
        {"action": "remove", "scopus_id": "424242", "reason": "ghost"}]})
    builder._client.replies = [ops]
    result = builder.revise(team, SOL, "remove the ghost")
    assert result.team.members == team.members  # unchanged
    assert any("ignored" in n for n in result.notes)


def test_revise_drops_overreach_ops_outside_critique_scope():
    # Issue 9 / mitigation 9: the critique names the PI, but the model also
    # proposes swapping the Domain Expert — the unrelated op must be dropped
    # in code and the kept member carried over untouched.
    builder, team = built_team()
    kept_before = team.members[1]  # Domain Expert (Bridge Person)
    ops = json.dumps({"operations": [
        {"action": "replace", "scopus_id": "2", "reason": "user asked"},
        {"action": "replace", "scopus_id": "3", "reason": "model overreach"}]})
    good = json.dumps({"members": [rationale_entry("1", ["E1"])]})
    builder._client.replies = [ops, good]
    result = builder.revise(team, SOL, "Replace the PI with someone else")
    assert result.team.members[1] is kept_before
    assert result.team.members[0].scopus_id == "1"
    assert any("ignored" in n for n in result.notes)


def test_critique_scope_guard_matches_member_names():
    builder, team = built_team()
    ops = json.dumps({"operations": [
        {"action": "replace", "scopus_id": "2", "reason": "overreach"}]})
    builder._client.replies = [ops]
    plan = plan_operations(builder._client, team, "Drop Bridge Person",
                           llm_settings())
    assert plan.ops == []  # op targets someone the critique never named


def test_critique_without_references_keeps_llm_scope():
    builder, team = built_team()
    ops = json.dumps({"operations": [
        {"action": "replace", "scopus_id": "2", "reason": "stronger team"},
        {"action": "replace", "scopus_id": "3", "reason": "stronger team"}]})
    builder._client.replies = [ops]
    plan = plan_operations(builder._client, team, "make the whole team stronger",
                           llm_settings())
    assert [op.scopus_id for op in plan.ops] == ["2", "3"]
    assert not plan.note


def test_plan_operations_survives_garbage_reply():
    builder, team = built_team()
    builder._client.replies = ["not json at all"]
    plan = plan_operations(builder._client, team, "do something", llm_settings())
    assert plan.ops == []
    assert "try rephrasing" in plan.note


# ------------------------------------------------------------- arc client


def test_arc_client_missing_env_keys(monkeypatch):
    for key in ("VT_ARC_API_KEY", "VT_ARC_BASE_URL", "VT_ARC_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(ArcConfigError) as exc:
        ArcClient(llm_settings())
    assert "VT_ARC_API_KEY" in str(exc.value)


def test_arc_client_constructs_with_env_set(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("VT_ARC_API_KEY", "test-key")
    monkeypatch.setenv("VT_ARC_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("VT_ARC_MODEL", "test-model")
    client = ArcClient(llm_settings())
    assert client.calls == 0
    assert client.model == "test-model"
