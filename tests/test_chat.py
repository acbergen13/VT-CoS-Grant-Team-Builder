"""Stage 5 chat-view tests — AppTest, fully offline.

These cover the UI contract that sits on top of the Stage 4 builder:
auto-build on handoff, traceable team cards, revision turns, clean errors,
and preserving a built team when the same solicitation is re-opened.
"""

from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import app.chat as chat
from gt.faculty.publications import Publication
from gt.faculty.roster import Faculty, Roster
from gt.llm.arc import ArcUnavailableError
from gt.llm.optimize import OptimizedQuery
from gt.llm.team_builder import BuildResult
from gt.retrieval.index import Chunk
from gt.state import ChatTurn, Solicitation, Team, TeamMember


def make_pipeline():
    faculty = Faculty(
        scopus_id="1",
        name="Ada Lovelace",
        department="Computer Science",
        rank="Professor",
        h5=20,
        citations=500,
        pi_eligible=True,
    )
    pub = Publication(
        eid="E1",
        title="Methods for explainable team formation",
        year=2024,
        abstract="",
        source_title="Journal of Tests",
        doc_type="Article",
        cited_by=0,
        doi="10.1/E1",
        link="https://scopus.example/E1",
        author_keywords="",
        index_keywords="",
        cos_author_ids=["1"],
    )
    return SimpleNamespace(
        roster=Roster([faculty]),
        chunks=[Chunk(id="E1", text="Verbatim source text.", pub=pub)],
    )


def make_team(opp_id="42", revision=0):
    return Team(
        solicitation_opp_id=opp_id,
        members=[
            TeamMember(
                scopus_id="1",
                name="Ada Lovelace",
                department="Computer Science",
                rank="Professor",
                role="Principal Investigator",
                score=0.81,
                score_components={"semantic": 0.9, "h5": 0.8, "citations": 0.7},
                evidence_eids=["E1"],
                rationale="She matches the solicitation methods.",
            )
        ],
        revision=revision,
        query="team formation methods",
    )


def make_result(revision=0, cached=False):
    return BuildResult(
        team=make_team(revision=revision),
        optimized=OptimizedQuery(
            query="team formation methods",
            domains=["team formation"],
            methodology=["ranking"],
        ),
        pool_size=1,
        llm_calls=0 if cached else 3,
        revision=revision,
        cached=cached,
    )


class FakeBuilder:
    def __init__(self, build_result=None, revise_result=None, error=None):
        self.build_result = build_result or make_result()
        self.revise_result = revise_result or make_result(revision=1)
        self.error = error
        self.build_calls = 0
        self.revise_calls = 0

    def build(self, solicitation):
        self.build_calls += 1
        if self.error:
            raise self.error
        return self.build_result

    def revise(self, team, solicitation, critique):
        self.revise_calls += 1
        return self.revise_result


def run_chat(monkeypatch, solicitation, builder, pipeline=None):
    pipeline = pipeline or make_pipeline()
    monkeypatch.setattr(chat, "_get_builder", lambda: (builder, pipeline))
    at = AppTest.from_string(
        "import app.chat as chat\nchat.render()\n",
        default_timeout=10,
    )
    at.session_state["solicitation"] = solicitation
    at.session_state["pipeline"] = pipeline
    at.run()
    return at


def solicitation():
    return Solicitation(
        opp_id="42",
        number="NSF-42",
        title="Explainable Team Formation",
        agency="NSF",
        agency_code="NSF",
        status="posted",
        synopsis="Build reliable research teams.",
        detail_loaded=True,
    )


def test_auto_builds_and_renders_traceable_team(monkeypatch):
    at = run_chat(monkeypatch, solicitation(), FakeBuilder())

    assert not at.exception
    assert any("Team-Building Chatbot" in t.value for t in at.title)
    assert any("Proposed team" in s.value for s in at.subheader)
    assert any("Principal Investigator" in m.value for m in at.markdown)
    assert any(e.label.startswith("Supporting publications") for e in at.expander)
    assert any("E1" in m.value for m in at.caption)
    assert any("Build complete" in m.value for m in at.markdown)


def test_revision_turn_updates_team(monkeypatch):
    builder = FakeBuilder()
    at = run_chat(monkeypatch, solicitation(), builder)
    assert at.chat_input

    at.chat_input[0].set_value("replace the PI with a full professor").run()

    assert not at.exception
    assert builder.revise_calls == 1
    assert any("replace the PI" in m.value for m in at.markdown)
    assert any("Revision complete" in m.value for m in at.markdown)


def test_build_error_is_user_facing_and_retryable(monkeypatch):
    builder = FakeBuilder(error=ArcUnavailableError("endpoint down"))
    at = run_chat(monkeypatch, solicitation(), builder)

    assert not at.exception
    assert builder.build_calls == 1
    assert any("could not be reached" in e.value for e in at.error)
    assert any("Retry build" in b.label for b in at.button)


def test_same_solicitation_reuses_existing_team(monkeypatch):
    builder = FakeBuilder(error=AssertionError("must not rebuild"))
    pipeline = make_pipeline()
    monkeypatch.setattr(chat, "_get_builder", lambda: (builder, pipeline))
    at = AppTest.from_string(
        "import app.chat as chat\nchat.render()\n",
        default_timeout=10,
    )
    at.session_state["solicitation"] = solicitation()
    at.session_state["team"] = make_team()
    at.session_state["team_history"] = [make_team()]
    at.session_state["messages"] = []
    at.session_state["last_build_result"] = make_result()
    at.session_state["chat_build_attempted"] = True
    at.session_state["pipeline"] = pipeline
    at.run()

    assert not at.exception
    assert builder.build_calls == 0
    assert any("Proposed team" in s.value for s in at.subheader)


def test_new_solicitation_resets_old_conversation(monkeypatch):
    builder = FakeBuilder()
    pipeline = make_pipeline()
    monkeypatch.setattr(chat, "_get_builder", lambda: (builder, pipeline))
    at = AppTest.from_string(
        "import app.chat as chat\nchat.render()\n",
        default_timeout=10,
    )
    at.session_state["solicitation"] = solicitation()
    at.session_state["team"] = make_team(opp_id="old-opp")
    at.session_state["team_history"] = [make_team(opp_id="old-opp")]
    at.session_state["messages"] = [ChatTurn("assistant", "old conversation")]
    at.session_state["chat_build_attempted"] = True
    at.session_state["pipeline"] = pipeline
    at.run()

    assert not at.exception
    assert builder.build_calls == 1  # old attempt flag was cleared
    assert not any("old conversation" in m.value for m in at.markdown)
    assert any("Build complete" in m.value for m in at.markdown)
