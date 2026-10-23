"""View 2: team-building chatbot (Stage 5).

The browser hands over a fully loaded `Solicitation` object through session
state (req 24), so this view never re-fetches grants.gov. Entering the view
auto-builds the first proposal, then each chat turn is a critique that is
translated into targeted slot operations by the Stage 4 builder.

All heavy work stays behind the existing fixed workflow:
- `build_pipeline()` retrieves / ranks faculty (cached once per session);
- `TeamBuilder.build()` does 3 LLM calls, or 0 on a cache hit;
- `TeamBuilder.revise()` changes only the slots named in the critique.
The UI only renders results and surfaces errors/notes — it never chooses
members, roles, or scores.
"""

from __future__ import annotations

import streamlit as st

from gt.config import load_settings
from gt.faculty.elsevier import (
    ElsevierConfigError,
    ElsevierQuotaError,
    ElsevierResponseError,
    ElsevierUnavailableError,
)
from gt.llm.arc import ArcConfigError, ArcResponseError, ArcUnavailableError
from gt.llm.team_builder import BuildResult, TeamBuilder
from gt.retrieval.pipeline import RetrievalPipeline, build_pipeline
from gt.state import (
    ChatTurn,
    Solicitation,
    Team,
    TeamMember,
    init_session_state,
)

_BUILD_SPINNER = (
    "Retrieving faculty evidence and building a team... "
    "(the first run may download the embedding model)"
)
_REVISE_SPINNER = (
    "Revising the team — translating your critique, re-ranking candidates, "
    "and verifying the changes..."
)


def render() -> None:
    init_session_state()
    solicitation = st.session_state.solicitation
    if solicitation is None:
        _render_empty_state()
        return

    _reset_for_new_solicitation(solicitation)
    _render_header(solicitation)
    _maybe_build_first_team(solicitation)
    _render_messages()

    team = st.session_state.team
    if team is not None:
        _render_team(team, st.session_state.pipeline)
        _render_build_details(st.session_state.last_build_result)
        _render_revision_input(solicitation)
    elif st.session_state.chat_build_attempted:
        _render_retry()


def _render_empty_state() -> None:
    st.title("Team-Building Chatbot")
    st.info(
        "No solicitation selected yet — pick one in the browser view and click "
        "**Build a team**."
    )
    if st.button("← Back to browser"):
        st.session_state.view = "browser"
        st.rerun()


def _render_header(solicitation: Solicitation) -> None:
    title_col, back_col = st.columns([5, 1])
    with title_col:
        st.title("Team-Building Chatbot")
    with back_col:
        if st.button("← Browser", use_container_width=True):
            st.session_state.view = "browser"
            st.rerun()

    st.subheader(solicitation.title)
    st.caption(
        f"Opportunity {solicitation.opp_id} · {solicitation.agency} · "
        f"closes {solicitation.close_date or 'open-ended'}"
    )
    with st.expander("Solicitation context", expanded=False):
        st.markdown(
            f"**Opportunity:** {solicitation.number or solicitation.opp_id} · "
            f"**Status:** {solicitation.status or '—'} · "
            f"**Funding:** {solicitation.funding_instrument or '—'}"
        )
        if solicitation.eligibility:
            st.markdown(f"**Eligibility:** {solicitation.eligibility}")
        if solicitation.synopsis:
            st.markdown("**Synopsis**")
            st.write(solicitation.synopsis)
        st.markdown(f"[View on grants.gov]({solicitation.detail_url})")


def _reset_for_new_solicitation(solicitation: Solicitation) -> None:
    """A new browser selection starts a new conversation, but returning to
    the same solicitation preserves the current team (req 24)."""
    team = st.session_state.team
    if team is None or team.solicitation_opp_id == solicitation.opp_id:
        return
    _clear_conversation()


def _clear_conversation() -> None:
    st.session_state.team = None
    st.session_state.team_history = []
    st.session_state.messages = []
    st.session_state.last_build_result = None
    st.session_state.chat_build_attempted = False
    st.session_state.chat_error = ""


def _get_builder() -> tuple[TeamBuilder, RetrievalPipeline]:
    """Create the ARC client + retrieval pipeline once per Streamlit session.

    The pipeline's BM25 index and fitted scorer are expensive to rebuild on
    every rerun; the builder is stateless apart from the disk cache. Construction
    is lazy so entering the browser view never loads the corpus or model.
    """
    settings = load_settings()
    if st.session_state.arc_client is None:
        st.session_state.arc_client = _make_arc_client(settings)
    if st.session_state.pipeline is None:
        st.session_state.pipeline = build_pipeline(settings)
    if st.session_state.builder is None:
        st.session_state.builder = TeamBuilder(
            st.session_state.arc_client, st.session_state.pipeline, settings
        )
    return st.session_state.builder, st.session_state.pipeline


def _make_arc_client(settings: dict):
    # Imported lazily so .env validation only happens when a team is actually
    # built, never when the browser view or offline tests import app.chat.
    from gt.llm.arc import ArcClient

    return ArcClient(settings)


def _maybe_build_first_team(solicitation: Solicitation) -> None:
    if st.session_state.team is not None or st.session_state.chat_build_attempted:
        return
    st.session_state.chat_build_attempted = True
    st.session_state.chat_error = ""
    with st.spinner(_BUILD_SPINNER):
        try:
            builder, _ = _get_builder()
            result = builder.build(solicitation)
        except Exception as exc:  # req 23: never leak a traceback to the user
            _record_error(exc, "Build")
            return
    _record_build_result(result, action="build")


def _render_retry() -> None:
    if st.session_state.chat_error:
        st.error(st.session_state.chat_error)
    if st.button("Retry build", type="primary"):
        st.session_state.chat_build_attempted = False
        st.session_state.chat_error = ""
        st.rerun()


def _record_build_result(result: BuildResult, *, action: str) -> None:
    st.session_state.last_build_result = result
    if result.team is None:
        # A failed revision must not discard the last good proposal.
        if action != "revision":
            st.session_state.team = None
        st.session_state.chat_error = "\n".join(result.notes) or (
            "No faculty matched this solicitation well enough to build a team."
        )
        st.session_state.messages.append(
            ChatTurn("assistant", _result_summary(result, action))
        )
        return

    previous = st.session_state.team
    st.session_state.chat_error = ""
    st.session_state.team = result.team
    if previous is not result.team:
        st.session_state.team_history.append(result.team)
    st.session_state.messages.append(
        ChatTurn("assistant", _result_summary(result, action))
    )


def _record_error(exc: Exception, action: str) -> None:
    message = _format_error(exc, action)
    st.session_state.chat_error = message
    st.session_state.messages.append(ChatTurn("assistant", message))
    # Only the initial build hides the team on failure. A failed revision must
    # keep the last good proposal visible and usable.
    if action == "Build":
        st.session_state.team = None


def _format_error(exc: Exception, action: str) -> str:
    if isinstance(exc, ArcConfigError):
        return (
            "The LLM is not configured. Copy `.env.example` to `.env` and set "
            f"`VT_ARC_API_KEY`, `VT_ARC_BASE_URL`, and `VT_ARC_MODEL`. ({exc})"
        )
    if isinstance(exc, ArcUnavailableError):
        return (
            "The VT ARC LLM endpoint could not be reached. Check your network "
            f"and `VT_ARC_BASE_URL`, then retry. ({exc})"
        )
    if isinstance(exc, ArcResponseError):
        return (
            "The LLM returned no usable text. Retry, or raise "
            f"`llm.max_output_tokens` in config/settings.yaml. ({exc})"
        )
    if isinstance(exc, ElsevierConfigError):
        return (
            "Production publication data is not configured. Set "
            "`ELSEVIER_API_KEY` and `ELSEVIER_BASE_URL` in `.env`, or switch "
            f"`data_source` back to `csv`. ({exc})"
        )
    if isinstance(exc, ElsevierQuotaError):
        return (
            "Elsevier rejected the request or your quota/entitlement is "
            "exhausted. Check your key and campus network/inst token, or use "
            f"`data_source: csv`. ({exc})"
        )
    if isinstance(exc, ElsevierUnavailableError):
        return (
            "The Elsevier API could not be reached. Check your network and "
            "try again, or use `data_source: csv`. "
            f"({exc})"
        )
    if isinstance(exc, ElsevierResponseError):
        return (
            "Elsevier returned an unexpected response. Retry, or use "
            f"`data_source: csv` if the problem persists. ({exc})"
        )
    if isinstance(exc, FileNotFoundError):
        return (
            "The faculty corpus is missing. Make sure `data/authors.csv` and "
            "`data/publications.csv` are present, then retry. "
            f"({type(exc).__name__}: {exc})"
        )
    return f"{action} failed: {type(exc).__name__}: {exc}"


def _render_messages() -> None:
    for turn in st.session_state.messages:
        with st.chat_message(turn.role):
            st.markdown(turn.content)


def _render_team(team: Team, pipeline: RetrievalPipeline | None) -> None:
    if pipeline is not None:
        roster = pipeline.roster.by_id
        chunk_by_eid = {c.id: c for c in pipeline.chunks}
    else:  # defensive: a cached team should still render without a pipeline
        roster = {}
        chunk_by_eid = {}

    st.subheader("Proposed team")
    for member in team.members:
        _render_member(member, roster.get(member.scopus_id), chunk_by_eid)


def _render_member(member: TeamMember, faculty, chunk_by_eid: dict) -> None:
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"### {member.role}")
            st.markdown(f"**{member.name}**")
            details = [member.department, member.rank, f"Scopus ID {member.scopus_id}"]
            st.caption(" · ".join(p for p in details if p))
        with right:
            st.metric("Match score", f"{member.score:.3f}")

        components = member.score_components or {}
        component_text = " · ".join(
            f"{label} {float(components.get(key) or 0.0):.2f}"
            for key, label in (
                ("semantic", "Semantic"),
                ("h5", "H5"),
                ("citations", "Citations"),
            )
        )
        metrics = []
        if faculty is not None:
            metrics.append(f"H5 {faculty.h5:.0f}")
            metrics.append(f"Citations {faculty.citations:,.0f}")
        if metrics:
            st.caption(f"Deterministic score: {component_text} · " + " · ".join(metrics))
        else:
            st.caption(f"Deterministic score: {component_text}")

        st.markdown(member.rationale or "_No rationale available._")
        _render_evidence(member, chunk_by_eid)


def _render_evidence(member: TeamMember, chunk_by_eid: dict) -> None:
    chunks = [
        chunk_by_eid[eid] for eid in member.evidence_eids if eid in chunk_by_eid
    ]
    if not chunks:
        return
    with st.expander(f"Supporting publications ({len(chunks)})"):
        for chunk in chunks:
            pub = chunk.pub
            st.markdown(f"**{pub.title or '(untitled)'}**")
            meta = [str(pub.year) if pub.year else "n.d.", pub.source_title]
            if pub.doi:
                meta.append(f"DOI {pub.doi}")
            meta.append(f"EID {pub.eid}")
            st.caption(" · ".join(p for p in meta if p))
            if pub.link:
                st.markdown(f"[Open source record]({pub.link})")
            # Req 18: the rationale is traceable to the exact indexed text.
            st.caption("Verbatim indexed source text:")
            st.text(chunk.text[:1600])


def _render_build_details(result: BuildResult | None) -> None:
    if result is None:
        return
    with st.expander("Build details"):
        if result.optimized is not None:
            st.markdown("**Retrieval query**")
            st.code(result.optimized.query, language=None)
            if result.optimized.domains:
                st.caption("Domains: " + ", ".join(result.optimized.domains))
            if result.optimized.methodology:
                st.caption("Methods: " + ", ".join(result.optimized.methodology))
        st.caption(
            f"Revision {result.revision + 1} · candidate pool {result.pool_size} · "
            f"LLM calls this turn {result.llm_calls} · "
            f"{'cache hit' if result.cached else 'fresh build'}"
        )
        for note in result.notes:
            st.write(f"- {note}")


def _render_revision_input(solicitation: Solicitation) -> None:
    prompt = st.chat_input(
        "Ask for a change — e.g. 'replace the PI with a full professor'"
    )
    if not prompt:
        return

    st.session_state.messages.append(ChatTurn("user", prompt))
    with st.spinner(_REVISE_SPINNER):
        try:
            builder, _ = _get_builder()
            result = builder.revise(st.session_state.team, solicitation, prompt)
        except Exception as exc:  # req 23: keep the last good team visible
            _record_error(exc, "Revision")
            st.rerun()
            return

    _record_build_result(result, action="revision")
    st.rerun()


def _result_summary(result: BuildResult, action: str) -> str:
    if result.team is None:
        notes = " ".join(result.notes) or (
            "No faculty matched this solicitation well enough to build a team."
        )
        return f"I could not build a team. {notes}"

    lines = [
        f"**{action.capitalize()} complete** — "
        f"{len(result.team.members)} members, revision {result.team.revision + 1}."
    ]
    if result.cached:
        lines.append("Served from the local build cache (0 LLM calls).")
    elif result.llm_calls:
        lines.append(f"Used {result.llm_calls} LLM call(s).")
    if result.optimized is not None and result.optimized.source == "fallback":
        lines.append(
            "The LLM was unavailable for query optimization; a deterministic "
            "query built from the solicitation was used."
        )
    if result.notes:
        lines.append("\n".join(f"- {note}" for note in result.notes))
    return "\n\n".join(lines)
