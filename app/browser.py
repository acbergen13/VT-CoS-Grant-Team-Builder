"""View 1: solicitation browser (refresh, filter/sort, select, build a team).

Data flow (architecture doc): the grants.gov search2 harvest lands in the
local JSON store; this view reads the store, filters/sorts client-side
(req 7), and fetches full details via fetchOpportunity only when the user
selects an opportunity (req 11). "Build a team" packages the selection into
the session-state Solicitation object and switches to the chat view (req 24).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from gt.grants.client import GrantsClient, GrantsClientError
from gt.grants.service import load_detail, run_refresh
from gt.grants.store import default_store
from gt.state import Solicitation, init_session_state

SORT_OPTIONS = {
    "Close date (soonest first)": "close_date",
    "Award ceiling (highest first)": "award_desc",
    "Recently opened": "open_desc",
    "Title (A-Z)": "title",
}


def render() -> None:
    st.title("VT CoS Grant Team Builder")
    st.caption(
        "Browse open grants.gov solicitations and build Virginia Tech College "
        "of Science faculty teams."
    )
    init_session_state()
    store = default_store()
    client = GrantsClient()

    _render_refresh_controls(store, client)
    opps = _load_opps(store)
    if not opps:
        st.info(
            "No solicitations in the local database yet — click "
            "**Refresh solicitations** above to harvest open opportunities "
            "from grants.gov."
        )
        return

    filters = _render_filters(opps)
    filtered = _apply_filters(opps, filters)
    _apply_sort(filtered, filters["sort"])
    st.caption(f"Showing **{len(filtered)}** of {len(opps)} open solicitations.")

    selected = _render_table(filtered)
    if selected is not None:
        _render_detail_and_handoff(client, store, selected, filtered)


def _render_refresh_controls(store, client: GrantsClient) -> None:
    """Refresh button + last-harvest metadata (reqs 1, 6, 23)."""
    left, right = st.columns([1, 3])
    if left.button("🔄 Refresh solicitations", type="primary"):
        with st.spinner("Harvesting open grants.gov solicitations..."):
            try:
                result = run_refresh(client, store)
            except GrantsClientError as exc:
                st.error(f"Refresh failed: {exc}")  # req 23: user-facing error
                return
        st.session_state.opps = None  # invalidate the listing cache
        st.session_state.last_refresh_result = result
        st.success(
            f"Harvested **{result.harvested}** open opportunities — "
            f"{len(result.newly_added)} new, {len(result.updated)} updated, "
            f"{len(result.newly_closed)} newly closed (archived, never shown)."
        )
    harvested_at = store.harvested_at()
    if harvested_at:
        right.caption(f"Local database last refreshed: {harvested_at} UTC")


def _load_opps(store) -> list[dict]:
    """Listing cached in session state; re-read only after a refresh."""
    if st.session_state.opps is None:
        st.session_state.opps = store.load_active()
    return st.session_state.opps


def _render_filters(opps: list[dict]) -> dict:
    """Sidebar filter/sort controls (req 7)."""
    with st.sidebar:
        st.header("Filters")
        status = st.selectbox("Status", ["All", "posted", "forecasted"])
        doc_types = sorted({o.get("doc_type") or "unknown" for o in opps})
        doc_type = st.selectbox("Opportunity type", ["All", *doc_types])
        agency_query = st.text_input("Agency contains", "")
        keyword = st.text_input(
            "Keyword (title / CFDA)", ""
        )
        st.subheader("Closes between")
        close_from = st.date_input("From", value=None, key="close_from")
        close_to = st.date_input("To", value=None, key="close_to")
        min_award = st.number_input(
            "Min award ceiling ($)",
            min_value=0,
            value=0,
            step=100000,
            help="Award amounts are known once an opportunity's details have "
            "been fetched (select it once, or refresh details).",
        )
        st.subheader("Sort")
        sort = st.selectbox("Sort by", list(SORT_OPTIONS))
    return {
        "status": status,
        "doc_type": doc_type,
        "agency_query": agency_query,
        "keyword": keyword,
        "close_from": close_from,
        "close_to": close_to,
        "min_award": min_award,
        "sort": sort,
    }


def _apply_filters(opps: list[dict], f: dict) -> list[dict]:
    out = []
    for o in opps:
        if f["status"] != "All" and o.get("status") != f["status"]:
            continue
        if f["doc_type"] != "All" and (o.get("doc_type") or "unknown") != f["doc_type"]:
            continue
        if f["agency_query"]:
            haystack = f"{o.get('agency', '')} {o.get('agency_code', '')}".lower()
            if f["agency_query"].lower() not in haystack:
                continue
        if f["keyword"]:
            hay = " ".join(
                [o.get("title", ""), " ".join(o.get("cfda") or [])]
            ).lower()
            if f["keyword"].lower() not in hay:
                continue
        close_date = o.get("close_date")
        if f["close_from"] and (not close_date or close_date < f["close_from"].isoformat()):
            continue
        if f["close_to"] and (not close_date or close_date > f["close_to"].isoformat()):
            continue
        if f["min_award"] and (o.get("award_ceiling") or 0) < f["min_award"]:
            continue
        out.append(o)
    return out


def _apply_sort(opps: list[dict], sort_key: str) -> None:
    if sort_key == "close_date":
        opps.sort(key=lambda o: (o.get("close_date") is None, o.get("close_date") or "9999-12-31"))
    elif sort_key == "award_desc":
        opps.sort(key=lambda o: o.get("award_ceiling") or 0, reverse=True)
    elif sort_key == "open_desc":
        opps.sort(key=lambda o: o.get("open_date") or "", reverse=True)
    else:
        opps.sort(key=lambda o: o.get("title", "").casefold())


def _money(value) -> str:
    return f"${value:,.0f}" if isinstance(value, (int, float)) else "—"


def _render_table(opps: list[dict]) -> dict | None:
    """Virtualized table with single-row selection (indices align with `opps`)."""
    rows = [
        {
            "ID": o.get("opp_id", ""),
            "Title": o.get("title", ""),
            "Status": o.get("status", ""),
            "Type": o.get("doc_type", ""),
            "Agency": o.get("agency", ""),
            "Opened": o.get("open_date") or "—",
            "Closes": o.get("close_date") or "—",
            "Award ceiling": _money(o.get("award_ceiling")),
            "Funding": o.get("funding_instrument") or "—",
        }
        for o in opps
    ]
    event = st.dataframe(
        pd.DataFrame(rows),
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
        height=430,
        hide_index=True,
    )
    selected_rows = event.selection.rows
    if not selected_rows:
        return None
    return opps[selected_rows[0]]


def _render_detail_and_handoff(client, store, selected: dict, filtered: list[dict]) -> None:
    """Fetch detail for the selection (req 11) and offer the chatbot handoff."""
    opp_id = selected.get("opp_id", "")
    with st.spinner(f"Fetching opportunity {opp_id} details from grants.gov..."):
        try:
            solicitation = load_detail(client, store, opp_id)
        except GrantsClientError as exc:
            st.error(f"Could not load details: {exc}")  # req 23
            return

    st.divider()
    st.subheader(solicitation.title)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", solicitation.status or "—")
    col2.metric("Closes", solicitation.close_date or "open-ended")
    col3.metric("Award ceiling", _money(solicitation.award_ceiling))
    col4.metric("Funding", solicitation.funding_instrument or "—")

    with st.expander("Solicitation details", expanded=True):
        st.markdown(
            f"**Opportunity:** {solicitation.number} · **Agency:** "
            f"{solicitation.agency} ({solicitation.agency_code}) · "
            f"**Category:** {solicitation.category or '—'} · "
            f"**Cost sharing:** {'Yes' if solicitation.cost_sharing else 'No' if solicitation.cost_sharing is not None else '—'}"
        )
        if solicitation.eligibility:
            st.markdown(f"**Eligibility:** {solicitation.eligibility}")
        if solicitation.estimated_funding is not None:
            st.markdown(f"**Estimated funding:** {_money(solicitation.estimated_funding)}"
                        + (f" ({solicitation.number_of_awards} expected award(s))" if solicitation.number_of_awards else ""))
        st.markdown(f"[View on grants.gov]({solicitation.detail_url})")
        if solicitation.synopsis:
            st.markdown("**Synopsis**")
            st.write(solicitation.synopsis)
        else:
            st.info("No synopsis text available for this opportunity yet.")

    if st.button("🚀 Build a team", type="primary"):
        # req 24: hand the structured solicitation to the chatbot view via
        # shared session state — no re-fetch, no re-entry. A different
        # solicitation starts a fresh conversation; re-opening the same one
        # keeps the existing team and chat history.
        current = st.session_state.solicitation
        if current is None or current.opp_id != solicitation.opp_id:
            st.session_state.team = None
            st.session_state.team_history = []
            st.session_state.messages = []
            st.session_state.last_build_result = None
            st.session_state.chat_build_attempted = False
            st.session_state.chat_error = ""
        st.session_state.solicitation = solicitation
        st.session_state.view = "chat"
        st.rerun()
