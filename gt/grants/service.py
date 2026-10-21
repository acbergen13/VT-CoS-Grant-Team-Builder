"""Orchestration shared by the Streamlit UI and the CLI refresh script.

Keeping the harvest->store and select->detail flows here means both entry
points get identical semantics, and the transactionality rule (store is only
written after a fully successful harvest) lives in exactly one place.
"""

from __future__ import annotations

from gt.grants.client import GrantsClient, GrantsClientError, apply_detail
from gt.grants.store import GrantStore, RefreshResult
from gt.state import Solicitation


def run_refresh(client: GrantsClient, store: GrantStore) -> RefreshResult:
    """Harvest all open opportunities and update the store.

    Raises GrantsClientError on any harvest failure — the store is left
    untouched, so a failed harvest can never archive anything (req 23).
    """
    harvested = client.harvest()  # complete list or raises
    return store.refresh(harvested)


def load_detail(client: GrantsClient, store: GrantStore, opp_id: str) -> Solicitation:
    """Solicitation with fetchOpportunity detail applied (req 11).

    Uses the cached detail in the store when present; otherwise fetches,
    applies, and persists it back (no re-fetch on re-selection).
    """
    record = next(
        (o for o in store.load_active() if o.get("opp_id") == opp_id), None
    )
    if record is None:
        raise GrantsClientError(
            f"Opportunity {opp_id} is not in the local database — "
            "try refreshing solicitations first."
        )
    solicitation = Solicitation.from_dict(record)
    if solicitation.detail_loaded:
        return solicitation
    detail = client.fetch_opportunity(opp_id)
    apply_detail(solicitation, detail)
    store.save_detail(solicitation)
    return solicitation
