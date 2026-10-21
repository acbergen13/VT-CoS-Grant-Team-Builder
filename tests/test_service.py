"""Service orchestration: failed harvests never touch the store; detail
fetches are cached after the first load."""

import pytest

from gt.grants.client import GrantsClientError, apply_detail
from gt.grants.service import load_detail, run_refresh
from gt.grants.store import GrantStore
from gt.state import Solicitation


class FailingClient:
    def harvest(self):
        raise GrantsClientError("grants.gov is not responding")


class FakeFetchClient:
    """Returns canned details and counts fetch_opportunity calls."""

    def __init__(self, detail):
        self.detail = detail
        self.fetch_calls = 0

    def fetch_opportunity(self, opp_id):
        self.fetch_calls += 1
        return self.detail


def test_failed_harvest_leaves_store_untouched(tmp_path):
    store = GrantStore(tmp_path / "g")
    store.refresh([Solicitation(opp_id="1"), Solicitation(opp_id="2")])
    before = store.load_active()

    with pytest.raises(GrantsClientError):
        run_refresh(FailingClient(), store)

    assert store.load_active() == before  # no mass-archiving on failure
    assert store.load_archive() == []


def test_load_detail_fetches_once_then_uses_cache(tmp_path):
    store = GrantStore(tmp_path / "g")
    store.refresh([Solicitation(opp_id="42", title="Listing title")])
    detail = {
        "opportunityTitle": "Detail title",
        "docType": "synopsis",
        "ost": "POSTED",
        "synopsis": {"synopsisDesc": "<p>Solicitation text</p>"},
    }
    client = FakeFetchClient(detail)

    first = load_detail(client, store, "42")
    assert first.title == "Detail title"
    assert first.synopsis == "Solicitation text"
    assert first.detail_loaded is True
    assert client.fetch_calls == 1

    second = load_detail(client, store, "42")  # served from the store
    assert second.detail_loaded is True
    assert client.fetch_calls == 1  # no re-fetch


def test_load_detail_rejects_unknown_opportunity(tmp_path):
    store = GrantStore(tmp_path / "g")
    store.refresh([Solicitation(opp_id="1")])
    with pytest.raises(GrantsClientError, match="not in the local database"):
        load_detail(FakeFetchClient({}), store, "999")


def test_apply_detail_is_idempotent_for_cached_records(tmp_path):
    store = GrantStore(tmp_path / "g")
    store.refresh([Solicitation(opp_id="42", title="T")])
    solicitation = load_detail(FakeFetchClient({"opportunityTitle": "T", "synopsis": {}}), store, "42")
    # Re-applying the same cached detail must not corrupt the record.
    apply_detail(solicitation, solicitation.raw)
    assert solicitation.detail_loaded is True
