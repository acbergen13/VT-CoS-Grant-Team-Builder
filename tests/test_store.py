"""GrantStore refresh semantics: upsert, closed->archive, reopen, atomicity."""

import json

from gt.grants.store import GrantStore
from gt.state import Solicitation


def make_store(tmp_path):
    return GrantStore(tmp_path / "grants")


def make_solicitation(opp_id, title="T", status="posted"):
    return Solicitation(opp_id=opp_id, title=title, status=status)


def test_first_refresh_adds_everything(tmp_path):
    store = make_store(tmp_path)
    result = store.refresh([make_solicitation("1"), make_solicitation("2")])
    assert result.newly_added == ["1", "2"]
    assert result.newly_closed == []
    assert result.active_now == 2
    assert result.archived_total == 0
    assert [o["opp_id"] for o in store.load_active()] == ["1", "2"]


def test_refresh_archives_closed_and_hides_them(tmp_path):
    store = make_store(tmp_path)
    store.refresh([make_solicitation("101"), make_solicitation("102"), make_solicitation("103")])

    # 102 no longer open on the next harvest -> archived, never shown.
    result = store.refresh([make_solicitation("101"), make_solicitation("103")])
    assert result.newly_closed == ["102"]
    active_ids = [o["opp_id"] for o in store.load_active()]
    assert active_ids == ["101", "103"]
    archived = store.load_archive()
    assert [o["opp_id"] for o in archived] == ["102"]
    assert archived[0]["archived_at"]  # timestamped for the record

    # A later refresh must not resurrect it (req 3: closed never shown).
    store.refresh([make_solicitation("101")])
    assert "102" not in [o["opp_id"] for o in store.load_active()]
    assert "102" in [o["opp_id"] for o in store.load_archive()]


def test_reopened_opportunity_returns_to_active(tmp_path):
    store = make_store(tmp_path)
    store.refresh([make_solicitation("1"), make_solicitation("2")])
    store.refresh([make_solicitation("1")])  # 2 closes
    result = store.refresh([make_solicitation("1"), make_solicitation("2")])  # 2 reopens

    assert result.reopened == ["2"]
    active_ids = [o["opp_id"] for o in store.load_active()]
    assert active_ids == ["1", "2"]
    assert store.load_archive() == []  # active and archive stay disjoint


def test_refresh_upserts_changed_records(tmp_path):
    store = make_store(tmp_path)
    store.refresh([make_solicitation("1", title="Old title")])
    result = store.refresh([make_solicitation("1", title="New title")])
    assert result.updated == ["1"]
    assert result.newly_added == []
    assert store.load_active()[0]["title"] == "New title"


def test_harvested_at_metadata_round_trips(tmp_path):
    store = make_store(tmp_path)
    store.refresh([make_solicitation("1")])
    first = store.harvested_at()
    assert first is not None
    store.refresh([make_solicitation("1")])
    assert store.harvested_at() >= first  # monotonic timestamp


def test_save_detail_persists_and_preserves_metadata(tmp_path):
    store = make_store(tmp_path)
    store.refresh([make_solicitation("1", title="Original")])
    harvested_at = store.harvested_at()

    solicitation = Solicitation.from_dict(store.load_active()[0])
    solicitation.synopsis = "full detail text"
    solicitation.detail_loaded = True
    store.save_detail(solicitation)

    record = store.load_active()[0]
    assert record["detail_loaded"] is True
    assert record["synopsis"] == "full detail text"
    assert store.harvested_at() == harvested_at  # refresh metadata untouched


def test_save_detail_ignores_unknown_opportunities(tmp_path):
    store = make_store(tmp_path)
    store.refresh([make_solicitation("1")])
    store.save_detail(make_solicitation("999"))
    assert len(store.load_active()) == 1


def test_writes_are_atomic_and_valid_json(tmp_path):
    store = make_store(tmp_path)
    store.refresh([make_solicitation("1")])
    files = list((tmp_path / "grants").iterdir())
    assert not [f for f in files if f.suffix == ".tmp"]  # no temp leftovers
    for path in files:
        with open(path, encoding="utf-8") as f:
            json.load(f)  # round-trips cleanly
