"""GrantsClient tests — no network: a fake transport replays pinned API shapes."""

import pytest
import requests

from gt.grants.client import (
    FETCH_BACKEND_URL,
    FETCH_URL,
    SEARCH2_URL,
    GrantsClient,
    GrantsClientError,
    _parse_date,
    _parse_money,
    apply_detail,
    map_listing,
)
from gt.state import Solicitation

SETTINGS = {
    "grants": {
        "request_timeout_seconds": 1,
        "max_retries": 3,
        "opp_statuses": "forecasted|posted",
    }
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class FakeTransport:
    """Replays canned routes; records every call for assertions."""

    def __init__(self, routes=None):
        self.routes = routes or {}  # url -> payload | FakeResponse | Exception
        self.calls = []

    def __call__(self, url, json=None, data=None, timeout=None):
        self.calls.append({"url": url, "json": json, "data": data})
        route = self.routes[url]
        if isinstance(route, Exception):
            raise route
        if isinstance(route, FakeResponse):
            return route
        return FakeResponse(route)


def make_client(transport, settings=None):
    return GrantsClient(transport=transport, settings=settings or SETTINGS)


def search2_page(hit_count, hits):
    return {"errorcode": 0, "data": {"hitCount": hit_count, "oppHits": hits}}


# -- Mapping: search2 listing -------------------------------------------------


def test_map_listing_pins_fields_and_dates():
    hit = {
        "id": 357305,
        "number": "PAR-25-274",
        "title": "Feasibility Clinical Trials",
        "agencyCode": "HHS-NIH11",
        "agency": "National Institutes of Health",
        "openDate": "11/21/2024",
        "closeDate": "",
        "oppStatus": "posted",
        "docType": "synopsis",
        "cfdaList": ["93.213"],
    }
    s = map_listing(hit)
    assert s.opp_id == "357305"
    assert s.number == "PAR-25-274"
    assert s.status == "posted"
    assert s.doc_type == "synopsis"
    assert s.open_date == "2024-11-21"
    assert s.close_date is None  # empty closeDate string -> None
    assert s.cfda == ["93.213"]
    assert s.detail_loaded is False


# -- Mapping: fetchOpportunity detail -----------------------------------------


def test_apply_detail_synopsis_block():
    s = Solicitation(opp_id="1", title="old", close_date="2099-01-01")
    detail = {
        "opportunityTitle": "New Title",
        "opportunityNumber": "PAR-25-274",
        "docType": "synopsis",
        "ost": "POSTED",
        "opportunityCategory": {"description": "Discretionary"},
        "cfdas": [{"cfdaNumber": "93.213"}],
        "synopsis": {
            "synopsisDesc": "<p>Hello <b>world</b>.&nbsp;Bye</p>",
            "applicantEligibilityDesc": "Public and State controlled institutions",
            "awardCeiling": "none",
            "awardFloor": "none",
            "costSharing": "False",
            "fundingInstruments": [{"id": "G", "description": "Grant"}],
            "fundingActivityCategories": [{"id": "HL", "description": "Health"}],
            "responseDateStr": "2026-11-17-00-00-00",
            "postingDateStr": "2024-11-21-00-00-00",
            "agencyName": "National Institutes of Health",
        },
    }
    apply_detail(s, detail)
    assert s.title == "New Title"
    assert s.status == "posted"
    assert s.synopsis == "Hello world. Bye"  # HTML stripped, whitespace collapsed
    assert s.eligibility.startswith("Public and State controlled")
    assert s.award_ceiling is None  # "none" -> None
    assert s.cost_sharing is False  # string "False" -> bool
    assert s.funding_instrument == "Grant"
    assert s.funding_categories == "Health"
    assert s.close_date == "2026-11-17"
    assert s.category == "Discretionary"
    assert s.detail_loaded is True


def test_apply_detail_forecast_block():
    s = Solicitation(opp_id="2", close_date="2099-01-01")
    detail = {
        "opportunityTitle": "Forecasted Thing",
        "docType": "forecast",
        "ost": "FORECASTED",
        "forecast": {
            "forecastDesc": "Anticipated availability of funds",
            "awardCeiling": 700000,
            "awardFloor": 600000,
            "estimatedFunding": 700000,
            "numberOfAwards": 1,
            "applicantTypes": [{"id": "01", "description": "County governments"}],
            "fundingInstruments": [{"id": "G", "description": "Grant"}],
            "postingDateStr": "2024-08-01-00-00-00",
        },
    }
    apply_detail(s, detail)
    assert s.status == "forecasted"
    assert s.synopsis == "Anticipated availability of funds"
    assert s.award_ceiling == 700000.0
    assert s.award_floor == 600000.0
    assert s.estimated_funding == 700000.0
    assert s.number_of_awards == 1
    assert s.eligibility == "County governments"
    assert s.close_date is None  # forecasts have no hard close date
    assert s.open_date == "2024-08-01"


# -- Parsers -------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("11/21/2024", "2024-11-21"),
        ("1/5/2025", "2025-01-05"),
        ("2026-11-17-00-00-00", "2026-11-17"),
        ("", None),
        (None, None),
        ("Nov 17, 2026 12:00:00 AM EST", None),  # verbose form is not parsed
    ],
)
def test_parse_date(value, expected):
    assert _parse_date(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("none", None),
        ("", None),
        (700000, 700000.0),
        ("700000", 700000.0),
        ("$700,000", 700000.0),
        (True, None),  # bool guard
    ],
)
def test_parse_money(value, expected):
    assert _parse_money(value) == expected


# -- Pagination ------------------------------------------------------------------


def test_search_open_pages_until_hit_count_and_dedupes():
    pages = {
        1: search2_page(4, [{"id": "1", "title": "A-old"}, {"id": "2", "title": "B"}]),
        3: search2_page(4, [{"id": "1", "title": "A-new"}, {"id": "3", "title": "C"}]),
    }

    def transport(url, json=None, data=None, timeout=None):
        assert url == SEARCH2_URL
        assert json["oppStatuses"] == "forecasted|posted"  # req 22
        return FakeResponse(pages[json["startRecordNum"]])

    client = make_client(transport)
    hits = client.search_open()
    assert [h["id"] for h in hits] == ["1", "2", "3"]  # sorted, deduped
    assert hits[0]["title"] == "A-new"  # later page wins on conflict


def test_search_open_stops_on_empty_page():
    pages = {
        1: search2_page(10, [{"id": "1"}, {"id": "2"}]),
        3: search2_page(10, []),  # server short-changes us — must not loop
    }

    def transport(url, json=None, data=None, timeout=None):
        return FakeResponse(pages[json["startRecordNum"]])

    hits = make_client(transport).search_open()
    assert len(hits) == 2


# -- fetchOpportunity primary/fallback ---------------------------------------------


def test_fetch_opportunity_uses_documented_endpoint_when_healthy():
    opportunity = {"id": 7, "opportunityTitle": "From documented endpoint"}
    transport = FakeTransport({FETCH_URL: {"data": {"opportunity": opportunity}}})
    assert make_client(transport).fetch_opportunity("7") == opportunity
    assert [c["url"] for c in transport.calls] == [FETCH_URL]


def test_fetch_opportunity_falls_back_to_apply07_backend():
    opportunity = {"id": 7, "opportunityTitle": "From backend"}
    transport = FakeTransport(
        {
            FETCH_URL: FakeResponse({}, status_code=502),  # documented endpoint down
            FETCH_BACKEND_URL: opportunity,
        }
    )
    assert make_client(transport).fetch_opportunity("7") == opportunity
    urls = [c["url"] for c in transport.calls]
    assert urls == [FETCH_URL, FETCH_BACKEND_URL]
    assert transport.calls[1]["data"] == {"oppId": "7"}  # form-encoded fallback


def test_fetch_opportunity_raises_when_both_paths_fail():
    transport = FakeTransport(
        {
            FETCH_URL: FakeResponse({}, status_code=502),
            FETCH_BACKEND_URL: FakeResponse({}, status_code=502),
        }
    )
    with pytest.raises(GrantsClientError):
        make_client(transport).fetch_opportunity("7")


# -- Error wrapping (req 23) ---------------------------------------------------------


def test_transient_failures_become_user_facing_errors():
    single_try = {"grants": {"request_timeout_seconds": 1, "max_retries": 1, "opp_statuses": "forecasted|posted"}}
    transport = FakeTransport({SEARCH2_URL: requests.ConnectionError("boom")})
    with pytest.raises(GrantsClientError, match="not responding"):
        make_client(transport, single_try).search_open()


def test_permanent_http_error_is_not_retried():
    single_try = {"grants": {"request_timeout_seconds": 1, "max_retries": 3, "opp_statuses": "forecasted|posted"}}
    transport = FakeTransport({SEARCH2_URL: FakeResponse({}, status_code=404)})
    with pytest.raises(GrantsClientError, match="rejected the request"):
        make_client(transport, single_try).search_open()
    assert len(transport.calls) == 1  # 4xx fails fast — no retry loop
