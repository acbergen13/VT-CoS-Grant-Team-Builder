"""Elsevier production source tests — offline fake transport, temporary cache.

Covers the Stage 6 contract: pagination and normalization, duplicate-EID
merging across author searches, EID/author cache hits with zero API calls,
document-type filtering, bounded abstract fallback, typed errors, and the
factory switch. No live Elsevier requests are made.
"""

import json

import pandas as pd
import pytest
import requests

from gt.faculty.elsevier import (
    ElsevierCache,
    ElsevierClient,
    ElsevierConfigError,
    ElsevierFacultySource,
    ElsevierQuotaError,
    ElsevierResponseError,
    ElsevierUnavailableError,
)
from gt.faculty.source import get_faculty_source

ENV = {
    "ELSEVIER_API_KEY": "test-key",
    "ELSEVIER_BASE_URL": "https://api.elsevier.test",
    "ELSEVIER_INST_TOKEN": "",
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeTransport:
    """Records calls; delegates to a handler that returns payload/response."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        result = self.handler(url, params or {})
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, FakeResponse) else FakeResponse(result)


def elsevier_settings(**overrides):
    elsevier = {
        "timeout_seconds": 1,
        "max_retries": 1,
        "page_size": 2,
        "max_pages_per_author": 10,
        "cache_dir": "",
        "abstract_fallback": True,
        "abstract_fallback_limit": 10,
        "abstract_fallback_disable_after": 2,
    }
    elsevier.update(overrides)
    return {
        "data_source": "elsevier",
        "scoring": {"missing_metric_impute": 0, "weights": {}},
        "team": {"pi_eligible_ranks": ["Professor"]},
        "retrieval": {"exclude_doc_types": ["Erratum"]},
        "elsevier": elsevier,
    }


def write_roster(tmp_path, rows):
    path = tmp_path / "authors.csv"
    pd.DataFrame(
        rows,
        columns=["Author", "Scopus ID", "Rank"],
    ).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def entry(eid, **overrides):
    data = {
        "eid": eid,
        "dc:title": f"Title {eid}",
        "dc:description": f"Abstract {eid}",
        "prism:coverDate": "2024-03-01",
        "prism:publicationName": "Journal of Tests",
        "prism:doi": f"10.1/{eid}",
        "citedby-count": "5",
        "subtypeDescription": "Article",
        "authkeywords": "kw1 | kw2",
        "link": [{"@ref": "scopus", "@href": f"https://scopus.test/{eid}"}],
    }
    data.update(overrides)
    return data


def search_payload(entries, total=None):
    return {
        "search-results": {
            "opensearch:totalResults": str(total if total is not None else len(entries)),
            "entry": entries,
        }
    }


def make_client(transport, settings=None):
    return ElsevierClient(
        transport=transport, settings=settings or elsevier_settings(), env=ENV
    )


# -- Client: config, pagination, normalization ---------------------------------


def test_client_requires_env_keys():
    with pytest.raises(ElsevierConfigError, match="ELSEVIER"):
        ElsevierClient(
            settings=elsevier_settings(),
            env={"ELSEVIER_API_KEY": "", "ELSEVIER_BASE_URL": ""},
        )


def test_search_author_paginates_and_normalizes_entries():
    pages = {
        0: search_payload([entry("E1"), entry("E2")], total=3),
        2: search_payload([entry("E3")], total=3),
    }

    def handler(url, params):
        assert url.endswith("/content/search/scopus")
        assert params["query"] == "AU-ID(1)"
        assert params["view"] == "COMPLETE"
        assert params["count"] == 2
        return pages[params["start"]]

    transport = FakeTransport(handler)
    records = make_client(transport).search_author("1")

    assert [r["eid"] for r in records] == ["E1", "E2", "E3"]
    assert [c["params"]["start"] for c in transport.calls] == [0, 2]
    first = records[0]
    assert first["title"] == "Title E1"
    assert first["abstract"] == "Abstract E1"
    assert first["year"] == 2024
    assert first["doc_type"] == "Article"
    assert first["cited_by"] == 5
    assert first["doi"] == "10.1/E1"
    assert first["link"] == "https://scopus.test/E1"
    assert first["author_keywords"] == "kw1; kw2"
    assert first["cos_author_ids"] == ["1"]
    assert transport.calls[0]["headers"]["X-ELS-APIKey"] == "test-key"


def test_search_author_includes_optional_inst_token():
    transport = FakeTransport(
        lambda url, params: search_payload([entry("E1")], total=1)
    )
    client = ElsevierClient(
        transport=transport,
        settings=elsevier_settings(),
        env={**ENV, "ELSEVIER_INST_TOKEN": "inst-token"},
    )
    client.search_author("1")
    assert transport.calls[0]["headers"]["X-ELS-Insttoken"] == "inst-token"


# -- Cache: duplicate EIDs + zero-call repeat loads ----------------------------


def test_duplicate_eid_merges_authors_and_repeat_load_uses_cache(tmp_path):
    authors = write_roster(
        tmp_path,
        [
            {"Author": "Alpha, Ann", "Scopus ID": "1", "Rank": "Professor"},
            {"Author": "Beta, Bob", "Scopus ID": "2", "Rank": "Professor"},
        ],
    )
    cache_dir = tmp_path / "elsevier_cache"

    def handler(url, params):
        if params["query"] == "AU-ID(1)":
            return search_payload([entry("E1")], total=1)
        if params["query"] == "AU-ID(2)":
            return search_payload(
                [entry("E1", **{"dc:title": "Same paper, richer record"})], total=1
            )
        raise AssertionError(f"unexpected query {params['query']!r}")

    client1 = make_client(FakeTransport(handler))
    source1 = ElsevierFacultySource(
        settings=elsevier_settings(cache_dir=str(cache_dir)),
        roster_path=authors,
        client=client1,
    )
    pubs1 = source1.load_publications()
    assert len(pubs1) == 1
    assert pubs1[0].eid == "E1"
    assert pubs1[0].cos_author_ids == ["1", "2"]

    def no_calls(url, params):
        raise AssertionError("cache hit must not call the Elsevier API")

    client2 = make_client(FakeTransport(no_calls))
    source2 = ElsevierFacultySource(
        settings=elsevier_settings(cache_dir=str(cache_dir)),
        roster_path=authors,
        client=client2,
    )
    pubs2 = source2.load_publications()

    assert client2.calls == 0
    assert source2.stats["author_cache_hits"] == 2
    assert pubs2 == pubs1
    assert json.loads(
        (cache_dir / "publications" / "E1.json").read_text(encoding="utf-8")
    )["record"]["cos_author_ids"] == ["1", "2"]


def test_corrupt_cache_entry_is_a_miss(tmp_path):
    cache = ElsevierCache(tmp_path / "cache")
    path = tmp_path / "cache" / "publications" / "E1.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert cache.get_publication("E1") is None


# -- Source: filtering, fallback, stats ---------------------------------------


def test_source_filters_excluded_document_types(tmp_path):
    authors = write_roster(
        tmp_path, [{"Author": "Alpha, Ann", "Scopus ID": "1", "Rank": "Professor"}]
    )

    def handler(url, params):
        return search_payload(
            [entry("E1"), entry("E2", subtypeDescription="Erratum")], total=2
        )

    source = ElsevierFacultySource(
        settings=elsevier_settings(cache_dir=str(tmp_path / "cache")),
        roster_path=authors,
        client=make_client(FakeTransport(handler)),
    )
    assert [p.eid for p in source.load_publications()] == ["E1"]


def test_missing_abstract_uses_bounded_fallback(tmp_path):
    authors = write_roster(
        tmp_path, [{"Author": "Alpha, Ann", "Scopus ID": "1", "Rank": "Professor"}]
    )
    abstract_calls = []

    def handler(url, params):
        if "/content/search/scopus" in url:
            return search_payload(
                [entry("E1", **{"dc:description": ""})], total=1
            )
        assert "/content/abstract/eid/E1" in url
        assert params == {"view": "META_ABS"}
        abstract_calls.append(url)
        return {
            "abstracts-retrieval-response": {
                "coredata": {"dc:description": "Recovered abstract"}
            }
        }

    source = ElsevierFacultySource(
        settings=elsevier_settings(cache_dir=str(tmp_path / "cache")),
        roster_path=authors,
        client=make_client(FakeTransport(handler)),
    )
    pubs = source.load_publications()
    assert pubs[0].abstract == "Recovered abstract"
    assert len(abstract_calls) == 1


def test_abstract_fallback_disables_after_repeated_entitlement_failures(tmp_path):
    authors = write_roster(
        tmp_path, [{"Author": "Alpha, Ann", "Scopus ID": "1", "Rank": "Professor"}]
    )
    abstract_calls = []

    def handler(url, params):
        if "/content/search/scopus" in url:
            return search_payload(
                [entry("E1", **{"dc:description": ""}),
                 entry("E2", **{"dc:description": ""}),
                 entry("E3", **{"dc:description": ""})],
                total=3,
            )
        abstract_calls.append(url)
        return FakeResponse({}, status_code=403)

    settings = elsevier_settings(
        cache_dir=str(tmp_path / "cache"),
        abstract_fallback_disable_after=1,
    )
    source = ElsevierFacultySource(
        settings=settings,
        roster_path=authors,
        client=make_client(FakeTransport(handler), settings),
    )
    source.load_publications()
    assert len(abstract_calls) == 1  # disabled after the first 403


# -- Client: typed failures (req 23) ------------------------------------------


def test_429_becomes_quota_error():
    transport = FakeTransport(lambda url, params: FakeResponse({}, status_code=429))
    with pytest.raises(ElsevierQuotaError, match="quota"):
        make_client(transport).search_author("1")


def test_403_becomes_quota_entitlement_error():
    transport = FakeTransport(lambda url, params: FakeResponse({}, status_code=403))
    with pytest.raises(ElsevierQuotaError, match="ELSEVIER_API_KEY"):
        make_client(transport).search_author("1")


def test_5xx_becomes_unavailable_error():
    transport = FakeTransport(lambda url, params: FakeResponse({}, status_code=503))
    with pytest.raises(ElsevierUnavailableError, match="not responding"):
        make_client(transport).search_author("1")


def test_bad_json_becomes_response_error():
    transport = FakeTransport(
        lambda url, params: FakeResponse(ValueError("not json"))
    )
    with pytest.raises(ElsevierResponseError, match="unreadable"):
        make_client(transport).search_author("1")


def test_connection_error_becomes_unavailable_error():
    transport = FakeTransport(lambda url, params: requests.ConnectionError("boom"))
    with pytest.raises(ElsevierUnavailableError, match="not responding"):
        make_client(transport).search_author("1")


# -- Factory -------------------------------------------------------------------


def test_factory_returns_elsevier_source(monkeypatch, tmp_path):
    monkeypatch.setenv("ELSEVIER_API_KEY", "test-key")
    monkeypatch.setenv("ELSEVIER_BASE_URL", "https://api.elsevier.test")
    settings = elsevier_settings(cache_dir=str(tmp_path / "cache"))
    source = get_faculty_source(settings)
    assert isinstance(source, ElsevierFacultySource)
