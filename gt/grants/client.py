"""grants.gov REST client: search2 harvest + fetchOpportunity detail.

Endpoints (pinned against the live API on Oct 20, 2026):
- search2: POST https://api.grants.gov/v1/api/search2 with JSON body.
  Response: {"data": {"hitCount": int, "oppHits": [{id, number, title,
  agencyCode, agency, openDate, closeDate, oppStatus, docType, cfdaList}]}}
  Dates are MM/DD/YYYY; closeDate may be "".
- fetchOpportunity: the documented endpoint
  (POST https://api.grants.gov/v1/api/fetchOpportunity) was returning 502
  during development because its internal proxy was down. The backend it
  fronts — POST https://apply07.grants.gov/grantsws/rest/opportunity/details
  with form-encoded oppId — is the same service and works, so this client
  tries the documented endpoint first and falls back to the backend.
  Detail payloads have a "synopsis" block (posted opportunities) or a
  "forecast" block (forecasted ones); see apply_detail().

Neither endpoint requires authentication (req 22). All failures surface as
GrantsClientError with a message safe to show users (req 23).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Callable

import requests
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gt.config import REPO_ROOT, load_settings
from gt.state import Solicitation

SEARCH2_URL = "https://api.grants.gov/v1/api/search2"
FETCH_URL = "https://api.grants.gov/v1/api/fetchOpportunity"
# Backend that the documented fetchOpportunity endpoint proxies to.
FETCH_BACKEND_URL = "https://apply07.grants.gov/grantsws/rest/opportunity/details"

# Hard cap on search2 pages per harvest: a defensive guard against a
# misbehaving API sending us into an infinite pagination loop.
MAX_PAGES = 200


class GrantsClientError(Exception):
    """User-facing failure (req 23): message is safe to render in the UI."""


class _TransientAPIError(Exception):
    """Timeout / connection / 429 / 5xx — safe to retry."""


class _HTMLTextExtractor(HTMLParser):
    """Collects text content only, dropping tags (synopsisDesc is HTML).

    Inline tags (b, i, span...) join directly to surrounding text; block-level
    tags (p, div, br, li...) contribute a word separator.
    """

    BLOCK_TAGS = {
        "p", "div", "br", "li", "ul", "ol", "table", "tr", "td", "th",
        "h1", "h2", "h3", "h4", "h5", "h6", "hr", "blockquote",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _strip_html(text: str) -> str:
    """Convert an HTML fragment to a single line of plain text."""
    if not text:
        return ""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(text)
        parser.close()
        raw = "".join(parser.parts)
    except Exception:
        raw = text  # not valid HTML — fall back to the raw string
    return re.sub(r"\s+", " ", raw).strip()


def _parse_date(value: Any) -> str | None:
    """Normalize API dates to ISO "YYYY-MM-DD" (or None).

    Accepts the two forms grants.gov uses: listing "MM/DD/YYYY" and detail
    "YYYY-MM-DD-HH-MM-SS" (*Str fields). Anything else (including "") -> None.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})", value)
    if iso:
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"
    us = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", value)
    if us:
        return f"{us.group(3)}-{int(us.group(1)):02d}-{int(us.group(2)):02d}"
    return None


def _parse_money(value: Any) -> float | None:
    """Award amounts come as numbers, numeric strings, "none", or ""."""
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "")
        if not cleaned or cleaned.lower() == "none":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _parse_bool(value: Any) -> bool | None:
    """costSharing arrives as the string "True"/"False" or a real bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _join_descriptions(items: list[dict] | None) -> str:
    """Join an API list of {id, description} into "A; B" (order preserved)."""
    descriptions = [i.get("description") for i in items or [] if i.get("description")]
    return "; ".join(dict.fromkeys(descriptions))


def map_listing(hit: dict) -> Solicitation:
    """Map one search2 oppHits record to a Solicitation (listing fields only)."""
    return Solicitation(
        opp_id=str(hit.get("id", "")),
        number=hit.get("number") or "",
        title=hit.get("title") or "",
        status=(hit.get("oppStatus") or "").lower(),
        doc_type=hit.get("docType") or "",
        agency=hit.get("agency") or "",
        agency_code=hit.get("agencyCode") or "",
        open_date=_parse_date(hit.get("openDate")),
        close_date=_parse_date(hit.get("closeDate")),
        cfda=list(hit.get("cfdaList") or []),
        raw=hit,
    )


def apply_detail(solicitation: Solicitation, detail: dict) -> Solicitation:
    """Fill detail fields (req 11) from a fetchOpportunity payload, in place.

    Posted opportunities carry a "synopsis" block; forecasted ones carry a
    "forecast" block with estimated amounts and dates instead of a hard
    close date.
    """
    solicitation.title = detail.get("opportunityTitle") or solicitation.title
    solicitation.number = detail.get("opportunityNumber") or solicitation.number
    solicitation.doc_type = detail.get("docType") or solicitation.doc_type
    if detail.get("ost"):
        solicitation.status = str(detail["ost"]).lower()
    solicitation.category = (detail.get("opportunityCategory") or {}).get("description") or ""
    cfda_numbers = [c.get("cfdaNumber") for c in detail.get("cfdas") or [] if c.get("cfdaNumber")]
    if cfda_numbers:
        solicitation.cfda = [str(c) for c in cfda_numbers]
    solicitation.raw = detail
    solicitation.detail_loaded = True

    synopsis = detail.get("synopsis") or {}
    forecast = detail.get("forecast") or {}
    block = synopsis or forecast
    if synopsis:
        solicitation.synopsis = _strip_html(synopsis.get("synopsisDesc") or "")
        solicitation.eligibility = _strip_html(synopsis.get("applicantEligibilityDesc") or "")
        solicitation.open_date = _parse_date(synopsis.get("postingDateStr")) or solicitation.open_date
        solicitation.close_date = _parse_date(synopsis.get("responseDateStr")) or solicitation.close_date
    elif forecast:
        solicitation.synopsis = _strip_html(forecast.get("forecastDesc") or "")
        # Forecasts list applicant types instead of an eligibility narrative.
        solicitation.eligibility = _join_descriptions(forecast.get("applicantTypes"))
        solicitation.open_date = _parse_date(forecast.get("postingDateStr")) or solicitation.open_date
        solicitation.close_date = None  # no hard close date until the synopsis posts
        solicitation.estimated_funding = _parse_money(forecast.get("estimatedFunding"))
        awards = forecast.get("numberOfAwards")
        solicitation.number_of_awards = awards if isinstance(awards, int) else None

    if block:
        solicitation.award_ceiling = _parse_money(block.get("awardCeiling"))
        solicitation.award_floor = _parse_money(block.get("awardFloor"))
        solicitation.cost_sharing = _parse_bool(block.get("costSharing"))
        instruments = _join_descriptions(block.get("fundingInstruments"))
        if instruments:
            solicitation.funding_instrument = instruments
        categories = _join_descriptions(block.get("fundingActivityCategories"))
        if categories:
            solicitation.funding_categories = categories
        agency_name = block.get("agencyName") or (detail.get("agencyDetails") or {}).get("agencyName")
        if agency_name:
            solicitation.agency = agency_name
    return solicitation


class GrantsClient:
    """Thin grants.gov API client. Inject `transport` for testing."""

    def __init__(
        self,
        transport: Callable[..., Any] | None = None,
        settings: dict | None = None,
    ) -> None:
        grants = (settings or load_settings())["grants"]
        self._timeout = grants["request_timeout_seconds"]
        self._max_retries = grants["max_retries"]
        self._statuses = grants["opp_statuses"]  # req 22: "forecasted|posted"
        self._transport = transport or requests.post

    # -- HTTP plumbing -----------------------------------------------------

    def _request_once(self, url: str, *, json: dict | None = None, data: dict | None = None) -> dict:
        """Single POST attempt. Raises _TransientAPIError for retryable
        failures and GrantsClientError for permanent ones."""
        try:
            response = self._transport(url, json=json, data=data, timeout=self._timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise _TransientAPIError(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise _TransientAPIError(f"HTTP {response.status_code} from {url}")
        if response.status_code >= 400:
            raise GrantsClientError(
                f"grants.gov rejected the request (HTTP {response.status_code})."
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise GrantsClientError("grants.gov returned an unreadable response.") from exc
        if not isinstance(body, dict):
            raise GrantsClientError("grants.gov returned an unexpected response format.")
        return body

    def _request(self, url: str, *, json: dict | None = None, data: dict | None = None) -> dict:
        """POST with retries on transient failures (tenacity), then convert
        to a user-facing error (req 23)."""
        retrying = Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(_TransientAPIError),
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    return self._request_once(url, json=json, data=data)
        except _TransientAPIError as exc:
            raise GrantsClientError(
                f"grants.gov is not responding ({exc}) after {self._max_retries} "
                "attempts. Please try again in a moment."
            ) from exc

    # -- Public API ---------------------------------------------------------

    def search_open(self, rows: int = 100) -> list[dict]:
        """Harvest ALL open opportunities (forecasted|posted, req 22), paging
        through search2. Returns raw oppHits records deduped by id."""
        hits: list[dict] = []
        start = 1
        hit_count: int | None = None
        pages = 0
        while hit_count is None or len(hits) < hit_count:
            if pages >= MAX_PAGES:
                raise GrantsClientError(
                    f"grants.gov pagination exceeded {MAX_PAGES} pages; aborting."
                )
            body = self._request(
                SEARCH2_URL,
                json={
                    "oppStatuses": self._statuses,
                    "startRecordNum": start,
                    "rows": rows,
                },
            )
            data = body.get("data") or {}
            page_hits = data.get("oppHits")
            if hit_count is None:
                hit_count = data.get("hitCount")
                if not isinstance(hit_count, int):
                    raise GrantsClientError("grants.gov search2 response format changed.")
            if not isinstance(page_hits, list):
                raise GrantsClientError("grants.gov search2 response format changed.")
            if not page_hits:
                break  # server returned fewer records than promised — stop safely
            hits.extend(page_hits)
            start += len(page_hits)
            pages += 1
        by_id: dict[str, dict] = {}
        for hit in hits:
            by_id[str(hit.get("id"))] = hit  # dedupe; later pages win
        return [by_id[key] for key in sorted(by_id, key=lambda k: int(k) if k.isdigit() else 0)]

    def harvest(self) -> list[Solicitation]:
        """Full harvest mapped to Solicitation objects (listing fields only)."""
        return [map_listing(hit) for hit in self.search_open()]

    def fetch_opportunity(self, opp_id: str) -> dict:
        """Full detail for one opportunity (req 11).

        Primary: the documented api.grants.gov endpoint, tried once — if its
        proxy is down (it 502'd during development), retrying just delays the
        fallback. Fallback: the apply07 grantsws backend the front endpoint
        proxies to, form-encoded, with the standard retry policy.
        """
        try:
            body = self._request_once(FETCH_URL, json={"oppId": str(opp_id)})
            opportunity = (body.get("data") or {}).get("opportunity")
            if isinstance(opportunity, dict) and opportunity:
                return opportunity
        except (GrantsClientError, _TransientAPIError, requests.RequestException):
            pass  # fall through to the backend
        body = self._request(FETCH_BACKEND_URL, data={"oppId": str(opp_id)})
        if not isinstance(body, dict) or "id" not in body:
            raise GrantsClientError(
                f"Could not load details for opportunity {opp_id} from grants.gov."
            )
        return body


def default_client() -> GrantsClient:
    """Client with production settings (kept out of views for testability)."""
    return GrantsClient()


DATA_DIR = REPO_ROOT / "data"  # re-exported for convenience; store lives under data/grants
