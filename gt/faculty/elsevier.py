"""Elsevier / Scopus production faculty source (reqs 5, 20, 26).

The roster still comes from ``authors.csv`` (req 4). Publication metadata and
abstracts, however, are fetched directly from the Elsevier API instead of
``publications.csv``:

- Scopus Search (``GET /content/search/scopus``) with ``query=AU-ID(<id>)``,
  ``view=COMPLETE``, and 0-based pagination. ``view=COMPLETE`` already carries
  the abstract (``dc:description``), keywords, DOI, EID, document type, and
  Scopus link, so Abstract Retrieval is only a bounded fallback for entries
  whose abstract is missing.
- An EID-keyed disk cache under ``data/elsevier_cache/`` keeps repeat loads
  free: ``publications/<EID>.json`` is the canonical record and
  ``authors/<scopus_id>.json`` stores that author's EID list. A cache hit makes
  zero API calls (req 26).
- Duplicate EIDs returned by more than one author search are merged by
  unioning their roster author IDs and writing the merged record back to the
  cache. The final publication list is sorted by EID for determinism.

Failure handling follows the rest of the app (req 23): missing credentials,
quota/entitlement problems, network/server failures, and malformed responses
each raise a typed, user-presentable exception. Abstract fallback failures are
capped and disabled after repeated 403/quota errors so an off-campus run
cannot turn into thousands of failed requests.

Keep ``data_source: csv`` as the shipped default: the submitted zip must run
without an Elsevier key. This module is the production switch.
"""

from __future__ import annotations

import json
import logging
import os
import re
from html import unescape
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gt.config import REPO_ROOT, load_settings
from gt.faculty.publications import Publication
from gt.faculty.roster import AUTHORS_CSV, Roster, load_roster

logger = logging.getLogger(__name__)

REQUIRED_ENV_KEYS = ("ELSEVIER_API_KEY", "ELSEVIER_BASE_URL")
OPTIONAL_ENV_KEYS = ("ELSEVIER_INST_TOKEN",)
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "elsevier_cache"
CACHE_FORMAT = "v1"

# Scopus Search caps the number of records a single query can page through;
# stop before that with a clear error rather than returning a partial author.
MAX_SEARCH_START = 5000


class ElsevierError(RuntimeError):
    """Base class for user-presentable Elsevier failures."""


class ElsevierConfigError(ElsevierError):
    """Missing or empty .env credentials (reqs 20/23)."""


class ElsevierQuotaError(ElsevierError):
    """Quota exhausted / key rejected / content not entitled (reqs 23/26)."""


class ElsevierUnavailableError(ElsevierError):
    """Endpoint unreachable or returned a retryable server failure (req 23)."""


class ElsevierResponseError(ElsevierError):
    """The API returned malformed or unexpectedly shaped data (req 23)."""


class _TransientElsevierError(Exception):
    """Timeout / connection / 429 / 5xx — safe to retry."""

    def __init__(self, message: str, *, quota: bool = False) -> None:
        super().__init__(message)
        self.quota = quota


def load_elsevier_env() -> dict[str, str]:
    """Load Elsevier credentials from the repo-root .env (never hardcoded)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:  # pragma: no cover - dotenv ships in the course env
        pass
    keys = REQUIRED_ENV_KEYS + OPTIONAL_ENV_KEYS
    return {key: os.environ.get(key, "") for key in keys}


def _as_list(value: Any) -> list:
    """Elsevier JSON uses either a scalar or a list for the same field."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _clean_text(value: Any) -> str:
    """Flatten API HTML/whitespace into one plain-text string."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", unescape(value))
    return re.sub(r"\s+", " ", text).strip()


def _parse_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _parse_year(value: Any) -> int | None:
    match = re.search(r"(\d{4})", str(value or ""))
    return int(match.group(1)) if match else None


def _extract_link(entry: dict) -> str:
    for link in _as_list(entry.get("link")):
        if isinstance(link, dict) and link.get("@ref") == "scopus":
            return str(link.get("@href") or "").strip()
    return str(entry.get("prism:url") or "").strip()


def _extract_keywords(value: Any) -> str:
    """Normalize ``authkeywords`` (``"a | b"`` or keyword objects) to "a; b"."""
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"\s*\|\s*|;", value) if p.strip()]
        return "; ".join(dict.fromkeys(parts))
    parts: list[str] = []
    if isinstance(value, dict):
        value = value.get("author-keyword") or value.get("author-keywords") or []
    for item in _as_list(value):
        if isinstance(item, dict):
            text = item.get("$") or item.get("_") or ""
        else:
            text = item
        text = str(text).strip()
        if text:
            parts.append(text)
    return "; ".join(dict.fromkeys(parts))


def _extract_index_keywords(value: Any) -> str:
    """Normalize Scopus ``idxterms`` when the response includes it."""
    if isinstance(value, str):
        return _extract_keywords(value)
    if isinstance(value, dict):
        value = value.get("mainterm") or value.get("term") or []
    parts: list[str] = []
    for item in _as_list(value):
        text = item.get("$") if isinstance(item, dict) else item
        text = str(text or "").strip()
        if text:
            parts.append(text)
    return "; ".join(dict.fromkeys(parts))


def normalize_entry(entry: dict, scopus_id: str) -> dict | None:
    """One Scopus ``view=COMPLETE`` entry -> the canonical cached record."""
    if not isinstance(entry, dict):
        return None
    eid = str(entry.get("eid") or "").strip()
    if not eid or eid.lower() == "nan":
        return None
    return {
        "eid": eid,
        "title": _clean_text(entry.get("dc:title")),
        "year": _parse_year(
            entry.get("prism:coverDate") or entry.get("prism:coverDisplayDate")
        ),
        "abstract": _clean_text(entry.get("dc:description")),
        "source_title": _clean_text(entry.get("prism:publicationName")),
        "doc_type": _clean_text(
            entry.get("subtypeDescription") or entry.get("prism:aggregationType")
        ),
        "cited_by": _parse_int(entry.get("citedby-count")),
        "doi": _clean_text(entry.get("prism:doi")),
        "link": _extract_link(entry),
        "author_keywords": _extract_keywords(entry.get("authkeywords")),
        "index_keywords": _extract_index_keywords(entry.get("idxterms")),
        "cos_author_ids": [str(scopus_id)],
    }


def _extract_abstract(body: dict) -> str:
    """Pull an abstract from the Abstract Retrieval response shapes."""
    root = body.get("abstracts-retrieval-response") or body
    coredata = root.get("coredata") or {}
    abstract = coredata.get("dc:description")
    if abstract:
        return _clean_text(abstract)

    item = root.get("item") or {}
    head = (item.get("bibrecord") or {}).get("head") or {}
    abstracts = head.get("abstracts") or {}
    if isinstance(abstracts, dict):
        abstract = abstracts.get("abstract") or abstracts.get("abstracts")
    elif isinstance(abstracts, str):
        abstract = abstracts
    if isinstance(abstract, dict):
        abstract = abstract.get("$") or abstract.get("p")
    if isinstance(abstract, list):
        abstract = " ".join(str(a) for a in abstract)
    return _clean_text(abstract)


class ElsevierClient:
    """Scopus Search + Abstract Retrieval with retries and typed errors.

    ``transport`` is injectable for offline tests (same pattern as
    ``GrantsClient``). It must accept ``url``, ``params``, ``headers``, and
    ``timeout`` keyword arguments and return an object with ``status_code`` and
    ``json()``.
    """

    def __init__(
        self,
        transport: Callable[..., Any] | None = None,
        settings: dict | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        cfg = (settings or load_settings()).get("elsevier", {})
        env = env if env is not None else load_elsevier_env()
        missing = [key for key in REQUIRED_ENV_KEYS if not (env.get(key) or "").strip()]
        if missing:
            raise ElsevierConfigError(
                f"Missing .env key(s): {', '.join(missing)}. Production mode "
                "requires ELSEVIER_API_KEY and ELSEVIER_BASE_URL (copy "
                ".env.example to .env and fill them in)."
            )
        self._api_key = env["ELSEVIER_API_KEY"].strip()
        self._base_url = env["ELSEVIER_BASE_URL"].strip().rstrip("/")
        self._inst_token = (env.get("ELSEVIER_INST_TOKEN") or "").strip()
        self._timeout = float(cfg.get("timeout_seconds", 30))
        self._max_retries = int(cfg.get("max_retries", 3))
        self._page_size = int(cfg.get("page_size", 25))
        self._max_pages = int(cfg.get("max_pages_per_author", 100))
        self._transport = transport or requests.get
        self.calls = 0  # request counter for diagnostics/tests (req 21)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> dict[str, str]:
        headers = {
            "X-ELS-APIKey": self._api_key,
            "Accept": "application/json",
        }
        if self._inst_token:
            headers["X-ELS-Insttoken"] = self._inst_token
        return headers

    def _request_once(self, url: str, params: dict) -> dict:
        self.calls += 1
        try:
            response = self._transport(
                url, params=params, headers=self._headers(), timeout=self._timeout
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise _TransientElsevierError(f"{type(exc).__name__}: {exc}") from exc
        except requests.RequestException as exc:
            raise _TransientElsevierError(f"{type(exc).__name__}: {exc}") from exc

        status = response.status_code
        if status in (401, 403):
            raise ElsevierQuotaError(
                "Elsevier rejected the request "
                f"(HTTP {status}). Check ELSEVIER_API_KEY/ELSEVIER_INST_TOKEN "
                "and whether your key is entitled to this content."
            )
        if status == 429 or status >= 500:
            raise _TransientElsevierError(
                f"HTTP {status} from {url}", quota=(status == 429)
            )
        if status >= 400:
            raise ElsevierResponseError(
                f"Elsevier returned HTTP {status} for {url}."
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ElsevierResponseError(
                "Elsevier returned an unreadable response."
            ) from exc
        if not isinstance(body, dict):
            raise ElsevierResponseError("Elsevier returned an unexpected response format.")
        if body.get("error-response"):
            raise ElsevierResponseError(
                f"Elsevier API error: {body.get('error-response')}"
            )
        return body

    def _request(self, url: str, params: dict) -> dict:
        retrying = Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(_TransientElsevierError),
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    return self._request_once(url, params)
        except _TransientElsevierError as exc:
            if exc.quota:
                raise ElsevierQuotaError(
                    "Elsevier quota is exhausted or rate-limited after "
                    f"{self._max_retries} attempts. Try again later, or use "
                    "the local CSV mode (`data_source: csv`)."
                ) from exc
            raise ElsevierUnavailableError(
                f"Elsevier is not responding ({exc}) after {self._max_retries} "
                "attempts. Check your network and try again."
            ) from exc

    def search_author(self, scopus_id: str) -> list[dict]:
        """All publications for one roster author, normalized and sorted."""
        url = f"{self._base_url}/content/search/scopus"
        records: dict[str, dict] = {}
        start = 0
        pages = 0
        total: int | None = None
        while True:
            if pages >= self._max_pages:
                raise ElsevierResponseError(
                    f"Elsevier pagination for author {scopus_id} exceeded "
                    f"{self._max_pages} pages; aborting."
                )
            body = self._request(
                url,
                {
                    "query": f"AU-ID({scopus_id})",
                    "view": "COMPLETE",
                    "count": self._page_size,
                    "start": start,
                },
            )
            results = body.get("search-results")
            if not isinstance(results, dict):
                raise ElsevierResponseError(
                    "Elsevier search response is missing 'search-results'."
                )
            entries = _as_list(results.get("entry"))
            if total is None:
                total = _parse_int(results.get("opensearch:totalResults"))
            if not entries:
                break
            for entry in entries:
                record = normalize_entry(entry, scopus_id)
                if record is not None:
                    records.setdefault(record["eid"], record)
            start += len(entries)
            pages += 1
            if total is not None and start >= total:
                break
            if len(entries) < self._page_size:
                break
            if start >= MAX_SEARCH_START:
                raise ElsevierResponseError(
                    f"Elsevier search for author {scopus_id} reached the "
                    "Scopus pagination limit; narrow the query."
                )
        return [records[eid] for eid in sorted(records)]

    def fetch_abstract(self, eid: str) -> str:
        """Abstract Retrieval fallback for a missing ``dc:description``."""
        url = f"{self._base_url}/content/abstract/eid/{quote(eid, safe='')}"
        return _extract_abstract(self._request(url, {"view": "META_ABS"}))


class ElsevierCache:
    """EID-keyed disk cache for publication records and author EID lists."""

    def __init__(self, cache_dir: Path | str | None = None, refresh: bool = False) -> None:
        self._dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._publications = self._dir / "publications"
        self._authors = self._dir / "authors"
        self.refresh = refresh

    def _read(self, path: Path) -> dict | None:
        if self.refresh or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("ignoring corrupt Elsevier cache entry %s", path)
            return None
        return payload if isinstance(payload, dict) else None

    def _write(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def get_publication(self, eid: str) -> dict | None:
        payload = self._read(self._publications / f"{eid}.json")
        if not payload or payload.get("format") != CACHE_FORMAT:
            return None
        record = payload.get("record")
        return record if isinstance(record, dict) else None

    def put_publication(self, eid: str, record: dict) -> None:
        self._write(
            self._publications / f"{eid}.json",
            {"format": CACHE_FORMAT, "record": record},
        )

    def get_author(self, scopus_id: str) -> list[str] | None:
        payload = self._read(self._authors / f"{scopus_id}.json")
        if not payload or payload.get("format") != CACHE_FORMAT:
            return None
        eids = payload.get("eids")
        if not isinstance(eids, list):
            return None
        return [str(eid) for eid in eids]

    def put_author(self, scopus_id: str, eids: list[str]) -> None:
        self._write(
            self._authors / f"{scopus_id}.json",
            {"format": CACHE_FORMAT, "eids": sorted(set(eids))},
        )


class ElsevierFacultySource:
    """Production ``FacultySource``: roster CSV + Scopus publications + cache."""

    def __init__(
        self,
        settings: dict | None = None,
        roster_path: Path | str | None = None,
        client: ElsevierClient | None = None,
        cache: ElsevierCache | None = None,
        refresh: bool = False,
    ) -> None:
        self._settings = settings or load_settings()
        self._roster_path = Path(roster_path) if roster_path else AUTHORS_CSV
        cfg = self._settings.get("elsevier", {})
        self._client = client or ElsevierClient(settings=self._settings)
        cache_dir = Path(cfg.get("cache_dir") or DEFAULT_CACHE_DIR)
        if not cache_dir.is_absolute():
            cache_dir = REPO_ROOT / cache_dir
        self._cache = cache or ElsevierCache(cache_dir, refresh=refresh)
        self._excluded_doc_types = set(
            self._settings["retrieval"]["exclude_doc_types"]
        )
        self._fallback_enabled = bool(cfg.get("abstract_fallback", True))
        self._fallback_limit = int(cfg.get("abstract_fallback_limit", 25))
        self._fallback_disable_after = int(
            cfg.get("abstract_fallback_disable_after", 3)
        )
        self._fallback_attempts = 0
        self._fallback_failures = 0
        self._fallback_disabled = False
        self.stats = {
            "author_cache_hits": 0,
            "author_cache_misses": 0,
            "publication_cache_hits": 0,
            "abstract_fallbacks": 0,
            "abstract_fallback_failures": 0,
        }

    def load_roster(self) -> Roster:
        return load_roster(self._roster_path, self._settings)

    def _store_publication(self, record: dict) -> None:
        """Persist a publication record, merging cached roster authors."""
        eid = record["eid"]
        existing = self._cache.get_publication(eid)
        if existing:
            record = {
                **existing,
                **record,
                "cos_author_ids": sorted(
                    set(existing.get("cos_author_ids") or [])
                    | set(record.get("cos_author_ids") or [])
                ),
            }
        self._cache.put_publication(eid, record)

    def _records_for_author(self, scopus_id: str) -> list[dict]:
        eids = self._cache.get_author(scopus_id)
        if eids is not None:
            records: list[dict] = []
            missing = False
            for eid in eids:
                record = self._cache.get_publication(eid)
                if record is None:
                    missing = True
                    break
                records.append(record)
                self.stats["publication_cache_hits"] += 1
            if not missing:
                self.stats["author_cache_hits"] += 1
                return records
            logger.warning(
                "author %s cache points at missing publications; refetching",
                scopus_id,
            )
        else:
            self.stats["author_cache_misses"] += 1

        records = self._client.search_author(scopus_id)
        for record in records:
            self._store_publication(record)
        self._cache.put_author(scopus_id, [record["eid"] for record in records])
        return records

    def _fill_missing_abstracts(self, records: dict[str, dict]) -> None:
        if not self._fallback_enabled or self._fallback_disabled:
            return
        for eid in sorted(records):
            if self._fallback_attempts >= self._fallback_limit:
                logger.info(
                    "abstract fallback limit reached (%s); leaving remaining "
                    "publications title/keyword-only",
                    self._fallback_limit,
                )
                return
            record = records[eid]
            if record.get("abstract"):
                continue
            self._fallback_attempts += 1
            self.stats["abstract_fallbacks"] += 1
            try:
                abstract = self._client.fetch_abstract(eid)
            except ElsevierQuotaError as exc:
                self._fallback_failures += 1
                self.stats["abstract_fallback_failures"] += 1
                logger.warning("abstract fallback failed for %s: %s", eid, exc)
                if self._fallback_failures >= self._fallback_disable_after:
                    self._fallback_disabled = True
                    logger.warning(
                        "disabling Elsevier abstract fallback after %s "
                        "entitlement/quota failures",
                        self._fallback_failures,
                    )
                    return
                continue
            except ElsevierError as exc:
                self._fallback_failures += 1
                self.stats["abstract_fallback_failures"] += 1
                logger.warning("abstract fallback failed for %s: %s", eid, exc)
                continue
            if abstract:
                record["abstract"] = abstract
                self._cache.put_publication(eid, record)

    def load_publications(self) -> list[Publication]:
        """Scopus publications for the roster, cached and deterministically ordered."""
        roster = self.load_roster()
        by_eid: dict[str, dict] = {}
        for member in roster:
            for record in self._records_for_author(member.scopus_id):
                eid = record["eid"]
                if eid in by_eid:
                    by_eid[eid]["cos_author_ids"] = sorted(
                        set(by_eid[eid].get("cos_author_ids") or [])
                        | set(record.get("cos_author_ids") or [])
                    )
                else:
                    by_eid[eid] = dict(record)

        # Persist the merged author lists so a later all-cache load keeps them.
        for record in by_eid.values():
            self._cache.put_publication(record["eid"], record)

        eligible = {
            eid: record
            for eid, record in by_eid.items()
            if record.get("doc_type") not in self._excluded_doc_types
        }
        self._fill_missing_abstracts(eligible)

        publications: list[Publication] = []
        for eid in sorted(eligible):
            record = eligible[eid]
            cos_ids = sorted(set(record.get("cos_author_ids") or []) & set(roster.by_id))
            if not cos_ids:
                continue
            publications.append(
                Publication(
                    eid=eid,
                    title=record.get("title", ""),
                    year=record.get("year"),
                    abstract=record.get("abstract", ""),
                    source_title=record.get("source_title", ""),
                    doc_type=record.get("doc_type", ""),
                    cited_by=int(record.get("cited_by", 0) or 0),
                    doi=record.get("doi", ""),
                    link=record.get("link", ""),
                    author_keywords=record.get("author_keywords", ""),
                    index_keywords=record.get("index_keywords", ""),
                    cos_author_ids=cos_ids,
                )
            )

        logger.info(
            "Elsevier load: %s publications (%s author cache hits, %s misses, "
            "%s abstract fallbacks)",
            len(publications),
            self.stats["author_cache_hits"],
            self.stats["author_cache_misses"],
            self.stats["abstract_fallbacks"],
        )
        return publications
