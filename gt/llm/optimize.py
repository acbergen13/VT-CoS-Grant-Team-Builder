"""Req 14 — LLM-based prompt optimization before retrieval.

The LLM extracts domain/methodology keywords from the structured
solicitation; the retrieval query itself is ASSEMBLED IN CODE
(canonical_query): a deterministic anchor built from the solicitation's own
fields plus the LLM's keywords, sorted and lowercased. This is the ONLY
place the LLM touches retrieval input, and the canonical form bounds its
drift (mitigation 4): a regenerated plan with the same keyword set yields a
bit-identical query — phrasing or ordering variance cannot reshape the
candidate pool. Measured Oct 21 (gpt-oss-120b): the model's free-text
"query" field was just a re-serialization of the synopsis and keyword lists,
differing only at the truncation tail, so it was dropped from the schema.

If the LLM is unavailable or returns junk, a deterministic fallback query
is built from solicitation fields — retrieval never depends on LLM
availability (req 23), and the degradation is recorded, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gt.config import load_settings
from gt.llm.arc import ArcClient
from gt.llm.prompts import optimize_prompt, parse_json_object
from gt.state import Solicitation


@dataclass
class OptimizedQuery:
    """Validated retrieval input for one team build."""

    query: str  # canonical query (anchor + sorted keywords), code-assembled
    domains: list[str] = field(default_factory=list)
    methodology: list[str] = field(default_factory=list)
    source: str = "llm"  # "llm" | "fallback"
    note: str = ""  # user-presentable degradation note (req 23)


def _clean_list(values, limit: int) -> list[str]:
    out: list[str] = []
    for v in values if isinstance(values, list) else []:
        s = " ".join(str(v).split()).strip(" .;-")
        if s and s not in out:
            out.append(s[:80])
        if len(out) >= limit:
            break
    return out


def fallback_query(solicitation: Solicitation, settings: dict | None = None) -> str:
    """Deterministic, code-built query from solicitation fields (no LLM)."""
    cfg = (settings or load_settings())["llm"]
    pieces = [solicitation.title, solicitation.agency, solicitation.category]
    synopsis = " ".join((solicitation.synopsis or "").split())
    if synopsis:
        pieces.append(" ".join(synopsis.split()[:40]))
    q = ", ".join(p for p in pieces if p)
    return q[: int(cfg["query_max_chars"])].strip()


def canonical_query(
    solicitation: Solicitation,
    domains: list[str],
    methodology: list[str],
    settings: dict | None = None,
) -> str:
    """Deterministic retrieval query: anchor + sorted keywords.

    The anchor (fallback_query) is pure solicitation fields; the keywords
    are the LLM's contribution, sorted and lowercased so only the keyword
    SET matters — not the order or phrasing the model happened to use. Each
    half is bounded by llm.query_max_chars, keeping total input bounded
    (mitigation 4). Same anchor + same keyword set -> identical string ->
    identical pool (req 19).
    """
    cfg = settings or load_settings()
    max_chars = int(cfg["llm"]["query_max_chars"])
    anchor = fallback_query(solicitation, cfg)
    terms = " ".join(
        sorted({t.lower() for t in list(domains) + list(methodology) if t})
    )[:max_chars]
    return f"{anchor} {terms}".strip()


def optimize_query(
    client: ArcClient,
    solicitation: Solicitation,
    settings: dict | None = None,
) -> OptimizedQuery:
    """LLM keyword extraction with a deterministic safety net. The reply's
    keyword lists are cleaned and sorted (stable input for the role
    decomposition prompt and the cache), and the retrieval query is built by
    canonical_query — a stray free-text "query" field in the reply is
    ignored."""
    cfg = settings or load_settings()
    try:
        reply = client.complete([{"role": "user", "content": optimize_prompt(solicitation, cfg)}])
        data = parse_json_object(reply)
        domains = sorted(_clean_list(data.get("domains"), 8))
        methodology = sorted(_clean_list(data.get("methodology"), 8))
        if not domains and not methodology:
            raise ValueError("no keywords extracted")
        return OptimizedQuery(
            query=canonical_query(solicitation, domains, methodology, cfg),
            domains=domains,
            methodology=methodology,
            source="llm",
        )
    except Exception as exc:  # Arc* errors or unparseable JSON -> degrade
        return OptimizedQuery(
            query=fallback_query(solicitation, cfg),
            source="fallback",
            note=(
                "LLM query optimization did not produce usable keywords ("
                f"{type(exc).__name__}); used a deterministic query built "
                "from the solicitation fields instead."
            ),
        )
