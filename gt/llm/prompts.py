"""Prompt templates for the LLM team-builder — assembled in code from the
structured Solicitation object (mitigation 4: the model fills slots, it never
rewrites the prompt, so drift is bounded by construction).

Token discipline (req 21): every text piece passed to the LLM is bounded
(max_input_chars), and the per-role fill prompt sees ONLY that role plus the
remaining candidates — not the conversation, not the whole corpus.

The model is asked for JSON in every template; parse_json_object tolerates
fences and surrounding prose. Anything the model must not invent (role names,
candidate IDs, EIDs) is restated in the prompt AND re-checked in verify.py.
"""

from __future__ import annotations

import json
from typing import Sequence

from gt.config import load_settings
from gt.faculty.scoring import ScoredCandidate
from gt.retrieval.evidence import EvidenceChunk
from gt.state import Solicitation

_JSON_RULE = (
    "Respond with ONLY a JSON object — no markdown fences, no prose before "
    "or after. Use double quotes for all keys and string values."
)


def parse_json_object(text: str) -> dict:
    """Extract the first balanced JSON object from an LLM reply.

    Tolerates ```json fences and surrounding prose. Raises ValueError with
    an excerpt when nothing parseable is found (callers treat that as a
    regeneration/fallback trigger, never a crash).
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.strip()
    start = s.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])
    raise ValueError(f"unbalanced JSON object in reply: {text[:120]!r}")


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and clip to a character budget."""
    s = " ".join((text or "").split())
    return s[:limit] + ("…" if len(s) > limit else "")


def solicitation_brief(solicitation: Solicitation, settings: dict | None = None) -> str:
    """Bounded text summary of the structured solicitation (shared by the
    optimize and role-decompose prompts). Fields, not free text."""
    cfg = (settings or load_settings())["llm"]
    limit = int(cfg["max_input_chars"])
    sol = solicitation
    parts = [
        f"Title: {sol.title or '(untitled)'}",
        f"Agency: {sol.agency or 'unknown'}",
        f"Funding instrument: {sol.funding_instrument or 'n/a'}",
        f"Award ceiling: {sol.award_ceiling if sol.award_ceiling is not None else 'n/a'}",
        f"Open/close dates: {sol.open_date or '?'} to {sol.close_date or 'open'}",
    ]
    if sol.synopsis:
        parts.append(f"Synopsis: {_clip(sol.synopsis, limit // 2)}")
    if sol.eligibility:
        parts.append(f"Eligibility: {_clip(sol.eligibility, limit // 6)}")
    budget = limit - sum(len(p) for p in parts) - len(parts)
    if sol.category:
        parts.append(f"Category: {sol.category}")
    return "\n".join(parts)[: max(limit, budget)]


def optimize_prompt(solicitation: Solicitation, settings: dict | None = None) -> str:
    """Req 14: extract retrieval keywords from the structured solicitation.
    The final query is assembled in code (optimize.canonical_query) from the
    solicitation's own fields plus these keywords — the model never writes
    the query string itself, so a regenerated reply with the same keywords
    cannot drift the retrieval input (mitigation 4)."""
    return (
        "You are preparing the search input for a literature search over "
        "university faculty publications, for a federal grant solicitation. "
        "Below is a structured summary of the solicitation.\n\n"
        f"{solicitation_brief(solicitation, settings)}\n\n"
        "List the technical domains and the required methods or techniques "
        "that matching faculty publications should cover. Concrete "
        "technical terms only — no person names, no filler words. At most "
        "8 entries per list.\n"
        f"{_JSON_RULE} Schema:\n"
        '{"domains": ["<technical domain>", ...], '
        '"methodology": ["<required method or technique>", ...]}'
    )


def role_decompose_prompt(
    solicitation: Solicitation,
    fixed_roles: Sequence[str],
    domains: Sequence[str],
    methodology: Sequence[str],
    settings: dict | None = None,
) -> str:
    """Mitigation 8 (role-by-role decomposition) under decision 9 (plan.md):
    the team structure is code-owned (config team.base_roles — the LLM's
    6-way role choice showed sampling variance across regenerations, e.g.
    Domain Expert <-> Project Coordinator for the same solicitation), so
    the model writes WHAT each fixed role must cover for THIS solicitation.
    The focus text is the generative part: it feeds the req-27 rationales
    and keeps the decomposition solicitation-specific."""
    roles_list = "\n".join(f'- "{r}"' for r in fixed_roles)
    extras = ""
    if domains or methodology:
        extras = (
            "\nExtracted solicitation focus:\n"
            f"Domains: {', '.join(domains) or 'n/a'}\n"
            f"Methodology: {', '.join(methodology) or 'n/a'}\n"
        )
    return (
        "You are designing a university research team for this grant "
        "solicitation.\n\n"
        f"{solicitation_brief(solicitation, settings)}\n"
        f"{extras}\n"
        f"The team has EXACTLY these roles:\n{roles_list}\n\n"
        "For EACH role, write a one-sentence focus: what that person must "
        "cover for THIS solicitation (required methods, domain depth, data "
        "and computational work, leadership).\n"
        f"{_JSON_RULE} Schema:\n"
        '{"roles": [{"role": "<exact name from the list>", '
        '"focus": "<one sentence>"}]}'
    )


def _dossier(
    candidate: ScoredCandidate,
    evidence: Sequence[EvidenceChunk],
    settings: dict | None = None,
) -> str:
    """One team member's bounded profile: the req 17 inputs (metrics,
    recency, prestige signal, PI eligibility) + publication titles keyed by
    EID. Titles only — verbatim abstracts stay in the evidence pack for
    verification (req 18) and the UI (req 27); the prompt stays small (req 21).
    """
    f = candidate.faculty
    quartile = f.top_journal_quartile
    lines = [
        f"## {f.name} (Scopus ID: {f.scopus_id})",
        f"Department: {f.department or 'n/a'} | Rank: {f.rank or 'n/a'}"
        f" | PI-eligible: {'yes' if f.pi_eligible else 'no'}",
        f"H5 index: {f.h5:.0f} | Citations: {f.citations:.0f}"
        + (f" | Top-journal quartile: {quartile:.0f}%" if quartile is not None else "")
        + f" | Overall match score: {candidate.score:.2f}"
        + f" (semantic {candidate.components['semantic']:.2f},"
          f" h5 {candidate.components['h5']:.2f},"
          f" citations {candidate.components['citations']:.2f})",
        "Top publications:",
    ]
    for e in evidence:
        year = e.year if e.year is not None else "n.d."
        lines.append(f"- ({year}) {_clip(e.title, 160)} [EID: {e.eid}]")
    if not evidence:
        lines.append("- (no indexed publications)")
    return "\n".join(lines)


def rationale_prompt(
    entries: Sequence[tuple],
    settings: dict | None = None,
) -> str:
    """Batched req-27 rationale writing for an ALREADY-ASSIGNED team (one
    LLM call per team, not per member — req 21). Each entry is
    (candidate, role, focus, evidence). Membership is deterministic
    (decision 8 in plan.md); the LLM explains WHY each person fits their
    role, citing only EIDs from that person's evidence list. verify.py
    re-checks the echoed Scopus IDs and every cited EID.
    """
    cfg = (settings or load_settings())
    blocks = []
    for candidate, role, focus, evidence in entries:
        blocks.append(
            f"### Role: {role}\nRole focus: {focus}\n"
            f"Assigned member:\n{_dossier(candidate, evidence, cfg)}"
        )
    return (
        "A research team has been assembled for a grant solicitation. "
        "Membership was fixed by a deterministic ranking; your job is to "
        "write each member's justification.\n\n"
        "For EVERY member below, the rationale must state: which part of "
        "the role focus the person satisfies, which publications (by EID) "
        "support it, and why they fit the role.\n\n"
        + "\n\n".join(blocks)
        + "\n\nRules:\n"
        "- Echo the member's exact Scopus ID in your reply for that member.\n"
        "- Cite only EIDs that appear in that person's publication list.\n"
        "- 2-4 sentences per member.\n"
        f"{_JSON_RULE} Schema:\n"
        '{"members": [{"scopus_id": "<echo the member>", '
        '"evidence_eids": ["<EID>", ...], '
        '"rationale": "<2-4 sentences>"}]}'
    )


def critique_prompt(
    team_summary: str,
    critique: str,
    settings: dict | None = None,
) -> str:
    """Req 15 / mitigation 9: translate a free-text user critique into slot
    OPERATIONS on the current team. The model never rewrites the team — code
    applies the validated ops and re-fills only the affected slots."""
    cfg = (settings or load_settings())["llm"]
    return (
        "The user is critiquing a proposed research team. Current team:\n\n"
        f"{_clip(team_summary, int(cfg['max_input_chars']) // 2)}\n\n"
        f"User critique: {_clip(critique, int(cfg['max_input_chars']) // 4)}\n\n"
        "Translate the critique into minimal operations. Only touch the "
        "members the critique is actually about.\n"
        f"{_JSON_RULE} Schema:\n"
        '{"operations": [{"action": "replace" | "remove", '
        '"scopus_id": "<member to change>", "reason": "<one sentence>"}]}\n'
        'Use "replace" to re-fill a slot with someone else and "remove" to '
        "drop the member entirely. An empty operations list is valid when "
        "the critique asks for no member changes."
    )


def team_summary_for_prompt(
    members: Sequence,
    evidence_by_id: dict[str, list[EvidenceChunk]] | None = None,
) -> str:
    """Compact current-team description used by the critique prompt."""
    lines = []
    for m in members:
        line = f"- {m.name} (Scopus ID: {m.scopus_id}) — {m.role}"
        if m.evidence_eids:
            line += f", evidence EIDs: {', '.join(m.evidence_eids)}"
        lines.append(line)
    return "\n".join(lines) or "(empty team)"
