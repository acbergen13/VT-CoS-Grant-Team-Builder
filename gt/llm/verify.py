"""Chain-of-verification (req 18, mitigation 2) — pure code, no LLM.

Every LLM proposal is checked against ground truth before it can join a
team: the offered candidate pool (roster), the fixed role list, PI-eligible
ranks (D8), and the author's evidence pack (real publication EIDs). Anything
unmatched is rejected with a concrete reason; the builder feeds the reasons
back into a regeneration prompt, and falls back to deterministic selection
when retries are exhausted. Nothing reaches the user unverified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gt.config import load_settings
from gt.retrieval.evidence import EvidenceChunk
from gt.state import Team


@dataclass
class FillCheck:
    """Verification verdict for one proposed role fill."""

    ok: bool
    reasons: list[str] = field(default_factory=list)


def verify_rationale(
    entry: dict,
    scopus_id: str,
    name: str,
    offered_eids: set[str],
) -> FillCheck:
    """Check one LLM rationale entry against the assigned member (req 18).

    The member was assigned deterministically (decision 8), so the model's
    reply is grounded iff it is ABOUT that member: the echoed Scopus ID must
    match, every cited EID must be in that member's evidence pack, and the
    rationale must be non-empty. Unmatched entries are rejected with reasons
    that feed the regeneration prompt; template rationales are the fallback.
    """
    reasons: list[str] = []

    if not isinstance(entry, dict):
        return FillCheck(False, [f"entry for {name} is not a JSON object"])

    echoed = str(entry.get("scopus_id", "")).strip()
    if echoed != scopus_id:
        reasons.append(
            f"reply is about Scopus ID {echoed!r} but must justify "
            f"{name} (Scopus ID {scopus_id!r})"
        )

    cited = entry.get("evidence_eids", [])
    cited = cited if isinstance(cited, list) else [cited]
    bad = [str(e) for e in cited if str(e) not in offered_eids]
    if bad:
        reasons.append(
            f"EIDs not found in {name}'s publication evidence: "
            f"{', '.join(bad)}; cite only the EIDs listed for that person"
        )

    if not str(entry.get("rationale", "")).strip():
        reasons.append(f"rationale for {name} is empty")

    return FillCheck(ok=not reasons, reasons=reasons)


def verify_team(
    team: Team,
    evidence_by_id: dict[str, list[EvidenceChunk]],
    settings: dict | None = None,
) -> FillCheck:
    """Final whole-team gate (req 13 + req 18): shape and every member's
    evidence re-checked against the same ground truth. Used by the builder's
    last line of defense and by tests."""
    cfg = settings or load_settings()
    team_cfg = cfg["team"]
    fixed_roles = set(team_cfg["roles"])
    reasons: list[str] = []

    roles = [m.role for m in team.members]
    ids = [m.scopus_id for m in team.members]
    if len(set(ids)) != len(ids):
        reasons.append("a person holds more than one role")
    bad = [r for r in roles if r not in fixed_roles]
    if bad:
        reasons.append(f"roles outside the fixed list: {bad}")
    pi_count = roles.count("Principal Investigator")
    if pi_count < int(team_cfg["min_pi"]):
        reasons.append("team has no Principal Investigator")
    non_pi = [r for r in roles if r != "Principal Investigator"]
    if len(set(non_pi)) < int(team_cfg["min_additional_roles"]):
        reasons.append("team needs at least one additional role")

    for m in team.members:
        pack = evidence_by_id.get(m.scopus_id)
        if not pack:
            # Ground truth is the retrieved evidence pool: a member with no
            # pack was never offered as a candidate (e.g. an injected fake
            # author from a bad LLM reply or a corrupted cache entry).
            reasons.append(f"{m.name}: not in the verified candidate pool")
            continue
        offered = {e.eid for e in pack}
        unmatched = [e for e in m.evidence_eids if e not in offered]
        if unmatched:
            reasons.append(f"{m.name}: unmatched evidence EIDs {unmatched}")
        if not m.evidence_eids:
            reasons.append(f"{m.name}: no supporting publication evidence")
        if not (m.rationale or "").strip():
            reasons.append(f"{m.name}: empty rationale")

    return FillCheck(ok=not reasons, reasons=reasons)
