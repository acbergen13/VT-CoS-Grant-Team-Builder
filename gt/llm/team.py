"""Team-shape operations for revision turns (req 15, mitigation 9).

The user critiques the proposed team in free text; ONE LLM call translates
that critique into slot operations. Code then validates the operations
against the actual team — unknown people, malformed actions, and duplicate
targets are dropped, and three deterministic guards keep the team legal and
the change minimal:

- scope guard (issue 9 / mitigation 9): when the critique explicitly refers
  to particular member(s) — by name, Scopus ID, or role — operations may
  only touch those members. The model can never drop or swap unrelated
  members, no matter what it proposes.
- removing the sole Principal Investigator becomes a replacement (req 13),
- removes that would break the minimum team shape are refused.

Only the affected slots are re-filled; every other member object is carried
over untouched (the LLM never redesigns the whole team).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from gt.config import load_settings
from gt.llm.arc import ArcClient
from gt.llm.prompts import critique_prompt, parse_json_object, team_summary_for_prompt
from gt.state import Team

VALID_ACTIONS = ("replace", "remove")

# Critique phrasings that refer to a role, mapped from the fixed role list.
_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "Principal Investigator": ("principal investigator", "pi"),
    "Co-PI": ("co-principal investigator", "co-pi", "co-principal"),
    "Co-Principal Investigator": ("co-principal investigator", "co-pi", "co-principal"),
    "Methods/Technical Lead": ("methods lead", "technical lead", "methods"),
    "Domain Expert": ("domain expert",),
    "Data/Computational Specialist": (
        "data specialist",
        "computational specialist",
        "data",
        "computational",
    ),
    "Project Coordinator": ("project coordinator", "coordinator"),
}


def _mentioned_member_ids(critique: str, team: Team) -> set[str] | None:
    """Members the critique explicitly refers to, or None when it names no
    one in particular (then the critique is about the team as a whole and
    the LLM's translation decides scope). Matching is deterministic:
    full/last/first name, Scopus ID, or a role alias, on word boundaries."""
    hay = (critique or "").lower()
    if not hay.strip():
        return None

    def _hit(needle: str) -> bool:
        return bool(re.search(rf"\b{re.escape(needle.lower())}\b", hay))

    mentioned: set[str] = set()
    for m in team.members:
        parts: list[str] = [(m.name or "").strip()]
        name = parts[0]
        if not name:
            continue
        if "," in name:  # roster format: "Last, First"
            last, first = (p.strip() for p in name.split(",", 1))
            parts += [last, first]
        # Non-comma names match on the full name only — last-token matching
        # over-matches when members share tokens (e.g. "ML Person" vs
        # "Bridge Person").
        if m.scopus_id:
            parts.append(m.scopus_id)
        if any(len(p) >= 3 and _hit(p) for p in parts):
            mentioned.add(m.scopus_id)
        aliases = _ROLE_ALIASES.get(m.role, (m.role.lower(),))
        if any(_hit(a) for a in aliases):
            mentioned.add(m.scopus_id)
    return mentioned or None


@dataclass
class SlotOp:
    """One validated operation on a current team member."""

    action: str  # "replace" | "remove"
    scopus_id: str
    reason: str = ""


@dataclass
class RevisionPlan:
    """Validated critique translation + any user-presentable notes."""

    ops: list[SlotOp] = field(default_factory=list)
    note: str = ""


def plan_operations(
    client: ArcClient,
    team: Team,
    critique: str,
    settings: dict | None = None,
) -> RevisionPlan:
    """Free-text critique -> validated slot operations (one LLM call)."""
    cfg = settings or load_settings()
    team_cfg = cfg["team"]
    member_ids = {m.scopus_id for m in team.members}
    summary = team_summary_for_prompt(team.members)
    try:
        reply = client.complete(
            [{"role": "user", "content": critique_prompt(summary, critique, cfg)}]
        )
        data = parse_json_object(reply)
        raw_ops = data.get("operations", [])
        if not isinstance(raw_ops, list):
            raw_ops = []
    except Exception as exc:
        return RevisionPlan(
            note=(
                "Could not translate the critique into team changes "
                f"({type(exc).__name__}); try rephrasing. The current team "
                "is unchanged."
            )
        )

    ops: list[SlotOp] = []
    seen: set[str] = set()
    dropped = 0
    for raw in raw_ops:
        if not isinstance(raw, dict):
            dropped += 1
            continue
        action = str(raw.get("action", "")).strip().lower()
        scopus_id = str(raw.get("scopus_id", "")).strip()
        reason = " ".join(str(raw.get("reason", "")).split())[:300]
        if action not in VALID_ACTIONS or scopus_id not in member_ids or scopus_id in seen:
            dropped += 1
            continue
        ops.append(SlotOp(action=action, scopus_id=scopus_id, reason=reason))
        seen.add(scopus_id)

    # Guard 0 (issue 9 / mitigation 9): when the critique explicitly refers
    # to particular member(s), operations may only touch those members —
    # the LLM must never redesign the rest of the team.
    mentioned = _mentioned_member_ids(critique, team)
    if mentioned:
        scoped = [op for op in ops if op.scopus_id in mentioned]
        dropped += len(ops) - len(scoped)
        ops = scoped

    # Guard 1 (req 13): the team always keeps a Principal Investigator —
    # removing the last PI becomes a replacement of that slot.
    pi_ids = {m.scopus_id for m in team.members if m.role == "Principal Investigator"}
    for op in ops:
        if op.action == "remove" and op.scopus_id in pi_ids and len(pi_ids) == 1:
            op.action = "replace"
            op.reason = (op.reason + " " if op.reason else "") + (
                "(PI slot must stay filled — converted to a replacement)"
            )

    # Guard 2: refuse removes that would break the minimum team shape
    # (>= min_pi PIs and >= min_additional_roles other roles).
    min_members = int(team_cfg["min_pi"]) + int(team_cfg["min_additional_roles"])
    remaining = len(team.members)
    kept: list[SlotOp] = []
    for op in ops:
        if op.action == "remove":
            if remaining - 1 < min_members:
                dropped += 1
                continue
            remaining -= 1
        kept.append(op)

    note = ""
    if dropped:
        note = (
            f"{dropped} critique operation(s) ignored (unknown member, "
            "malformed, outside the scope of the critique, or they would "
            "break the required team shape)."
        )
    return RevisionPlan(ops=kept, note=note)
