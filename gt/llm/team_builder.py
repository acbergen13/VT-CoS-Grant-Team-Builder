"""Fixed-workflow LLM team builder (Stage 4) — the workflow-agency core.

One team build is a fixed sequence wired by code (the model never chooses
tools, weights, ordering, membership, or team structure):

    optimize query (1 LLM call, req 14) -- keywords only; the retrieval
      query is code-assembled (anchor + sorted keywords, mitigation 4) and
      cached per solicitation
      -> deterministic retrieval (Stage 3, no LLM: req 12 pool + evidence)
      -> role focus decomposition (1 LLM call, mitigation 8 under decision
         9): the LLM writes what each FIXED, config-owned role must cover
         for this solicitation -- cached per solicitation
      -> GREEDY DETERMINISTIC assignment (decision 8, plan.md): each role
         takes the top-ranked remaining eligible candidate, in code
      -> batched rationale writing (1 LLM call for the whole team, req 27),
         chain-of-verified (req 18): echoed Scopus IDs and cited EIDs are
         re-checked against each member's evidence pack; rejects regenerate;
         template rationales are the fallback
      -> final verify_team assertion (reqs 13/18)

Why deterministic membership and structure (measured Oct 21): the ARC
endpoint's thinking models are not sampling-reproducible even at
temperature 0.0 with a fixed seed — LLM-chosen membership overlapped 0-1/5
across identical builds, a drifting optimized query swapped the entire
candidate pool, and the LLM's 6-way role choice flipped (Domain Expert <->
Project Coordinator) across regenerations. With membership and structure
assigned in code and the plan cached per solicitation, the same prompt
produces a bit-identical team on every run, even with the LLM completely
unreachable (deterministic fallback query + template rationales). The LLM
still carries the generative requirements — query keywords (14), per-role
focuses, and grounded rationales (27) — under verification (18).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from gt.config import load_settings
from gt.llm.arc import ArcClient
from gt.llm.cache import BuildCache, build_cache_key, plan_cache_key
from gt.llm.optimize import OptimizedQuery, fallback_query, optimize_query
from gt.llm.prompts import parse_json_object, rationale_prompt, role_decompose_prompt
from gt.llm.team import plan_operations
from gt.llm.verify import FillCheck, verify_rationale, verify_team
from gt.retrieval.evidence import EvidenceChunk, build_evidence
from gt.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from gt.state import Solicitation, Team, TeamMember

# Deterministic team structure (decision 9, plan.md): every team is built
# for exactly these roles, PI first (scarce pool fills first). Used when
# config team.base_roles is absent; the config value is the source of truth.
# Always a legal shape (PI + three additional roles, req 13).
DEFAULT_ROLES: list[tuple[str, str]] = [
    ("Principal Investigator", "Lead the project and own the submission."),
    ("Methods/Technical Lead", "Own the solicitation's required methodology."),
    ("Domain Expert", "Provide subject-matter depth for the core topic."),
    ("Data/Computational Specialist", "Handle data and computational work."),
]
DEFAULT_FOCUS = "Cover the solicitation's needs for this role."


@dataclass
class BuildResult:
    """Outcome of one build or revise turn (rendered by the chat UI)."""

    team: Team | None  # None only when no candidates match at all
    optimized: OptimizedQuery | None = None
    pool_size: int = 0
    notes: list[str] = field(default_factory=list)  # user-presentable
    llm_calls: int = 0
    revision: int = 0
    cached: bool = False  # served from disk cache; zero LLM calls


class TeamBuilder:
    """Deterministic membership + verified LLM rationales, end to end."""

    def __init__(
        self,
        client: ArcClient,
        pipeline: RetrievalPipeline,
        settings: dict | None = None,
        cache: BuildCache | None = None,
    ) -> None:
        self._client = client
        self._pipeline = pipeline
        self._settings = settings or load_settings()
        self._llm_cfg = self._settings["llm"]
        self._team_cfg = self._settings["team"]
        self._retrieval_cfg = self._settings["retrieval"]
        self._cache = cache if cache is not None else BuildCache()
        self._cache_enabled = bool(self._llm_cfg.get("build_cache", True))

    # ------------------------------------------------------------------ build

    def build(self, solicitation: Solicitation, extra_instructions: str = "") -> BuildResult:
        cache_key = build_cache_key(
            solicitation.opp_id,
            self._client.model,
            revision=0,
            anchor=(solicitation.title or "") + (solicitation.synopsis or "")[:200],
            extra=extra_instructions,
        )
        if self._cache_enabled:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return self._result_from_cache(hit)

        calls_before = self._client.calls
        notes: list[str] = []

        plan, plan_was_cached = self._load_plan(solicitation, extra_instructions)
        optimized = OptimizedQuery(
            query=plan["query"],
            domains=plan.get("domains", []),
            methodology=plan.get("methodology", []),
            source=plan.get("source", "llm"),
            note=plan.get("note", "") if not plan_was_cached else "",
        )
        if optimized.note:
            notes.append(optimized.note)

        retrieval = self._pipeline.retrieve(optimized.query)
        if not retrieval.ranked:
            return BuildResult(
                team=None,
                optimized=optimized,
                notes=notes + [
                    "No faculty matched this solicitation's topic well enough "
                    "to build a team. Try a different solicitation."
                ],
                llm_calls=self._client.calls - calls_before,
            )

        evidence = dict(retrieval.evidence_by_id)
        members, used_by_id = self._assign_members(plan["roles"], retrieval, evidence, notes)
        self._write_rationales(
            members, [focus for _, focus in plan["roles"]], evidence, used_by_id, notes
        )

        team = Team(
            solicitation_opp_id=solicitation.opp_id,
            members=members,
            revision=0,
            query=optimized.query,
        )
        self._assert_valid(team, evidence)
        result = BuildResult(
            team=team,
            optimized=optimized,
            pool_size=len(retrieval.ranked),
            notes=notes,
            llm_calls=self._client.calls - calls_before,
            revision=0,
        )
        if self._cache_enabled and optimized.source == "llm":
            # Don't pin builds produced from the degraded fallback query —
            # once the LLM recovers, the next build should optimize again.
            self._cache.put(cache_key, self._result_to_cache(result))
        return result

    # ----------------------------------------------------------------- revise

    def revise(
        self, team: Team, solicitation: Solicitation, critique: str
    ) -> BuildResult:
        """Req 15 / mitigation 9: critique -> slot ops; only the affected
        slots are re-assigned (deterministically). Kept members are carried
        over as the SAME objects — the LLM never rewrites them."""
        cache_key = build_cache_key(
            solicitation.opp_id,
            self._client.model,
            revision=team.revision + 1,
            anchor=(team.query or "") + (critique or "")[:200],
            extra="",
        )
        if self._cache_enabled:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return self._result_from_cache(hit)

        calls_before = self._client.calls
        notes: list[str] = []

        plan = plan_operations(self._client, team, critique, self._settings)
        if plan.note:
            notes.append(plan.note)
        if not plan.ops:
            result = BuildResult(
                team=team,
                pool_size=0,
                notes=notes + ["No team changes were applied."],
                llm_calls=self._client.calls - calls_before,
                revision=team.revision,
            )
            # A failed critique translation (LLM down) must not pin "no
            # changes"; a genuine no-op may be cached.
            if "Could not translate" not in plan.note and self._cache_enabled:
                self._cache.put(cache_key, self._result_to_cache(result))
            return result

        query = team.query or fallback_query(solicitation, self._settings)
        retrieval = self._pipeline.retrieve(query)  # deterministic: same pool
        evidence = dict(retrieval.evidence_by_id)

        members = list(team.members)
        assigned = {m.scopus_id for m in members}
        used_by_id = {c.faculty.scopus_id: c for c in retrieval.ranked}
        changed: list[tuple[TeamMember, str]] = []  # (replacement, role focus)
        for op in plan.ops:
            idx = next(
                (i for i, m in enumerate(members) if m.scopus_id == op.scopus_id), None
            )
            if idx is None:
                continue
            old = members.pop(idx)
            if op.action == "remove":
                assigned.discard(old.scopus_id)
            # On "replace", old.scopus_id STAYS in assigned — the person
            # being replaced must not be greedily re-picked for their own
            # slot (that would silently undo the user's critique).
            pool = self._pool_for_role(old.role, retrieval, assigned, evidence, notes)
            if not pool:
                notes.append(
                    f"Could not re-fill {old.role!r} (no eligible candidate "
                    "left); the slot was removed."
                )
                continue
            focus = (
                f"Replacement for {old.name} ({old.role}): "
                f"{op.reason or 'user critique'}"
            )
            replacement = self._member_from(pool[0], old.role, evidence)
            used_by_id[replacement.scopus_id] = pool[0]
            members.insert(idx, replacement)
            assigned.add(replacement.scopus_id)
            changed.append((replacement, focus))

        if changed:
            self._write_rationales(
                [m for m, _ in changed],
                [focus for _, focus in changed],
                evidence,
                used_by_id,
                notes,
            )

        revised = Team(
            solicitation_opp_id=solicitation.opp_id,
            members=members,
            revision=team.revision + 1,
            query=query,
        )
        self._assert_valid(revised, evidence)
        result = BuildResult(
            team=revised,
            pool_size=len(retrieval.ranked),
            notes=notes,
            llm_calls=self._client.calls - calls_before,
            revision=revised.revision,
        )
        if self._cache_enabled:
            self._cache.put(cache_key, self._result_to_cache(result))
        return result

    # ------------------------------------------------------------- build plan

    def _load_plan(
        self, solicitation: Solicitation, extra_instructions: str
    ) -> tuple[dict, bool]:
        """Optimized query + role decomposition, cached per solicitation
        (reqs 19/21): the plan shapes the candidate pool, so pinning it pins
        every downstream result, and the cache skips 2 LLM calls per
        re-build. Fallback plans (LLM was down) are NOT cached — a transient
        outage must not permanently pin a degraded query.
        """
        key = plan_cache_key(
            solicitation.opp_id,
            self._client.model,
            solicitation.title or "",
            solicitation.synopsis or "",
            extra=extra_instructions,
        )
        if self._cache_enabled:
            hit = self._cache.get(key)
            if hit is not None:
                return hit, True

        optimized = optimize_query(self._client, solicitation, self._settings)
        roles = self._decompose_roles(solicitation, optimized, extra_instructions)
        plan = {
            "query": optimized.query,
            "domains": optimized.domains,
            "methodology": optimized.methodology,
            "source": optimized.source,
            "note": optimized.note,
            "roles": [[role, focus] for role, focus in roles],
        }
        if self._cache_enabled and optimized.source == "llm":
            self._cache.put(key, plan)
        return plan, False

    def _decompose_roles(
        self,
        solicitation: Solicitation,
        optimized: OptimizedQuery,
        extra_instructions: str,
    ) -> list[tuple[str, str]]:
        """Mitigation 8 (role-by-role decomposition) under decision 9: the
        LLM writes each FIXED role's focus for this solicitation; WHICH
        roles the team has is code-owned (config team.base_roles). Measured
        Oct 21 (gpt-oss-120b): the LLM's 6-way role choice varied across
        regenerations of the same solicitation (Domain Expert <->
        Project Coordinator in the 5th slot), flipping role assignments
        wholesale. Same remedy as decision 8 — discrete structure stays
        deterministic; the LLM keeps the generative part (focus text that
        feeds the req-27 rationales)."""
        base = [
            r
            for r in (self._team_cfg.get("base_roles") or [r for r, _ in DEFAULT_ROLES])
            if r in self._team_cfg["roles"]
        ] or [r for r, _ in DEFAULT_ROLES]
        prompt = role_decompose_prompt(
            solicitation,
            base,
            optimized.domains,
            optimized.methodology,
            self._settings,
        )
        if extra_instructions.strip():
            prompt += (
                "\n\nAdditional user instructions: "
                + " ".join(extra_instructions.split())[
                    : int(self._llm_cfg["query_max_chars"])
                ]
            )

        focuses: dict[str, str] = {}
        try:
            reply = self._client.complete([{"role": "user", "content": prompt}])
            data = parse_json_object(reply)
            raw = data.get("roles", [])
            for item in raw if isinstance(raw, list) else []:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip()
                focus = " ".join(str(item.get("focus", "")).split())[:300]
                if role in base and role not in focuses:
                    focuses[role] = focus or DEFAULT_FOCUS
        except Exception:
            focuses = {}  # every role falls back to the generic focus
        return [(role, focuses.get(role, DEFAULT_FOCUS)) for role in base]

    # ------------------------------------------------ deterministic assignment

    def _assign_members(
        self,
        roles: list[list[str]] | list[tuple[str, str]],
        retrieval: RetrievalResult,
        evidence: dict[str, list[EvidenceChunk]],
        notes: list[str],
    ) -> tuple[list[TeamMember], dict]:
        """Greedy deterministic fill (decision 8): each role takes the
        top-ranked remaining eligible candidate. Same pool + same roles ->
        bit-identical membership, with or without the LLM."""
        members: list[TeamMember] = []
        assigned: set[str] = set()
        used_by_id = {c.faculty.scopus_id: c for c in retrieval.ranked}
        for role, focus in roles:
            pool = self._pool_for_role(role, retrieval, assigned, evidence, notes)
            if not pool:
                notes.append(f"Skipped role {role!r}: no eligible candidate left.")
                continue
            member = self._member_from(pool[0], role, evidence)
            used_by_id[member.scopus_id] = pool[0]
            members.append(member)
            assigned.add(member.scopus_id)
        return members, used_by_id

    def _pool_for_role(
        self,
        role: str,
        retrieval: RetrievalResult,
        assigned: set[str],
        evidence: dict[str, list[EvidenceChunk]],
        notes: list[str],
    ) -> list:
        """Candidates eligible for one slot. PI slots widen deterministically
        to the full roster (same fitted scorer) when the retrieved top-k has
        no PI-eligible person — the team always gets its PI (req 13)."""
        pool = [c for c in retrieval.ranked if c.faculty.scopus_id not in assigned]
        if role != "Principal Investigator":
            return pool
        pi_pool = [c for c in pool if c.faculty.pi_eligible]
        if pi_pool:
            return pi_pool
        widened = [
            c
            for c in self._pipeline.scorer.score(retrieval.semantic_by_id)
            if c.faculty.pi_eligible and c.faculty.scopus_id not in assigned
        ][:5]
        if widened:
            notes.append(
                "No PI-eligible faculty in the retrieved top matches; the PI "
                "slot was widened to the top professorial-ranked candidates "
                "by the deterministic score."
            )
            for c in widened:
                if c.faculty.scopus_id not in evidence:
                    evidence[c.faculty.scopus_id] = self._recent_evidence(
                        c.faculty.scopus_id
                    )
        return widened

    def _recent_evidence(self, scopus_id: str) -> list[EvidenceChunk]:
        """Recency-sorted evidence pack for a candidate outside the retrieved
        pool (req 17 recency input; deterministic tie-break on EID)."""
        owned = [c for c in self._pipeline.chunks if scopus_id in c.pub.cos_author_ids]
        owned.sort(key=lambda c: (-(c.pub.year or 0), c.id))
        cap = int(self._retrieval_cfg["evidence_per_author"])
        return build_evidence([(c, 0.0) for c in owned[:cap]])

    def _member_from(
        self, candidate, role: str, evidence: dict[str, list[EvidenceChunk]]
    ) -> TeamMember:
        """Deterministic member record. Evidence EIDs are that person's top
        chunks — the UI links each one to its verbatim source (req 27). The
        rationale starts empty; _write_rationales fills it."""
        pack = evidence.get(candidate.faculty.scopus_id, [])
        cap = int(self._retrieval_cfg["evidence_per_author"])
        return TeamMember(
            scopus_id=candidate.faculty.scopus_id,
            name=candidate.faculty.name,
            department=candidate.faculty.department,
            rank=candidate.faculty.rank,
            role=role,
            score=candidate.score,
            score_components=dict(candidate.components),
            evidence_eids=[e.eid for e in pack[:cap]],
            rationale="",
        )

    # -------------------------------------------------------- LLM rationales

    def _write_rationales(
        self,
        members: list[TeamMember],
        focuses: list[str],
        evidence: dict[str, list[EvidenceChunk]],
        used_by_id: dict,
        notes: list[str],
    ) -> None:
        """One batched LLM call writes req-27 rationales for the given
        members; verify_rationale re-checks each entry (echoed Scopus ID +
        cited EIDs against that member's evidence pack); rejects regenerate
        as a batch; members still unverified after retries get deterministic
        template rationales (req 19: complete and explainable even LLM-down).
        """
        if not members:
            return
        cap = int(self._retrieval_cfg["evidence_per_author"])
        max_retries = int(self._llm_cfg["verify_max_retries"])

        # Members without a dossier (shouldn't happen) go straight to template.
        pending = [
            (m, focus)
            for m, focus in zip(members, focuses)
            if m.scopus_id in used_by_id
        ]
        templated = [m for m in members if m.scopus_id not in used_by_id]

        for _attempt in range(max_retries + 1):
            if not pending:
                break
            entries = [
                (
                    used_by_id[m.scopus_id],
                    m.role,
                    focus,
                    evidence.get(m.scopus_id, []),
                )
                for m, focus in pending
            ]
            try:
                reply = self._client.complete(
                    [{"role": "user", "content": rationale_prompt(entries, self._settings)}]
                )
                data = parse_json_object(reply)
                raw = data.get("members", [])
                raw = raw if isinstance(raw, list) else []
            except Exception:
                continue  # retry the same batch

            replied: dict[str, dict] = {}
            for entry in raw:
                if isinstance(entry, dict):
                    replied[str(entry.get("scopus_id", "")).strip()] = entry
            still_pending: list[tuple[TeamMember, str]] = []
            for member, focus in pending:
                entry = replied.get(member.scopus_id)
                if entry is None:
                    still_pending.append((member, focus))
                    continue
                check: FillCheck = verify_rationale(
                    entry,
                    member.scopus_id,
                    member.name,
                    {e.eid for e in evidence.get(member.scopus_id, [])},
                )
                if check.ok:
                    member.rationale = " ".join(
                        str(entry.get("rationale", "")).split()
                    )
                    eids = [str(e) for e in entry.get("evidence_eids", [])][:cap]
                    if eids:
                        member.evidence_eids = eids
                else:
                    still_pending.append((member, focus))
            pending = still_pending

        for member, _focus in pending + [(m, "") for m in templated]:
            self._template_rationale(member, evidence, notes)

    def _template_rationale(
        self,
        member: TeamMember,
        evidence: dict[str, list[EvidenceChunk]],
        notes: list[str],
    ) -> None:
        """Deterministic fallback rationale (reqs 19/27): states the score
        breakdown and lists the supporting publications by EID."""
        pack = evidence.get(member.scopus_id, [])
        cap = int(self._retrieval_cfg["evidence_per_author"])
        titles = "; ".join(f"{e.title[:80]} [EID {e.eid}]" for e in pack[:cap])
        member.rationale = (
            f"Highest-ranked remaining candidate for {member.role} "
            f"(match score {member.score:.2f}: semantic "
            f"{member.score_components['semantic']:.2f}, H5 "
            f"{member.score_components['h5']:.2f}, citations "
            f"{member.score_components['citations']:.2f}). Supporting "
            f"publications: {titles or 'none'}"
        )
        notes.append(
            f"The LLM rationale for {member.name} ({member.role}) could not "
            "be verified; a deterministic score-based rationale was used."
        )

    # -------------------------------------------------------------- final gate

    def _assert_valid(self, team: Team, evidence: dict[str, list[EvidenceChunk]]) -> None:
        check = verify_team(team, evidence, self._settings)
        if not check.ok:
            raise RuntimeError(
                f"internal error: assembled team failed final verification: {check.reasons}"
            )

    # ------------------------------------------------------------ cache serde

    def _result_to_cache(self, result: BuildResult) -> dict:
        return {
            "team": asdict(result.team) if result.team else None,
            "optimized": asdict(result.optimized) if result.optimized else None,
            "pool_size": result.pool_size,
            "notes": result.notes,
            "llm_calls": result.llm_calls,
            "revision": result.revision,
        }

    def _result_from_cache(self, payload: dict) -> BuildResult:
        team = None
        if payload.get("team"):
            t = dict(payload["team"])
            t["members"] = [TeamMember(**m) for m in t.get("members", [])]
            team = Team(**t)
        optimized = None
        if payload.get("optimized"):
            optimized = OptimizedQuery(**payload["optimized"])
        return BuildResult(
            team=team,
            optimized=optimized,
            pool_size=int(payload.get("pool_size", 0)),
            notes=list(payload.get("notes", [])),
            llm_calls=0,
            revision=int(payload.get("revision", 0)),
            cached=True,
        )
