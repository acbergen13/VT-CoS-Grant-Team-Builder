# plan.md — Mid-term Build Plan (written Oct 20, 2026)

Working document for the build. **Sources of truth** (read before changing
anything here): `docs/architecture/mid_term_project_reqs.qmd` (27 binding
requirements + decided architecture; graded AI-free coursework — do NOT edit)
and `docs/specifications/specification.qmd` (use cases, 1000-pt rubric,
submission format). Repo conventions: `AGENTS.md`.

To resume a session: read `AGENTS.md`, this file, then check stage statuses in
the build plan below. **Note: the repo's git history may be stale**, so judge
what exists from the filesystem, not `git log`. **Current position (end of
Oct 22 session): Stages 0–7 are done. Stage 6 added the Elsevier production
source (`gt/faculty/elsevier.py`) behind the existing `FacultySource` seam,
with an EID/author disk cache, bounded abstract fallback, corpus-hash index
invalidation, and source-aware LLM cache keys. Stage 7 hardened verification,
added browser error-path tests, passed the final two-tier reproducibility
gate, and built/validated the submission zip. `data_source: csv` remains the
submitted default. The only manual deliverable left is the OBS screen
recording.** Implementation notes and the Stage 4–7 evidence are in "Session
handoff" below.

## Locked decisions (Oct 20)

1. **Elsevier production source: full implementation + EID-keyed disk cache**
   behind the `FacultySource` interface (Stage 6). CSV stays the default run
   mode so the instructor's unzip-and-run works off-campus. [user decision]
2. Data CSVs stay out of git (`publications.csv` is 51MB); data ships only in
   the submission zip. `.gitignore`: `.env`, `data/`, `__pycache__/`.
3. All Python work in conda env `ads_5984` (Python 3.12). Present: streamlit,
   chromadb, langchain + langchain-openai + langchain-core, openai, neo4j,
   pandas, numpy, sentence-transformers, torch 2.12, transformers, scikit-learn,
   python-dotenv, rank-bm25, pyyaml, tenacity, httpx. **Missing: pytest**
   (pip install into the env when Stage 2 starts).
4. Deadline: source zip + OBS video **Fri Oct 23**. Instructor unzips and runs
   from README alone — README runnability is graded.
5. Instructor runs the app in the shared `ads_5984` conda env (confirmed
   Oct 20). README quick start leads with `conda activate ads_5984`;
   `requirements.txt` was removed (Oct 20) — the course env is the only
   supported environment, so no dependency manifest ships in the zip.
6. **fetchOpportunity fallback (Oct 20):** the documented
   `api.grants.gov/v1/api/fetchOpportunity` endpoint returns 502 — its
   internal proxy is down. The backend it fronts
   (`POST https://apply07.grants.gov/grantsws/rest/opportunity/details`,
   form-encoded `oppId`) works and returns the full detail payload. The
   client tries the documented endpoint once (no retry) and falls back to the
   backend with retries; if the front endpoint heals we return to it
   transparently. Pinned detail schemas: posted opps carry a `synopsis`
   block (`synopsisDesc` HTML, `applicantEligibilityDesc`, `awardCeiling`
   may be the string "none"); forecasted opps carry a `forecast` block
   (numeric awards, `forecastDesc`, estimated dates, no hard close date).
7. **Award filtering (Oct 20):** search2 listings carry no award amounts —
    they exist only after fetchOpportunity. The award-ceiling filter therefore
    applies to detail-loaded records only (UI says so). Bulk detail enrichment
    is a Stage 5 decision, not Stage 1.
8. **Deterministic member assignment (Oct 21, measured):** live gating showed
   GLM-5.3's thinking pass is non-reproducible at any temperature (temp 0.0:
   0/5 member overlap across three builds, verification retries 3→14, and a
   drifting optimized query swapped the whole candidate pool). Per the
   architecture doc's own mitigations 6/8 ("deterministic process", "we make
   a greedy choice"), membership is now assigned greedily in code (rank
   order per LLM-decomposed role, PI-eligibility enforced); the LLM keeps
   query optimization (req 14), role decomposition, and batched grounded
   rationales (req 27) under chain-of-verification (req 18); decision 9 later
   narrowed the LLM's role work to per-role focus sentences only. A per-
   solicitation disk cache pins the optimized query + role plan so re-runs
   hit identical pools with no repeat LLM calls (req 21). Result: same
   prompt → bit-identical team membership, even LLM-down.
9. **Deterministic team structure + canonical query (Oct 21, measured on
   `gpt-oss-120b`).** The endpoint is not sampling-reproducible even at
   `temperature=0.0` with a fixed seed: keyword sets and phrasing drift
   ("mixed-methods design" vs "mixed-methods research"), and the LLM's
   choice of roles for the same solicitation flipped (Domain Expert ↔
   Project Coordinator in the 5th slot) across regenerations. Same principle
   as decision 8 — discrete choices move into code, the LLM keeps generative
   text:
   - **Team structure:** config `team.base_roles` = Principal Investigator,
     Methods/Technical Lead, Domain Expert, Data/Computational Specialist,
     built in that order (PI first). The LLM's role-decompose call now writes
     only each role's one-sentence solicitation-specific FOCUS; it never
     picks which roles. `DEFAULT_FOCUS` covers missing/junk focuses.
   - **Optimized query:** the LLM extracts only `domains` + `methodology`;
     `gt/llm/optimize.py:canonical_query` assembles the search string in code
     = deterministic `fallback_query` anchor (title/agency/description) +
     sorted, lowercased, deduped terms, each half capped by
     `llm.query_max_chars` (400 → ≤800 chars total). Measured rationale: the
     old free-text LLM query was a near-verbatim re-serialization of the
     synopsis + terms, differing only at the truncation tail — no added
     signal, only drift. Empty extraction falls back to the deterministic
     query with `fallback=True`.
   - Rejected: temp-0.0 per-call override and self-consistency/majority
     voting (measured: temp 0.0 does not stabilize; phrase variants make
     string-level voting fragile; +2 calls/build).
10. **Two-tier reproducibility contract (Oct 21) — document in README.**
    - *Same prompt served again (cache hit):* bit-identical team including
      rationales, **0 LLM calls** (plan/build cache, `CACHE_FORMAT = "v4"`).
    - *Full regeneration (cache cleared):* "relatively consistent" (req 19
      wording) — identical role structure, same people measured 4/4 on opp
      45810 in the final gate; residual variance is endpoint term selection,
      not membership logic (assignment is greedy/deterministic).
    Three LLM calls per build (optimize + focuses + batched rationales); a
    revise turn is two calls. Cache keys hash content + `CACHE_FORMAT`, so
    format changes invalidate old entries cleanly.

## .env contract (names only — values never in git/docs)

| Key | Purpose |
|---|---|
| `VT_ARC_API_KEY` | LLM auth (required) |
| `VT_ARC_BASE_URL` | `https://llm-api.arc.vt.edu/api/v1` (required) |
| `VT_ARC_MODEL` | chat model id (required) |
| `ELSEVIER_API_KEY` | production publication source (Stage 6) |
| `ELSEVIER_BASE_URL` | Elsevier API base (required in prod mode) |
| `ELSEVIER_INST_TOKEN` | optional: entitled content off-campus |

- Endpoint + key verified working Oct 20 via `GET /models`.
- Served chat models observed: `vt-arc-llm` (spec default; preset routing to
  DeepSeek-V4.1-Flash), `GLM-5.3`, `GLM-5.3-thinking-high`, `gpt-oss-120b`
  (+thinking variants), `DeepSeek-V4.1-Flash` (+thinking variants).
- Served embedding model: `Qwen3-Embedding-4B` (vLLM, 8192-token ctx).
  **Default plan stays local sentence-transformers** (512-token model) so the
  instructor needs no extra endpoint setup; ARC embeddings are a fallback
  option, not the default.
- Current `.env` uses `VT_ARC_MODEL=gpt-oss-120b` (verified served Oct 21;
  switched from GLM-5.3, which spent `max_tokens` on hidden reasoning and
  returned empty `content` with `finish_reason=length`). `gpt-oss-120b`
  returns clean JSON in `content` (~3s/call); its hidden reasoning appears in
  `message.model_extra["reasoning"]`. Original file used `OPEN_AI_*` key
  names — renamed to `VT_ARC_*` Oct 20 to match the spec's expected-keys
  requirement before any code was written.
- grants.gov needs no auth; ChromaDB is local; nothing else belongs in `.env`.

## Data profile (verified Oct 20 with pandas, `encoding="utf-8-sig"`)

### authors.csv — 501 rows × 18 cols, all College of Science

| Field | Missing | Notes |
|---|--:|---|
| `Author` | 0 | `"Last, First"` — canonical display name |
| `Scopus ID` | 0 | join key; **1 duplicated value** (D1) |
| `H-Index` | 5 | 0–110 |
| `H5 Index` | 76 | 1–26; the 0.25-weight scoring input (D2) |
| `Scholarly Output` | 0 | 1–1,885 |
| `Citation Count` | 0 | 0–49,594, median 526 — extreme skew (D3) |
| `Corporate Collaboration %` | 5 | |
| `International Collaboration %` | 55 | |
| `Top Journal Quartile %` | 58 | 30 zeros; only prestige signal that exists (D7) |
| `Gender`, `Appointment Type`, `Class Description` | 0 | |
| `Rank` | 0 | 25 ranks incl. 92 postdocs, ~60 instructors/lecturers (D8) |
| `Tenure` | 3 | |
| `College` | 0 | all `College of Science` |
| `Department` | 0 | Math 94, Bio 83, Chem 67, Phys 61, Geo 49, Stats 41, Psych 37, Neuro 25, Econ 25, Dean 16, ADS 3 |
| `Functional Title` | 149 | |
| `Hire Date` | 0 | |

### publications.csv — 21,485 rows × 26 cols

| Field | Missing | Notes |
|---|--:|---|
| `Authors` / `Author full names` | 0 | abbreviated + `"Last, First (ScopusID)"` forms |
| `Author(s) ID` | 0 | `"; "`-separated Scopus IDs, no trailing `;` (D4) |
| `Title` / `Year` | 0 | year range 1963–**2026** (early access) |
| `Source title` | 1 | journal name |
| `Cited by` | 0 | pub-level citations |
| `DOI` | 1,399 | 73 duplicate DOIs |
| `Link` | 0 | Scopus URL — use for claim traceability (D9) |
| `Abstract` | 969 (4.5%) | median 1,103 chars / 156 words; p90 263 words (D5) |
| `Author Keywords` / `Index Keywords` | 10,254 / 17,026 | 48% / 79% |
| `Document Type` | 0 | Article 17,594, Conf 1,937, Review 770, Chapter 458, **Editorial 205, Erratum 193, Note 159, Letter 73** (D5) |
| `Publication Stage` | 0 | all `Final` |
| `EID` | 0 | **0 duplicates — canonical pub key** (D9) |
| `Open Access`, `Source`, `ISSN`, `ISBN`, `CODEN` | 0–21,485 | mostly filler |

### Join reality

- 50,868 unique author IDs appear in pubs; roster has 500 unique IDs
  (501 rows − 1 dup). Overlap = 500 → **every roster ID has ≥1 pub**.
- 50,368 external co-author IDs; **1,425 rows contain no CoS author**.
- Pubs per CoS author: median 20, p90 107, **max 1,898** (D6).
- 934 of 20,060 CoS-linked rows have no abstract → index on title+keywords
  fallback, or skip; decide in Stage 3.

## Design flags

- **D1 Duplicate Scopus ID:** `56940337900` = Iliescu, Traian (Math, Professor)
  AND Driscoll, Anne R. (Stats). Deterministic rule: keep first by file order,
  log the drop, document in README.
- **D2 Missing metrics:** H5 missing on 15% (postdocs/instructors mostly).
  Impute to 0 in config — the LLM never sees or fills these (req 12).
- **D3 Citation skew:** 49,594 outlier destroys raw min-max. Use `log1p` then
  min-max; fixed in config.
- **D4 Join mechanics:** split `Author(s) ID` on `";"`, strip, **filter to
  roster set** — external co-authors and no-CoS rows never enter the candidate
  pool or author grouping.
- **D5 Chunking & noise:** one chunk per publication (title + abstract +
  keywords when present). Median 156 words → 512-token embedder (256-token
  MiniLM-L6 would truncate ~25%). Exclude Erratum/Editorial/Note/Letter before
  indexing (placeholder abstracts like "This corrects the article DOI: …").
- **D6 Prolific author:** one author has 1,898 pubs → author-level aggregation
  + MMR diversity is load-bearing (mitigates issue 5 in the architecture doc).
- **D7 Req 17 inputs:** recency from `Year`; prestige exists only author-level
  (`Top Journal Quartile %`) — no per-pub quartile. Recency in retrieval
  metadata; author prestige in the evidence pack.
- **D8 PI eligibility:** 92 postdocs on roster. PI must hold a tenure-track/
  faculty rank — eligible ranks encoded in config, never an LLM choice.
- **D9 Verification keys:** `EID` = unique pub key for chain-of-verification;
  `Link` = complete for traceability UI (reqs 18/27).
- **D10 Environment:** pytest missing (install into `ads_5984`); local
  embedding model needs one-time ~90MB HF download — README must note
  first-run internet. Alternative: `Qwen3-Embedding-4B` on the ARC endpoint.
- **D11 BOM:** both CSVs parse with `encoding="utf-8-sig"`.

## Project structure

```
midterm_project/
├── main.py                  # Streamlit entrypoint: `streamlit run main.py`
├── README.md                # run instructions + .env keys + design justification (graded)
├── .env.example             # VT_ARC_* + ELSEVIER_API_KEY names
├── config/
│   └── settings.yaml        # weights, roles, PI-eligible ranks, retrieval/LLM params
├── gt/                      # the package (imports resolve from repo root)
│   ├── state.py             # session-state schemas: Solicitation, Team, TeamMember, ChatTurn
│   ├── grants/
│   │   ├── client.py        # search2 + fetchOpportunity; timeouts, retries, user-facing errors
│   │   ├── store.py         # local JSON DB: upsert open; on refresh closed → archive (never shown)
│   │   └── service.py       # orchestration shared by UI + CLI (refresh, cached detail load)
│   ├── faculty/
│   │   ├── roster.py        # authors.csv: utf-8-sig, dedupe (D1), impute (D2), normalize (D3)
│   │   ├── publications.py  # publications.csv: CoS filter (D4), doc-type filter (D5)
│   │   ├── source.py        # FacultySource protocol + csv/elsevier factory seam
│   │   ├── elsevier.py      # Scopus Search + Abstract Retrieval + EID/author cache (prod)
│   │   └── scoring.py       # deterministic 0.6/0.25/0.15 ranker
│   ├── retrieval/
│   │   ├── index.py         # chunk builder + persistent ChromaDB index (metadata: author IDs, year)
│   │   ├── search.py        # hybrid: vector + BM25 → reciprocal-rank fusion, wide recall
│   │   ├── rerank.py        # group by author → author-level score → MMR select (D6)
│   │   └── evidence.py      # Evidence objects carrying verbatim source text (grounding)
│   └── llm/
│       ├── arc.py           # OpenAI-compatible VT ARC client, config from .env
│       ├── prompts.py       # templates: optimize / role-decompose / role-fill / revise / verify
│       ├── optimize.py      # req-14 prompt optimization from structured solicitation pieces
│       ├── team_builder.py  # LLM lists needed roles → greedy per-role fill from ranked candidates
│       ├── team.py          # Team object + critique→slot-operations (only affected slots re-run)
│       └── verify.py        # chain-of-verification: roster IDs + EIDs + verbatim quotes
├── app/
│   ├── browser.py           # view 1: refresh, filter/sort, detail fetch, "build a team"
│   └── chat.py              # view 2: chatbot, follow-ups, team cards w/ traceable rationale
├── scripts/
│   ├── refresh_grants.py    # CLI harvest
│   └── build_index.py       # CLI embed + index
└── tests/                   # scoring determinism, store archive semantics, verify rejection
```

## config/settings.yaml spec

```yaml
scoring:
  weights: {semantic: 0.6, h5: 0.25, citations: 0.15}   # never LLM-controlled (req 12)
  h5_normalization: minmax
  citations_normalization: log1p_minmax
  missing_metric_impute: 0
team:
  roles: [Principal Investigator, Co-PI, Methods/Technical Lead, Domain Expert,
          Data/Computational Specialist, Project Coordinator]   # fixed list (req 13)
  base_roles: [Principal Investigator, Methods/Technical Lead, Domain Expert,
               Data/Computational Specialist]                   # decision 9, fill order = priority
  min_pi: 1
  min_additional_roles: 1
  one_role_per_person: true
  team_size: 5
  pi_eligible_ranks: [...]        # tenure-track/faculty ranks; excludes postdocs/instructors/lecturers
retrieval:
  embedding_model: BAAI/bge-small-en-v1.5
  chunk: one per publication (title+abstract+keywords)
  exclude_doc_types: [Erratum, Editorial, Note, Letter]
  candidate_k: 100
  author_pool_k: 25
  select_k: 10                    # authors returned to the team builder
  evidence_per_author: 3
  mmr_lambda: 0.7
llm:
  temperature: 0.1
  seed: 42
  verify_max_retries: 2
  timeout_seconds: 120
  max_output_tokens: 3072
  max_input_chars: 6000
  query_max_chars: 400            # per half of the canonical query
  build_cache: true               # data/llm_cache/, CACHE_FORMAT = "v4"
grants:
  opp_statuses: "forecasted|posted"
elsevier:                           # production source (Stage 6)
  timeout_seconds: 30
  max_retries: 3
  page_size: 25
  max_pages_per_author: 100
  cache_dir: data/elsevier_cache
  abstract_fallback: true
  abstract_fallback_limit: 25
  abstract_fallback_disable_after: 3
data_source: csv                   # csv | elsevier (csv is the shipped default)
```

Local artifacts (gitignored under `data/`): `data/grants/active.json`,
`data/grants/archive.json`, `data/chroma/`, `data/llm_cache/`,
`data/elsevier_cache/`.

## Staged build plan

| Stage | When | Deliverables | Verification gate | Reqs / rubric pts | Status |
|---|---|---|---|---|---|
| **0. Scaffold** | Mon ~1h | tree, settings.yaml, .env.example, README skeleton, main.py view switch + empty session state | app boots in `ads_5984` | — | **done Oct 20** |
| **1. Grants layer** | Mon | client.py (paged search2, fetchOpportunity w/ apply07 fallback, tenacity retries), store.py (open upsert / closed→archive), service.py, filters/sort in browser.py, refresh_grants.py | refresh ×2 live (1467 active, stable); injected fake closed opp → archived, not shown; 39 tests green; headless boot clean | 2,3,6,7,11,22,23 / 150 | **done Oct 20** |
| **2. Faculty layer** | Mon PM–Tue AM | roster.py (dedupe D1, impute D2, PI ranks D8), publications.py (CoS filter D4, doc-type filter D5), source.py FacultySource protocol + CSV impl, scoring.py deterministic ranker | 63 tests green (hand-computed scores, bit-identical reruns); live: 500 faculty (D1 drop = Iliescu kept/Driscoll dropped), 19,461 pubs, all 500 authors covered, 1.8s load+score | 4,5,12 / 100 | **done Oct 20** |
| **3. Retrieval** | Tue | build_index.py, search.py (vector+BM25→RRF), rerank.py (author group → MMR) | "machine learning for climate" returns sensible faculty; no author >2 of top-10; warm query <2s | 10 / 150 | **done Oct 20** — 79 tests green; live index 19,461 chunks (~50s build); gate: sensible cross-dept pool (Carey/Shirzaei/Gel/Allen/Ferreira…), output top-10 = 10 distinct authors (raw chunk recall is crowded — 6/10 one author — which is exactly what author grouping + MMR mitigates), warm query 0.03s, two runs bit-identical (ids, scores, evidence) |
| **4. LLM layer** | Tue PM–Wed AM | arc.py, prompts.py, optimize.py, team_builder.py (canonical query → deterministic role focuses → greedy fill, JSON output), team.py (critique→slot ops w/ scope guard), verify.py (reject/regenerate), cache.py | live gate: role structure identical across regens, people overlap ≥ n−2; cached rerun bit-identical + 0 calls; revise keeps untouched members and replaces PI; 114 tests green | 9,13,14,15,17,18,19,21,27 / 350 | **done Oct 21** — 114 tests at Stage 4 (119 now); see "Session handoff" below for evidence |
| **5. UI** | Wed | browser.py (refresh+spinner, filters, select→detail, build-a-team), session-state handoff, chat.py (auto-build, messages, error surfaces, follow-ups, team cards w/ EID/Link/source traceability) | launch→team in ≤5 interactions; handoff loses nothing; network kill → clean error | 1,8,24 / 100 | **done Oct 21** — AppTest covers auto-build, revision, reuse of the same solicitation, and LLM-down error; 119 tests green; live build rendered team cards with evidence links/verbatim text |
| **6. Prod source + README** | Wed PM–Thu | elsevier.py behind FacultySource (Scopus search + Abstract Retrieval, EID-keyed disk cache), config switch, README design section | offline fake-transport tests pass; cache hit = 0 API calls; corpus hash forces rebuild; factory switch works; on-campus live gate still pending | 25,26,20 | **done Oct 22** — 136 tests green; on-campus live verification remains in Stage 7 |
| **7. Harden & submit** | Thu PM–Fri | reproducibility runs, error paths, zip (with data/), OBS video | two-tier reproducibility pass (decision 10); instructor-follows-README dry run | 19 / submission | **done Oct 22** — 143 tests; two-tier gate passed on opp 363268 (fresh 3/4 overlap ≥ n−2, cached 0 calls + bit-identical); verify_team rejects injected fake authors/publications; browser refresh/detail errors covered; zip built without `.env`/chroma and dry-run from a clean unzip. OBS recording remains manual. |

## Session handoff (Oct 21–22) — resume here

**Stages 4 and 5 are complete.** 119 tests pass:
`conda run -n ads_5984 python -m pytest -q`.

Final live gate (`gpt-oss-120b`, opp 45810 "Sociology", caches cleared):

- Build: 3 LLM calls, pool=10, ~35s; cache-cleared rebuild ~33s.
- Role names run1 == run2 (base 4 from `team.base_roles`).
- Membership `(id, role)` overlap run1~run2: **4/4**; people overlap 4/4.
- Optimized query run1 != run2 (term selection still drifts) — accepted by
  the two-tier contract (decision 10); canonical assembly bounds the damage.
- Cached run3: **0 LLM calls**, membership and full team (incl. rationales)
  bit-identical to run2.
- Revise (replace PI): 2 calls; kept members untouched = True; PI replaced by
  a new person = True.
- GATE PASSED.

**Last-minute implementation notes from this session (all in code):**

- `gt/llm/optimize.py`: `canonical_query()`; `optimize_query()` sorts/dedupes
  terms and falls back when the LLM returns none.
- `gt/llm/team.py`: critique scope guard — ops targeting members the critique
  never mentions are dropped (`_mentioned_member_ids`, `_ROLE_ALIASES`).
  Non-comma names match full name only (last-token matching over-matched,
  e.g. "ML Person" vs "Bridge Person").
- `gt/llm/team_builder.py`: focuses-only decompose against `base_roles`;
  fixed pre-existing bug where role/focus tuples were passed to
  `_write_rationales` (rationale prompt showed tuple reprs).
- `gt/llm/cache.py`: `CACHE_FORMAT = "v4"` (v2 = canonical query, v3 =
  base_roles structure, v4 = source/corpus fingerprint). Old cache entries
  are ignored, not migrated.
- Revise no-op message: "No team changes were applied."

**Stage 5 completion (Oct 21):**

- `app/chat.py` now auto-builds the first team on entering the chat view,
  renders member cards (role, score breakdown, rationale, EID/DOI/Scopus
  links, expandable verbatim source text), exposes build diagnostics, and
  sends critique turns to `TeamBuilder.revise`.
- `gt/state.py` holds the chat state plus lazy session-scoped caches for the
  pipeline / ARC client / builder; the browser resets the conversation only
  when the selected opportunity changes (same solicitation keeps its team).
- `tests/test_chat.py` uses `streamlit.testing.v1.AppTest` to cover auto-build,
  revision, same-solicitation reuse, and an LLM-down error with a retry button.
  Full suite: 119 tests offline.
- `.env.example` now defaults to `gpt-oss-120b` (the model used by the live
  gates); stale "Stages 0–4" README status and the Stage 4 cleanup are done.
- Scratch scripts (`_gate_stage4.py`, `_debug_*.py`) were removed.

**Stage 6 completion (Oct 22):**

- `gt/faculty/elsevier.py` implements `ElsevierClient` (paginated Scopus
  Search with `AU-ID`, `view=COMPLETE`, 0-based `start`, retries and typed
  errors), `ElsevierCache` (`publications/<EID>.json` +
  `authors/<scopus_id>.json`, atomic writes, corrupt-entry fallback,
  duplicate-EID author union), and `ElsevierFacultySource` (roster order,
  doc-type filter, roster-linked records, bounded abstract fallback with
  disable-after-failures). `gt/faculty/source.py` is now the factory/re-export
  seam; the `NotImplementedError` placeholder is gone.
- `config/settings.yaml` has an `elsevier:` block; `.env.example` documents
  `ELSEVIER_BASE_URL`. `data_source: csv` remains the committed default.
- `gt/retrieval/index.py` now writes `corpus.sha256` and `needs_build()`
  compares ids **and** text content, so CSV→Elsevier with identical EIDs
  rebuilds.
- `gt/llm/cache.py` is `CACHE_FORMAT = "v4"` and keys include a
  `data_source:corpus_hash` fingerprint from the pipeline, so CSV-built teams
  cannot be served after the source switch.
- `app/chat.py:_format_error` maps Elsevier config/quota/network/response
  failures to user-facing messages (req 23).
- Offline coverage: `tests/test_elsevier.py` (pagination, normalization,
  cache hits with 0 calls, duplicate-EID merge, doc-type filter, abstract
  fallback/disable, typed errors, factory); index content-hash and cache-key
  tests added. Full suite: **136 tests green**.
- Live smoke test (Oct 22): one roster author search against the real
  Elsevier endpoint succeeded (18 records, abstract present, 1 API call), so
  credentials/base URL/network work. The full on-campus gate (set
  `data_source: elsevier`, rebuild the index, verify ~500 authors / ~19.5k
  publications, build a team) is still pending and belongs to Stage 7. Until
  then the submitted default stays `csv`.

**Stage 6 implementation notes.** The API findings below were verified against
Elsevier's official payload sample and implemented in `elsevier.py`. Keep
`data_source: csv` for the submitted app; use Elsevier only for an on-campus
production run.

API findings (verified against Elsevier's official payload sample):

- Scopus Search: `GET https://api.elsevier.com/content/search/scopus` with
  headers `X-ELS-APIKey`, `Accept: application/json`, optional
  `X-ELS-Insttoken`. Query one roster author at a time with
  `query=AU-ID(<scopus_id>)`, `view=COMPLETE`, `count=25`, and 0-based
  `start`; paginate until `search-results["opensearch:totalResults"]`.
- `view=COMPLETE` entries already carry the fields we need: `eid`,
  `dc:title`, `dc:description` (the abstract — present in the official JSON
  sample), `prism:coverDate`, `prism:publicationName`, `prism:doi`,
  `citedby-count`, `subtypeDescription`, `authkeywords`,
  `author[].authid`, and `link[]` (`@ref=scopus`). **This avoids one Abstract
  Retrieval call per publication.**
- Abstract Retrieval remains the fallback only for entries with a missing
  abstract: `GET /content/abstract/eid/{eid}?view=META_ABS`. Cap fallback
  attempts and disable them after repeated 403/entitlement failures so an
  off-campus run cannot turn into thousands of failed calls.
- The roster still comes from `authors.csv` (req 4); only publication
  metadata/abstracts move behind the API (reqs 5/26). The app's default run
  mode stays `csv` so the submitted zip works without an Elsevier key.

Implementation map:

1. Add `gt/faculty/elsevier.py`; keep `gt/faculty/source.py` as the factory /
   protocol seam and re-export `ElsevierFacultySource` there.
   - `ElsevierClient`: injectable transport (same offline-test pattern as
     `GrantsClient`), env/config loading, paginated `search_author()`,
     `fetch_abstract()`, retries for timeout/429/5xx, and typed
     `ElsevierConfigError` / `ElsevierQuotaError` /
     `ElsevierUnavailableError` / `ElsevierResponseError`.
   - `ElsevierCache`: `data/elsevier_cache/publications/<EID>.json` is the
     canonical EID-keyed record; `data/elsevier_cache/authors/<scopus_id>.json`
     stores that author's EID list so repeat loads make **0 API calls**.
     Use a cache-format constant, atomic writes, corrupt-entry fallback, and a
     `refresh=True` override. Merge duplicate EIDs found in multiple author
     searches by unioning `cos_author_ids`; sort final EIDs for determinism.
   - `ElsevierFacultySource.load_publications()`: iterate roster order, filter
     excluded document types, keep only roster-linked publications, enrich
     missing abstracts through the bounded fallback, then return `Publication`
     objects exactly like the CSV source. Log request/cache-hit counts.
2. Add an `elsevier:` block to `config/settings.yaml`: timeout, retries,
   `page_size: 25`, `max_pages_per_author`, `cache_dir`, abstract fallback
   switch/limit. Add `ELSEVIER_BASE_URL` to `.env.example` (required in prod
   mode for req 20); `ELSEVIER_INST_TOKEN` stays optional.
3. Fix index invalidation before trusting prod mode: `PublicationIndex`'s
   manifest currently stores only EID set, so switching csv→elsevier with the
   same EIDs would silently reuse CSV chunks. Add a corpus content hash
   sidecar (e.g. `corpus.sha256`) to `needs_build()`, plus a test where same
   EIDs with changed text force a rebuild.
4. Extend `app/chat.py:_format_error` with user-facing Elsevier messages
   (missing keys, quota, network, bad response) for req 23.
5. Tests offline with a fake transport: pagination, duplicate-EID merge,
   doc-type filtering, EID/author cache hit with 0 calls, missing-abstract
   fallback, typed errors, and factory selection. Replace the current
   `test_factory_fails_loudly_for_elsevier_until_stage6` expectation.
6. After code passes, update `README.md` (Stage 6 status, production switch,
   cache/index behavior, expected keys), `AGENTS.md`, and mark the Stage 6
   row/gates done here.
7. Verification gate: on campus, set `data_source: elsevier`, rebuild the
   index deliberately, confirm the corpus is ~500 authors / ~19.5k
   publications, and run a team build through the app. Then decide whether to
   leave csv or elsevier as the submitted default (locked decision 1 says
   csv). **Code/tests are complete; the live on-campus gate is still pending.**

**Stage 7 completion (Oct 22):**

- **Reproducibility gate** (shipped CSV corpus + `data/chroma/`, opp `363268`
  "Unlocking Dataset Value for AI-Enabled Scientific Discovery"): two
  cache-cleared builds produced the same four-role structure and 3/4 people
  overlap (contract ≥ n−2 = 2); the drift was confined to the LLM keyword half
  of the canonical query. A third build hit the disk cache with **0 LLM
  calls** and was bit-identical to the second, rationales included.
- **Verification hardening:** `gt/llm/verify.py:verify_team` now rejects a
  member absent from the offered evidence pool, an empty evidence list, an
  unmatched EID, and an empty rationale. This closes the "injected fake
  author/publication" gate even if a corrupted cache or bad reply tries to
  introduce one. New tests: `test_verify_team_rejects_injected_fake_author_*`.
- **Error paths:** `tests/test_browser.py` (AppTest) covers a failed refresh,
  a failed detail fetch, and the empty-store prompt; `tests/test_chat.py`
  already covered LLM-down and Elsevier config/quota/network/response errors.
- **Live build status:** `app/chat.py` now uses an expandable `st.status` and
  `TeamBuilder.build(..., on_optimized=...)` notifies it immediately after the
  keyword-extraction call, so the assembled optimized retrieval prompt is
  visible during the remaining retrieval/rationale work. Covered by
  `test_build_shows_optimized_prompt`. Suite: **143 tests green** in ~4s.
- **Submission zip:** `submission/vt_cos_team_builder.zip` contains source,
  README, `.env.example`, docs, tests, `data/authors.csv`,
  `data/publications.csv`, `data/grants/active.json`, and the pre-warmed
  `data/llm_cache/` entry for opp 363268. It excludes `.env`, `.git`,
  `__pycache__`, and the 387 MB `data/chroma/` (auto-rebuilds on first build).
  A clean unzip dry run passed the test suite and booted the app headless
  without keys (browser view).
- **Manual remaining step:** record the OBS run-through (refresh + filter →
  build team on opp 363268 → follow-up revision). Recommended example
  follow-up: "Replace the PI with a full professor whose evidence covers data
  governance." The cache makes the initial build return instantly with 0 LLM
  calls; the revision makes 2.
- **Elsevier full-corpus gate:** not run end to end in this session — it needs
  campus-network entitlement and hundreds of paginated searches. The
  one-author live smoke test passed, the EID/author cache and source
  fingerprint contracts are covered offline, and the submitted default remains
  `csv`.

**Open items / gotchas:**

- **OBS video** is the only unfinished deliverable. Use opp `363268`
  (cross-department: Biological Sciences, Statistics, Geosciences, Biological
  Systems Engineering) and show refresh/filter → select → Build a team →
  follow-up revision. The shipped `data/llm_cache/` entry makes the first
  build a 0-call cache hit; the revision makes 2 calls.
- `data/` is gitignored (data ships only in the zip), so docs/source are the
  tracked changes; the zip itself lives under `submission/` and is not
  committed.
- The browser table relies on `st.dataframe` row selection; if a future
  Streamlit version changes that widget API, re-run
  `tests/test_chat.py`/manual click-through before recording.
- Optional production verification remains: set `data_source: elsevier`,
  rebuild the Chroma index (the `corpus.sha256` sidecar will force it),
  confirm the corpus size, run one team build, then restore
  `data_source: csv`. It needs campus-network entitlement and a large API
  budget; the offline contract already proves 0-call cache hits and
  source-fingerprint invalidation.

## Contingency drop order (if time runs short)

1. Cross-encoder re-rank (RRF fusion alone suffices)
2. Elsevier impl (keep FacultySource protocol + CSV, document the switch in README)
3. Test breadth (keep scoring-determinism + verify-rejection tests)

## Final acceptance gates (before zip)

- Reproducibility (two-tier, decision 10): same prompt served again →
  bit-identical team + 0 LLM calls; cache-cleared regeneration → identical
  role structure and people overlap ≥ n−2.
- Injected fake author/publication → rejected by verification stage.
- Refresh ×2 → closed solicitation archived, never displayed.
- Launch → proposed team in ≤5 user interactions, spinner on every API/LLM call.
- Browser → chatbot handoff: no re-fetch, no data loss (session-state object).
- Elsevier prod path: offline fake-transport tests pass; a repeat
  `elsevier` load hits the EID/author cache with 0 API calls; switching
  `csv`→`elsevier` forces a ChromaDB rebuild via the corpus hash; an
  on-campus run builds a team end to end.
- README: exact run instructions, expected `.env` keys, design justification
  (ChromaDB vector RAG + hybrid search + re-ranking + MMR author diversity).
