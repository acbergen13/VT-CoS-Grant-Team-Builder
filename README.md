# VT CoS Grant Team Builder

Streamlit app that harvests **open grants.gov solicitations** and builds
**explainable teams of Virginia Tech College of Science faculty** for a
selected solicitation, via a RAG chatbot grounded in faculty publications and
author metrics.

Mid-term project for VT ADS 5984. Binding requirements and architecture:
`docs/architecture/mid_term_project_reqs.qmd`. Build plan and data profile:
`plan.md`.

## What it does

1. **Browse** — pull the latest open solicitations from grants.gov
   (`forecasted|posted` only), filter/sort by type, date, amount; solicitations
   that close are archived locally and never shown.
2. **Build a team** — select a solicitation and click *Build a team*: the
   chatbot retrieves faculty evidence (ChromaDB vector RAG over publication
   abstracts + author metrics) and proposes a team with roles and rationale.
3. **Refine** — critique the proposal conversationally; only the affected
   slots are revised, and every factual claim is verified against source text
   before it reaches you.

> **Build status (Oct 22):** Stages 0–7 complete — grants.gov harvest, local
> store, browser UI, deterministic faculty ranking, hybrid retrieval, the
> verified LLM team-builder, the conversational chat UI, the Elsevier
> production publication source (behind the same `FacultySource` seam, with an
> EID-keyed disk cache), and submission hardening. **CSV remains the default
> run mode**, so the submitted app runs without an Elsevier key. The suite is
> `143` offline tests; the final two-tier reproducibility gate passed on the
> shipped corpus (cached rerun bit-identical with **0 LLM calls**).

## Quick start

```bash
# course conda env — all dependencies already installed
conda activate ads_5984

# configure keys (table below)
cp .env.example .env   # then fill in values

# optional: pre-build the vector index (~50s; the app builds it
# automatically on first team-build if missing)
python scripts/build_index.py

streamlit run main.py
```

> First index build downloads a small local embedding model
> (`BAAI/bge-small-en-v1.5`, ~130 MB) — internet required once.

A team build is automatic after you select a solicitation and click **Build a
team**. The first build retrieves evidence and makes three LLM calls; the chat
UI shows a spinner throughout. Re-opening the same solicitation keeps the
existing team, and cached builds return with zero LLM calls.

### What ships in the submission zip

- All source (`main.py`, `app/`, `gt/`, `scripts/`, `tests/`, `config/`,
  `.streamlit/`), this README, `.env.example`, and the design docs.
- `data/authors.csv`, `data/publications.csv`, the harvested
  `data/grants/active.json`, and a pre-warmed `data/llm_cache/` entry for the
  demo solicitation (`363268`, NSF AI Datasets).
- **No `.env`** (keys are never submitted) and no `data/chroma/`: the index is
  rebuilt automatically from the shipped corpus on the first team build (or
  deliberately with `python scripts/build_index.py`). This keeps the zip small;
  the one-time embedding-model download needs internet.

## Expected `.env` keys

| Key | Required | Purpose |
|---|---|---|
| `VT_ARC_API_KEY` | yes | VT ARC LLM endpoint auth |
| `VT_ARC_BASE_URL` | yes | `https://llm-api.arc.vt.edu/api/v1` |
| `VT_ARC_MODEL` | yes | chat model id; `.env.example` uses `gpt-oss-120b` (thinking models may return empty content after spending the token budget on hidden reasoning) |
| `ELSEVIER_API_KEY` | prod mode | Scopus publication data (csv mode runs without it) |
| `ELSEVIER_BASE_URL` | prod mode | Elsevier API base URL, e.g. `https://api.elsevier.com` |
| `ELSEVIER_INST_TOKEN` | optional | entitled Elsevier content off-campus |

Keys are never hardcoded or committed; `.env` is gitignored (req 20).

## Data

- `data/authors.csv` — 501 rows, ~500 CoS faculty with metrics (H-Index, H5
  Index, Citation Count). Join key: `Scopus ID`.
- `data/publications.csv` — ~21.5k publications (51 MB). Links to authors via
  `Author(s) ID` (semicolon-separated Scopus IDs).
- Both CSVs ship in the submission zip but stay out of git.

### Production mode: Elsevier / Scopus

Set `data_source: elsevier` in `config/settings.yaml` to load publication
metadata and abstracts directly from Elsevier instead of `publications.csv`.
The roster still comes from `authors.csv` (req 4); only the publication source
switches. Required `.env` keys are `ELSEVIER_API_KEY` and
`ELSEVIER_BASE_URL`; `ELSEVIER_INST_TOKEN` is optional and may be needed for
entitled abstracts off campus. Run on the campus network when possible.

- Scopus Search uses `AU-ID(<scopus_id>)`, `view=COMPLETE`, and 0-based
  pagination; `view=COMPLETE` normally includes the abstract, so Abstract
  Retrieval is only a bounded fallback for missing abstracts.
- Results are cached under `data/elsevier_cache/`:
  `publications/<EID>.json` is the canonical record, and
  `authors/<scopus_id>.json` stores that author's EID list. A repeat load hits
  the cache with **zero API calls**; pass `refresh=True` (or delete the cache
  directory) to force a refresh.
- Switching CSV → Elsevier changes publication text even when EIDs match, so
  the index tracks a corpus content hash and forces a rebuild via
  `python scripts/build_index.py`. The LLM build cache is also keyed by the
  data source/corpus fingerprint, so CSV-built teams are never served after
  the switch.
- Abstract fallback is capped and disables itself after repeated
  entitlement/quota failures, so an off-campus run cannot become thousands of
  failed requests.
- `data_source: csv` is the submitted default: the instructor can unzip and
  run without an Elsevier key. The production path is covered by offline
  fake-transport tests plus a live one-author smoke test. A full-corpus
  Elsevier run is optional and needs campus-network entitlement plus a large
  API budget (hundreds of paginated searches); it does not block the submitted
  CSV mode.

**Cleaning rules applied at load time** (`gt/faculty/`, deterministic — same
input, same output, every run):

- Both files start with a UTF-8 BOM; everything parses with
  `encoding="utf-8-sig"`.
- One Scopus ID (`56940337900`) is shared by two people (Iliescu/Driscoll).
  The loader keeps the first row in file order and records the drop — the
  known data-quality caveat, not a heuristic guess.
- Missing H5 Index / H-Index (15% of rows) impute to 0 **before**
  normalization; `Top Journal Quartile %` stays unknown (imputing 0 would
  falsely claim bottom-quartile journals).
- Citation counts are heavily right-skewed (max ≈49.6k vs median ≈526), so
  they are `log1p`-transformed before min-max normalization; H5 uses plain
  min-max.
- Publications are filtered to rows with ≥1 roster author (~50k external
  co-author IDs and 1,425 no-CoS rows never enter the pool) and noise
  document types (Erratum, Editorial, Note, Letter) are excluded — 19,461
  publications survive.
- PI eligibility is a deterministic rank lookup (professorial ranks only;
  postdocs and instructors are never PI) — configured in
  `config/settings.yaml`, never decided by the LLM.

## How faculty are ranked (deterministic)

Team candidates are ranked by a fixed score computed in code, not by the LLM:

```
score = 0.6 × semantic similarity + 0.25 × normalized H5 + 0.15 × normalized citations
```

Weights live in `config/settings.yaml` and are never set or changed by the
model. The ranking is a total order (score, then Scopus ID), so the same
inputs produce bit-identical rankings on every run. Each result carries its
component breakdown (`semantic` / `h5` / `citations`) so the UI can explain
why a member ranked where they did.

## Design justification (retrieval)

A query like *"machine learning for climate"* has to find the right **people**,
not just matching documents, and the architecture doc commits to ChromaDB
vector RAG with hybrid search, re-ranking, and MMR author-level diversity.
The pipeline in `gt/retrieval/pipeline.py` implements that as fixed code
stages — no tool or weight choices are delegated to the LLM:

1. **Hybrid recall (BM25 + vector, fused by RRF).** Lexical BM25 catches
   exact solicitation vocabulary ("sensor", "watershed"); dense vectors
   (`BAAI/bge-small-en-v1.5`, ChromaDB persistent index over 19,461
   publication chunks — one per CoS author × publication) catch paraphrase.
   Reciprocal-rank fusion (k=60) merges both ranked lists without needing
   comparable scores. Chunking at the author×publication grain means every
   retrieved chunk is attributable to a specific faculty member — the unit
   the team builder needs, and the unit verification can check (req 18).
2. **Author-level grouping + MMR re-ranking.** Raw chunk recall is naturally
   dominated by prolific authors: in our live gate the top-10 chunks spanned
   only 5 authors (one owned 6 of 10). Chunks are grouped per author (max
   fused relevance wins), then maximal-marginal-relevance (λ=0.7, config
   `retrieval.mmr_lambda`) selects a diverse candidate pool of authors.
   MMR decides **pool membership only**.
3. **Deterministic ranking.** Pool order comes from the fixed
   0.6/0.25/0.15 score (above) computed in code. The effect is visible in
   the gate: Gel had the top semantic similarity (0.774) but Carey ranked
   first overall (0.772 vs 0.722) on H5/citation strength — the LLM will
   never see, let alone change, this ordering.
4. **Evidence packs.** Each selected author carries their top chunks as
   verbatim `EvidenceChunk` records (title, year, DOI, Scopus link) — the
   grounding material for chain-of-verification and the clickable citations
   in the UI (req 27).

Determinism is structural: every sort tie-breaks on `(-score, scopus_id)`,
scoring is fitted once per roster, and queries are stateless. Measured on
the live index: warm query ~0.03s; repeated runs return bit-identical ids,
scores, and evidence.

## How the chatbot builds and revises teams

The browser passes one structured `Solicitation` object to the chat view via
Streamlit session state (`gt/state.py`), so detail data is never re-fetched or
re-entered (req 24). Entering chat auto-builds the first proposal; every API /
LLM operation runs under a visible status/spinner (req 1). As soon as the
keyword-extraction call returns, the live build status shows the assembled
**optimized retrieval prompt** while the rest of the build continues.

The LLM does not choose membership, roles, scores, or ordering. A build is a
fixed three-call workflow in `gt/llm/team_builder.py`:

1. **Keyword extraction** — the LLM returns technical domains and methods
   (req 14). The retrieval query is assembled in code by
   `canonical_query`: a deterministic solicitation anchor plus sorted,
   lowercased terms. Only the keyword *set* affects retrieval, bounding model
   phrasing drift.
2. **Per-role focuses** — the roles come from `team.base_roles` in
   `config/settings.yaml`; the LLM writes only each role's
   solicitation-specific focus. Membership is then a greedy deterministic
   assignment over the ranked candidate pool (PI eligibility enforced).
3. **Batched rationales** — one call writes all member justifications, which
   are checked by `gt/llm/verify.py` against the member's actual Scopus ID,
   evidence-pack EIDs, and non-empty text. Failed entries regenerate; after
   retries, deterministic score-based rationales are used (reqs 18/27).

A critique turn makes two calls: one translates the critique into validated
slot operations, one rewrites only the changed members' rationales. A code
scope guard drops operations on members the critique did not name, and the
last PI cannot be removed (reqs 13/15).

Reproducibility has two tiers. A cached build (keyed by opportunity, model,
and content; `data/llm_cache/`) returns a bit-identical team and rationales
with **0 LLM calls**. A cache-cleared regeneration keeps the same role
structure and at least n−2 of the same people; residual variance is endpoint
keyword selection, never membership logic (req 19). Degraded fallback output
is never cached.

**Final Stage 7 gate** (shipped CSV corpus + index, opportunity `363268`
"Unlocking Dataset Value for AI-Enabled Scientific Discovery"): two
cache-cleared builds produced the same four-role structure and **3/4** people
overlap (the contract requires ≥ n−2 = 2); the query text drifted only in the
LLM keyword half, as designed. A third cached build made **0 LLM calls** and
was bit-identical to the second, rationales included. The same gate also
verified that an injected fake author/publication is rejected by
`gt/llm/verify.py`, and that browser refresh/detail failures render as
user-facing errors (`tests/test_browser.py`).

The chat UI renders each member's role, deterministic score breakdown,
rationale, and supporting publications. Every publication shows its EID,
DOI/Scopus link, and expandable verbatim indexed text, so claims are
traceable to source (reqs 18/27). Build details expose the canonical query,
candidate-pool size, LLM-call count, and cache/fallback notes (req 21).

## Project structure

```
main.py                 # Streamlit entrypoint
config/settings.yaml    # weights, roles, PI-eligible ranks, retrieval/LLM params
gt/                     # package: state, config
│   └── grants/         #   grants.gov client, JSON store, orchestration
│   └── faculty/        #   roster + publications loading, FacultySource seam,
│                       #   Elsevier production source + EID cache, scoring
│   └── retrieval/      #   chunk index, hybrid search, author MMR, evidence packs
│   └── llm/            #   ARC client, canonical query, team builder, verification, cache
app/                    # Streamlit views: browser.py, chat.py (shared session state)
scripts/                # CLI: refresh_grants.py, build_index.py
tests/                  # pytest (root conftest.py makes gt/ importable)
data/                   # CSVs + grants store + LLM cache in the zip; chroma/ rebuilt locally
```

## Running tests

```bash
conda activate ads_5984
pytest -q
```

The suite is offline (fake grants transport, fake LLM client, temporary
directories). `tests/test_chat.py` and `tests/test_browser.py` also drive the
Streamlit views through `streamlit.testing.v1.AppTest`; no live API calls are
made.

## Troubleshooting

- **`Missing .env key(s)` on the first team build** — copy `.env.example` to
  `.env` and fill `VT_ARC_API_KEY`, `VT_ARC_BASE_URL`, and `VT_ARC_MODEL`.
  The browser view loads without keys; only the LLM build needs them.
- **First build pauses on “may download the embedding model”** — one-time
  ~130 MB HuggingFace download. If the machine is offline, pre-populate the HF
  cache or switch `retrieval.embedding_model` to the ARC `Qwen3-Embedding-4B`
  endpoint.
- **Refresh fails** — grants.gov is external; use a network connection or keep
  the shipped `data/grants/active.json` and skip the refresh. Failed refreshes
  never modify or archive the local store.
- **Empty or truncated LLM replies** — raise `llm.max_output_tokens` in
  `config/settings.yaml`, or use `gpt-oss-120b` (thinking models can spend the
  budget on hidden reasoning).
- **`data/chroma/` absent after unzip** — expected; the first build recreates
  it, or run `python scripts/build_index.py` once.
- **Faculty corpus missing** — `data/authors.csv` and `data/publications.csv`
  must be present at the repo root's `data/` directory.
