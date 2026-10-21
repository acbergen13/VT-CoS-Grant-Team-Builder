"""Session-state schemas and the shared browser->chatbot handoff object.

The two Streamlit views communicate exclusively through st.session_state
(req 24): the browser packages the selected solicitation into one structured
Solicitation object; the chatbot reads that same object every turn, so nothing
is re-fetched or re-entered.
"""

from dataclasses import dataclass, field, fields


@dataclass
class Solicitation:
    """Structured solicitation handed from browser to chatbot.

    Listing fields come from the grants.gov search2 result; the richer detail
    fields (synopsis, eligibility, award amounts, funding instrument) are
    filled from fetchOpportunity once the user selects an opportunity
    (see gt/grants/client.py for the pinned API schemas). Dates are ISO
    "YYYY-MM-DD" strings or None; money fields are floats or None.
    """

    opp_id: str
    number: str = ""
    title: str = ""
    status: str = ""  # "posted" | "forecasted"
    doc_type: str = ""  # "synopsis" | "forecast" | ...
    agency: str = ""
    agency_code: str = ""
    open_date: str | None = None
    close_date: str | None = None  # None = open-ended / not yet set
    award_ceiling: float | None = None
    award_floor: float | None = None
    estimated_funding: float | None = None
    number_of_awards: int | None = None
    cost_sharing: bool | None = None
    funding_instrument: str = ""  # e.g. "Grant"
    category: str = ""  # e.g. "Discretionary"
    funding_categories: str = ""  # e.g. "Health"
    eligibility: str = ""  # applicant eligibility text (detail only)
    synopsis: str = ""  # solicitation description, HTML-stripped (detail only)
    cfda: list[str] = field(default_factory=list)
    detail_loaded: bool = False  # fetchOpportunity detail applied?
    raw: dict = field(default_factory=dict)  # last raw API payload

    @property
    def detail_url(self) -> str:
        """Human-readable grants.gov page for this opportunity."""
        return f"https://www.grants.gov/search-results-detail/{self.opp_id}"

    @classmethod
    def from_dict(cls, record: dict) -> "Solicitation":
        """Rebuild from a stored record (active.json), ignoring unknown keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in record.items() if k in names})


@dataclass
class TeamMember:
    """One proposed faculty member with role, score, and traceable evidence."""

    scopus_id: str
    name: str
    department: str = ""
    rank: str = ""
    role: str = ""
    score: float = 0.0
    # Component breakdown from the deterministic scorer, e.g.
    # {"semantic": 0.71, "h5": 0.42, "citations": 0.35} — shown in the UI so
    # the ranking is explainable (req 27).
    score_components: dict = field(default_factory=dict)
    # Publication EIDs backing the rationale; each must resolve to a real
    # publication in the indexed data (chain-of-verification, req 18).
    evidence_eids: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class Team:
    """A proposed team. User critiques translate into slot operations that
    produce a new revision — the LLM never redesigns the whole team."""

    solicitation_opp_id: str
    members: list[TeamMember] = field(default_factory=list)
    revision: int = 0
    # Retrieval query that produced this team; revision turns re-run the
    # same deterministic pipeline so the candidate pool is identical.
    query: str = ""


@dataclass
class ChatTurn:
    """One chatbot message. role is "user" or "assistant"."""

    role: str
    content: str


# Session-state keys. "solicitation" is the browser->chatbot handoff object
# (req 24); "team"/"team_history" hold the current proposal and its revisions;
# "opps"/"last_refresh_result" cache the local grants DB listing in the
# browser view so filters/sorts don't re-read the JSON file every rerun.
DEFAULT_SESSION: dict = {
    "view": "browser",
    "solicitation": None,
    "team": None,
    "team_history": [],
    "messages": [],
    "opps": None,
    "last_refresh_result": None,
    # Stage 5 chat state. Heavy resources are lazy session-scoped caches, so
    # the browser view never loads the corpus/LLM and reruns stay fast.
    "pipeline": None,
    "arc_client": None,
    "builder": None,
    "chat_build_attempted": False,
    "chat_error": "",
    "last_build_result": None,
}


def init_session_state() -> None:
    """Seed st.session_state with defaults on first run (idempotent)."""
    import streamlit as st

    for key, default in DEFAULT_SESSION.items():
        if key not in st.session_state:
            st.session_state[key] = default
