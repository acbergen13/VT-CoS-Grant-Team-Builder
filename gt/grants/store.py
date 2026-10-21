"""Local grants database: active open solicitations + an archive of closed.

Files (under data/grants/, gitignored):
- active.json  {"harvested_at": iso, "harvest_count": int, "opportunities": [...]}
- archive.json {"opportunities": [...]}   (each record carries "archived_at")

Refresh semantics (per the architecture doc): search2 returns only OPEN
opportunities (oppStatuses=forecasted|posted), so any active record absent
from a completed harvest has closed since the last refresh -> it moves to the
archive and is never shown again (reqs 2, 3, 6). Transactionality is
structural: refresh() receives the complete harvest list, and the orchestrator
(gt.grants.service.run_refresh) only calls it after a successful harvest —
a failed or partial harvest can therefore never trigger archiving.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from gt.config import REPO_ROOT
from gt.state import Solicitation

ACTIVE_FILE = "active.json"
ARCHIVE_FILE = "archive.json"


@dataclass
class RefreshResult:
    """Outcome of one refresh, surfaced in the UI and CLI."""

    harvested: int
    active_now: int
    archived_total: int
    newly_added: list[str]
    newly_closed: list[str]
    updated: list[str]
    reopened: list[str]


def _opp_sort_key(record: dict) -> int:
    opp_id = str(record.get("opp_id", ""))
    return int(opp_id) if opp_id.isdigit() else 0


class GrantStore:
    """JSON-file store. Path-injected for testing (tmp_path)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.active_path = self.root / ACTIVE_FILE
        self.archive_path = self.root / ARCHIVE_FILE

    # -- Low-level file handling --------------------------------------------

    def _read_payload(self, path: Path) -> dict:
        if not path.exists():
            return {"opportunities": []}
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {"opportunities": []}
        payload.setdefault("opportunities", [])
        return payload

    def _write_payload(self, path: Path, payload: dict) -> None:
        """Atomic write: serialize to a temp file, then os.replace it into
        place so a crash can never leave a half-written JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload["opportunities"] = sorted(payload["opportunities"], key=_opp_sort_key)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

    # -- Reads ----------------------------------------------------------------

    def load_active(self) -> list[dict]:
        """Visible (open) opportunities — the archive is never included."""
        return list(self._read_payload(self.active_path)["opportunities"])

    def load_archive(self) -> list[dict]:
        return list(self._read_payload(self.archive_path)["opportunities"])

    def harvested_at(self) -> str | None:
        return self._read_payload(self.active_path).get("harvested_at")

    # -- Writes ---------------------------------------------------------------

    def refresh(self, harvested: list[Solicitation]) -> RefreshResult:
        """Upsert a completed harvest; archive anything now absent (closed)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fresh = {s.opp_id: asdict(s) for s in harvested}
        active = self._read_payload(self.active_path)
        current = {o["opp_id"]: o for o in active["opportunities"]}
        archive = {o["opp_id"]: o for o in self._read_payload(self.archive_path)["opportunities"]}

        # 1) Closed since last refresh: in active, missing from the harvest.
        newly_closed = sorted(set(current) - set(fresh), key=lambda k: int(k) if k.isdigit() else 0)
        for opp_id in newly_closed:
            record = current.pop(opp_id)
            record["archived_at"] = now
            archive[opp_id] = record

        # 2) Reopened: previously archived but open again — restore to active.
        reopened = sorted(set(fresh) & set(archive), key=lambda k: int(k) if k.isdigit() else 0)
        for opp_id in reopened:
            archive.pop(opp_id)

        # 3) Upsert the fresh harvest.
        newly_added = sorted(set(fresh) - set(current), key=lambda k: int(k) if k.isdigit() else 0)
        updated = sorted(set(fresh) & set(current), key=lambda k: int(k) if k.isdigit() else 0)
        current.update(fresh)

        active["opportunities"] = list(current.values())
        active["harvested_at"] = now
        active["harvest_count"] = len(fresh)
        self._write_payload(self.active_path, active)
        self._write_payload(self.archive_path, {"opportunities": list(archive.values())})

        return RefreshResult(
            harvested=len(fresh),
            active_now=len(current),
            archived_total=len(archive),
            newly_added=newly_added,
            newly_closed=newly_closed,
            updated=updated,
            reopened=reopened,
        )

    def save_detail(self, solicitation: Solicitation) -> None:
        """Persist a fetched detail back into active.json so re-selecting an
        opportunity does not re-call the API (no-re-fetch principle)."""
        active = self._read_payload(self.active_path)
        current = {o["opp_id"]: o for o in active["opportunities"]}
        if solicitation.opp_id not in current:
            return  # archived or unknown — nothing to update
        current[solicitation.opp_id] = asdict(solicitation)
        active["opportunities"] = list(current.values())
        self._write_payload(self.active_path, active)


def default_store() -> GrantStore:
    """Production store location: data/grants/ under the repo root."""
    return GrantStore(REPO_ROOT / "data" / "grants")
