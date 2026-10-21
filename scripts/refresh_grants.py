"""CLI: harvest open grants.gov solicitations into the local JSON store.

Usage (from the repo root, in the ads_5984 env):
    python scripts/refresh_grants.py [--data-dir PATH]

Exit code 0 on success, 1 on a user-facing grants.gov failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as "python scripts/refresh_grants.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gt.grants.client import GrantsClient, GrantsClientError  # noqa: E402
from gt.grants.service import run_refresh  # noqa: E402
from gt.grants.store import GrantStore, default_store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Harvest open (forecasted|posted) grants.gov solicitations."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Store directory (default: data/grants under the repo root).",
    )
    args = parser.parse_args(argv)
    store = GrantStore(args.data_dir) if args.data_dir else default_store()

    print("Harvesting open solicitations from grants.gov (forecasted|posted)...")
    try:
        result = run_refresh(GrantsClient(), store)
    except GrantsClientError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"harvested          : {result.harvested} open opportunities")
    print(f"active             : {result.active_now}")
    print(f"newly added        : {len(result.newly_added)}")
    print(f"updated            : {len(result.updated)}")
    print(f"newly closed       : {len(result.newly_closed)} -> archived (never shown)")
    print(f"reopened           : {len(result.reopened)}")
    print(f"archive total      : {result.archived_total}")
    print(f"store              : {store.active_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
