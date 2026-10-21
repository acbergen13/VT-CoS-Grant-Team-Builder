"""Load config/settings.yaml. All app behavior keys off this file."""

from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> dict:
    """Read settings.yaml once per process. Cached across Streamlit reruns."""
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    weights = settings["scoring"]["weights"]
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"scoring weights must sum to 1.0, got {weights} "
            "(req 12: weights are fixed and deterministic)"
        )
    return settings
