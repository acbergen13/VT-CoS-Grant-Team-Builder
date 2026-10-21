"""VT ARC LLM client — the only module that reads VT_ARC_* env keys (reqs 16, 20).

OpenAI-compatible schema over https://llm-api.arc.vt.edu/api/v1. Keys live in
.env (never hardcoded, never committed). Every call carries the config
temperature and a fixed seed (req 19): the endpoint's vLLM backend honors
per-request seeds, so sampling is deterministic on top of temp 0.1.

Error contract (req 23): three typed exceptions with user-presentable
messages — ArcConfigError (missing .env keys), ArcUnavailableError (network /
timeout / server errors, safe to retry), ArcResponseError (model returned no
usable content, e.g. the token budget was consumed by reasoning).
"""

from __future__ import annotations

import os
from typing import Iterable

from gt.config import REPO_ROOT, load_settings

REQUIRED_ENV_KEYS = ("VT_ARC_API_KEY", "VT_ARC_BASE_URL", "VT_ARC_MODEL")


class ArcConfigError(RuntimeError):
    """Missing or empty .env keys — the app cannot talk to the LLM at all."""


class ArcUnavailableError(RuntimeError):
    """Endpoint unreachable / timed out / returned a server error (req 23)."""


class ArcResponseError(RuntimeError):
    """Model returned no usable content (empty reply or token-limit cutoff)."""


def load_arc_env() -> dict[str, str]:
    """Load .env from the repo root (no-op when already loaded)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:  # pragma: no cover - dotenv is installed in ads_5984
        pass
    return {k: os.environ.get(k, "") for k in REQUIRED_ENV_KEYS}


class ArcClient:
    """Thin wrapper around the OpenAI-compatible ARC chat endpoint."""

    def __init__(self, settings: dict | None = None) -> None:
        cfg = (settings or load_settings())["llm"]
        env = load_arc_env()
        missing = [k for k, v in env.items() if not v.strip()]
        if missing:
            raise ArcConfigError(
                f"Missing .env key(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in "
                "(VT_ARC_API_KEY, VT_ARC_BASE_URL, VT_ARC_MODEL)."
            )
        self._model = env["VT_ARC_MODEL"].strip()
        self._temperature = float(cfg["temperature"])
        self._seed = int(cfg["seed"])
        self._max_tokens = int(cfg["max_output_tokens"])

        from openai import OpenAI

        self._client = OpenAI(
            base_url=env["VT_ARC_BASE_URL"].strip(),
            api_key=env["VT_ARC_API_KEY"].strip(),
            timeout=float(cfg["timeout_seconds"]),
            max_retries=2,
        )
        # Req 21 visibility: every completion is counted; the UI shows the
        # total per team build so call-frequency discipline is observable.
        self.calls = 0

    @property
    def model(self) -> str:
        """Served model id — part of the build-cache key (different model ->
        different plan)."""
        return self._model

    def complete(
        self,
        messages: Iterable[dict],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """One chat completion -> the message content string.

        Deterministic by default: config temperature/seed unless overridden
        (tests override nothing; nothing else should either).
        """
        self.calls += 1
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                temperature=self._temperature if temperature is None else temperature,
                seed=self._seed if seed is None else seed,
                max_tokens=self._max_tokens if max_tokens is None else max_tokens,
            )
        except Exception as exc:  # openai raises many exception types
            raise ArcUnavailableError(
                f"The VT ARC LLM endpoint could not be reached or failed "
                f"({type(exc).__name__}). Check your network and "
                "VT_ARC_BASE_URL in .env, then try again."
            ) from exc

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        if not content:
            detail = "token limit reached before any visible output" if (
                choice.finish_reason == "length"
            ) else "the model returned an empty response"
            raise ArcResponseError(
                f"The LLM produced no usable text ({detail}). "
                "Try again or raise llm.max_output_tokens in config/settings.yaml."
            )
        return content
