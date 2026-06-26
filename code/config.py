"""
config.py — Central configuration for the Multi-Modal Evidence Review pipeline.

Single source of truth for:
  * Model tiers (workhorse vs. escalation) and generation parameters.
  * Confidence-gated escalation thresholds.
  * Filesystem paths to the dataset, images, and output.
  * Secret resolution — API keys are read from ENVIRONMENT VARIABLES ONLY.

SECURITY CONTRACT (AGENTS.md §6.2):
  - Secrets are NEVER hardcoded here. `ANTHROPIC_API_KEY` (required) and
    `GEMINI_API_KEY` (optional, only for the future hybrid path) are resolved
    at call time from the process environment, optionally hydrated from a local
    `.env` file that is git-ignored. If python-dotenv is not installed, the
    real environment is still honored.

No model-calling logic lives here. This module only describes *how* to call.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# .env hydration (optional, never required). Real env vars always win.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    # Load code/.env if present; does not override already-set env vars.
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    # python-dotenv is optional; absence is not an error.
    pass


# ---------------------------------------------------------------------------
# Model tiers — trust-tiered, confidence-gated escalation
# ---------------------------------------------------------------------------
# WORKHORSE handles every claim by default (fast, cheap, high-volume).
# ESCALATION is invoked only when the workhorse's self-reported confidence
# falls below ESCALATION_CONFIDENCE_THRESHOLD, or on a hard verifier conflict.
WORKHORSE_MODEL: str = "claude-sonnet-4-6"
ESCALATION_MODEL: str = "claude-opus-4-8"

# Generation parameters. temperature=0 keeps the pipeline as deterministic as
# the API allows (AGENTS.md §6.2 "deterministic where possible").
TEMPERATURE: float = 0.0
PERCEPTION_MAX_TOKENS: int = 1024
RECONCILIATION_MAX_TOKENS: int = 1536

# Confidence gating: reconciliation returns a 0.0–1.0 confidence. At or below
# this threshold the claim is re-run on the escalation tier before the
# deterministic verifier finalizes the row.
ESCALATION_CONFIDENCE_THRESHOLD: float = 0.55

# Prompt caching: attach cache_control to the static SYSTEM blocks so the long
# instructions are billed once, not per-image. (Wired in at the call sites.)
ENABLE_PROMPT_CACHING: bool = True

# Optional hybrid path (disabled until explicitly enabled). When True, the
# pipeline may consult Gemini as a second perception opinion. GEMINI_API_KEY is
# only consulted in this mode.
ENABLE_GEMINI_HYBRID: bool = False
GEMINI_MODEL: str = "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CODE_DIR: Path = REPO_ROOT / "code"
DATASET_DIR: Path = REPO_ROOT / "dataset"

SAMPLE_CLAIMS_CSV: Path = DATASET_DIR / "sample_claims.csv"
CLAIMS_CSV: Path = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV: Path = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV: Path = DATASET_DIR / "evidence_requirements.csv"
IMAGES_DIR: Path = DATASET_DIR / "images"

# Final predictions for dataset/claims.csv (problem_statement.md §Required output).
OUTPUT_CSV: Path = REPO_ROOT / "output.csv"

# ---------------------------------------------------------------------------
# Secret resolution — environment only, resolved lazily
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY_ENV: str = "ANTHROPIC_API_KEY"
GEMINI_API_KEY_ENV: str = "GEMINI_API_KEY"


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key from the environment, or raise.

    Never falls back to a hardcoded value. Call this at client-construction
    time, not at import time, so tooling can import config without secrets set.
    """
    key = os.environ.get(ANTHROPIC_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{ANTHROPIC_API_KEY_ENV} is not set. Export it or place it in "
            f"code/.env (see code/.env.example). Secrets are read from the "
            f"environment only and must never be hardcoded."
        )
    return key


def get_gemini_api_key() -> str:
    """Return the optional Gemini API key. Only meaningful when the hybrid
    path is enabled; raises if requested but unset."""
    key = os.environ.get(GEMINI_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{GEMINI_API_KEY_ENV} is not set but the Gemini hybrid path was "
            f"requested. Export it or place it in code/.env."
        )
    return key
