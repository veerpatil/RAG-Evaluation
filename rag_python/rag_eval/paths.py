"""Shared path helpers for the rag_python pipeline."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# rag_python/ (project root for this uv workspace)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Prefer local .env, then fall back to repo-root .env
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(PROJECT_DIR.parent / ".env")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
