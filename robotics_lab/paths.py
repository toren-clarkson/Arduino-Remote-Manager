"""Paths relative to this checkout, independent of the current working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "models"
RUNS = PROJECT_ROOT / "runs"
TAGS = PROJECT_ROOT / "assets" / "tags"
