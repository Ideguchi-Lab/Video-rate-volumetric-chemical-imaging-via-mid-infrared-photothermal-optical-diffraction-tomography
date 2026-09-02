"""Tiny YAML config loader used by every ``figures/*/run.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(figure_file: str | Path, name: str = "config.yaml") -> dict[str, Any]:
    """Load ``config.yaml`` that sits next to the calling ``run.py``."""
    path = Path(figure_file).resolve().parent / name
    with path.open() as f:
        return yaml.safe_load(f) or {}
