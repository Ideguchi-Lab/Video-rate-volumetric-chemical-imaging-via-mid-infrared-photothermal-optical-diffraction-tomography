"""Repository paths.

Raw data is *not* tracked in git. Set ``MIPODT_DATA_DIR`` to point at the
downloaded dataset; it defaults to ``<repo>/data``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MIPODT_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()
FIGURES_DIR = REPO_ROOT / "figures"


def data_path(*parts: str) -> Path:
    """Return a path under ``DATA_DIR`` and raise a clear error if it is missing."""
    path = DATA_DIR.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset (see data/README.md) or set MIPODT_DATA_DIR."
        )
    return path


def figure_output_dir(figure_file: str | os.PathLike[str]) -> Path:
    """Return ``<figure dir>/output`` for the calling ``run.py`` and create it."""
    out = Path(figure_file).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out
