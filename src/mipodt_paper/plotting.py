"""Matplotlib style shared by every figure so panels look consistent."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Nature-style single/double column widths in inches.
SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.2

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # editable text in Illustrator
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_paper_style() -> None:
    mpl.rcParams.update(PAPER_RC)


def savefig(fig: plt.Figure, out_dir: Path, stem: str, formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Save ``fig`` as ``<out_dir>/<stem>.<fmt>`` for every format and return the paths."""
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        paths.append(path)
    return paths
