"""Reproduce Fig. 2: Spatial resolution and refractive-index accuracy.

Usage:
    uv run figures/fig2_resolution/run.py
    uv run figures/fig2_resolution/run.py --plot-only
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from mipodt_paper import apply_paper_style, figure_output_dir, savefig
from mipodt_paper.config import load_config
from mipodt_paper.paths import data_path
from mipodt_paper.plotting import SINGLE_COLUMN_IN

OUT = figure_output_dir(__file__)
CFG = load_config(__file__)


def compute() -> dict[str, np.ndarray]:
    """Heavy part: load raw data, reconstruct, and cache the arrays needed to plot."""
    raw = data_path(CFG["data"]["raw"])  # noqa: F841  # replace with actual loading
    result = {"example": np.zeros((64, 64))}
    np.savez_compressed(OUT / "cache.npz", **result)
    return result


def plot(result: dict[str, np.ndarray]) -> plt.Figure:
    """Light part: turn cached arrays into the final panel(s)."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, SINGLE_COLUMN_IN))
    ax.imshow(result["example"], vmin=CFG["plot"]["vmin"], vmax=CFG["plot"]["vmax"], cmap="gray")
    ax.set_axis_off()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="skip compute() and reuse output/cache.npz")
    args = parser.parse_args()

    if args.plot_only:
        result = dict(np.load(OUT / "cache.npz"))
    else:
        result = compute()

    fig = plot(result)
    for p in savefig(fig, OUT, CFG["figure_id"]):
        print(f"saved {p}")


if __name__ == "__main__":
    main()
