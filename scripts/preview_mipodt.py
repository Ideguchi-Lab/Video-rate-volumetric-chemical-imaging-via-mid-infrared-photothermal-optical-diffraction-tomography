#!/usr/bin/env python3
"""Render physical-coordinate XY previews from one reconstructed MIP-ODT volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from mipodt_paper.mipodt import load_mipodt_config
from mipodt_paper.preview import plot_volume_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="reconstruction YAML")
    parser.add_argument("--volume", type=int, help="volume ID; defaults to preview.volume_id in YAML")
    parser.add_argument("--z", type=int, action="append", help="z index; repeat for multiple planes")
    parser.add_argument("--show", action="store_true", help="open an interactive Matplotlib window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_mipodt_config(args.config)
    volume_id = config.preview.volume_id if args.volume is None else args.volume
    z_indices = config.preview.z_indices if args.z is None else tuple(args.z)
    volume_path = config.output_path(volume_id)
    if not volume_path.is_file():
        raise FileNotFoundError(
            f"Reconstructed volume not found: {volume_path}. "
            f"Run scripts/reconstruct_mipodt.py with --volume {volume_id} first."
        )

    figure = plot_volume_slices(volume_path, config, z_indices)
    qc_dir = config.output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    for output_format in config.preview.formats:
        output_path = qc_dir / f"mipodt_r_{volume_id:03d}_slices.{output_format}"
        figure.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
