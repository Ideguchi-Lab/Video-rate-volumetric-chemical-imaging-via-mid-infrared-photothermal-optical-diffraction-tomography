#!/usr/bin/env python3
"""Reconstruct sequential MIP-ODT volumes from a frozen pairing manifest.

Examples
--------
Validate all inputs without allocating reconstruction arrays::

    uv run scripts/reconstruct_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml --validate-only

Reconstruct one volume, or the complete sequence::

    uv run scripts/reconstruct_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml --volume 0
    uv run scripts/reconstruct_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml
"""

from __future__ import annotations

import argparse
import gc
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mipodt_paper.mipodt import (
    MIPODTRunConfig,
    effective_manifest,
    load_mipodt_config,
    reconstruct_mipodt_volume,
    validate_muscopy_version,
)
from mipodt_paper.pairing import group_pairs_by_volume, load_pairing_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="reconstruction YAML")
    parser.add_argument(
        "--volume",
        type=int,
        action="append",
        help="reconstruct only this zero-based volume ID; repeat to select multiple volumes",
    )
    parser.add_argument("--start-volume", type=int, help="first volume ID to process (inclusive)")
    parser.add_argument("--stop-volume", type=int, help="last volume ID to process (exclusive)")
    parser.add_argument("--validate-only", action="store_true", help="validate config, pairing, and image paths")
    parser.add_argument("--overwrite", action="store_true", help="replace existing reconstructed volumes")
    return parser.parse_args()


def select_volume_ids(
    available: list[int], args: argparse.Namespace, defaults: tuple[int, ...] | None = None
) -> list[int]:
    """Resolve explicit IDs or a half-open range against available volumes."""
    if args.volume and (args.start_volume is not None or args.stop_volume is not None):
        raise ValueError("Use either --volume or --start-volume/--stop-volume, not both")
    if args.volume:
        selected = sorted(set(args.volume))
    elif args.start_volume is None and args.stop_volume is None and defaults is not None:
        selected = list(defaults)
    else:
        start = 0 if args.start_volume is None else args.start_volume
        stop = len(available) if args.stop_volume is None else args.stop_volume
        selected = list(range(start, stop))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Requested volume IDs are unavailable: {missing}")
    if not selected:
        raise ValueError("No volumes selected")
    return selected


def save_npy_atomic(path: Path, array: np.ndarray, *, overwrite: bool) -> None:
    """Save an NPY through a temporary file to avoid partial large outputs."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite to replace it): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_yaml_atomic(path: Path, content: dict[str, Any]) -> None:
    """Atomically update a small YAML run manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False, allow_unicode=True)
    temporary.replace(path)


def make_run_manifest(config: MIPODTRunConfig, selected_ids: list[int]) -> dict[str, Any]:
    manifest = effective_manifest(config)
    manifest.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "software": {
                "muscopy": version("muscopy"),
                "jax": version("jax"),
                "numpy": version("numpy"),
            },
            "selected_volume_ids": selected_ids,
            "volumes": [],
        }
    )
    return manifest


def main() -> None:
    args = parse_args()
    config = load_mipodt_config(args.config)
    validate_muscopy_version()
    pairs = load_pairing_csv(config.pairing_csv, config.image_dir, require_images=True)
    grouped_pairs = group_pairs_by_volume(pairs)
    selected_ids = select_volume_ids(list(grouped_pairs), args, config.default_volume_ids)

    pair_counts = [len(grouped_pairs[volume_id]) for volume_id in grouped_pairs]
    print(f"Dataset: {config.dataset_id}")
    print(f"Validated {len(grouped_pairs)} volumes and {len(pairs)} MIR-ON/OFF pairs")
    print(f"Pairs per volume: min={min(pair_counts)}, max={max(pair_counts)}")
    print(f"Selected volumes: {selected_ids}")
    if args.validate_only:
        print("Input validation completed; reconstruction was not run")
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "run_manifest.yaml"
    manifest = make_run_manifest(config, selected_ids)

    for position, volume_id in enumerate(selected_ids, start=1):
        output_path = config.output_path(volume_id)
        print(
            f"[{position}/{len(selected_ids)}] volume {volume_id}: "
            f"{len(grouped_pairs[volume_id])} pairs -> {output_path}"
        )
        if output_path.exists() and not args.overwrite:
            print("  existing output retained")
            status = "existing"
            shape = list(np.load(output_path, mmap_mode="r").shape)
        else:
            volume = reconstruct_mipodt_volume(config, grouped_pairs[volume_id])
            save_npy_atomic(output_path, volume, overwrite=args.overwrite)
            shape = list(volume.shape)
            del volume
            gc.collect()
            status = "reconstructed"
        manifest["volumes"].append(
            {
                "volume_id": volume_id,
                "pair_count": len(grouped_pairs[volume_id]),
                "output": output_path.name,
                "shape": shape,
                "status": status,
            }
        )
        write_yaml_atomic(manifest_path, manifest)

    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
