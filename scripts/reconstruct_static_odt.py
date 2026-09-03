#!/usr/bin/env python3
"""Reconstruct a real static-ODT delta-n volume from paired TIFF holograms."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from mipodt_paper.mipodt import (
    AlgorithmConfig,
    OpticalConfig,
    load_hologram,
    reconstruct_static_odt_volume,
    validate_muscopy_version,
)
from mipodt_paper.paths import DATA_DIR, RECONSTRUCTION_DIR, REPO_ROOT


def _save_npy_atomic(path: Path, array: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite to replace it): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        import numpy as np

        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_yaml_atomic(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False, allow_unicode=True)
    temporary.replace(path)


def _resolve(root: Path, value: str, key: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Configuration path '{key}' escapes its root: {value}") from exc
    return path


def _load_config(path: Path) -> tuple[dict[str, Any], OpticalConfig, AlgorithmConfig, Path, Path, Path]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported or missing schema_version; expected 1")
    optical_raw = raw["optical"]
    algorithm_raw = raw["reconstruction"]
    optical = OpticalConfig(
        objective_na=float(optical_raw["objective_na"]),
        illumination_na=float(optical_raw["illumination_na"]),
        wavelength_nm=float(optical_raw["wavelength_nm"]),
        camera_pixel_size_um=float(optical_raw["camera_pixel_size_um"]),
        magnification=float(optical_raw["magnification"]),
        medium_refractive_index=float(optical_raw["medium_refractive_index"]),
        image_size_px=int(optical_raw["image_size_px"]),
        offaxis_center_px=tuple(int(value) for value in optical_raw["offaxis_center_px"]),
    )
    algorithm = AlgorithmConfig(
        approximation=str(algorithm_raw["approximation"]),
        linear_approximation=bool(algorithm_raw["linear_approximation"]),
        hermite_symmetry=bool(algorithm_raw["hermite_symmetry"]),
        edge_size_px=int(algorithm_raw["edge_size_px"]),
        gradient_correction=bool(algorithm_raw["gradient_correction"]),
        offset_regions=algorithm_raw["offset_regions"],
        use_skimage_unwrap=bool(algorithm_raw["use_skimage_unwrap"]),
        ewald_embedding_mode=str(algorithm_raw["ewald_embedding_mode"]),
        integer_precision_bits=int(algorithm_raw["integer_precision_bits"]),
        float_precision_bits=int(algorithm_raw["float_precision_bits"]),
        verbose=bool(algorithm_raw.get("verbose", False)),
    )
    algorithm.validate_paper_profile()
    pairing_csv = _resolve(REPO_ROOT, raw["input"]["pairing_csv"], "input.pairing_csv")
    image_dir = _resolve(DATA_DIR, raw["input"]["image_dir"], "input.image_dir")
    output_path = _resolve(RECONSTRUCTION_DIR, raw["output"]["path"], "output.path")
    return raw, optical, algorithm, pairing_csv, image_dir, output_path


def _load_pairs(pairing_csv: Path, image_dir: Path) -> tuple[list[Path], list[Path]]:
    with pairing_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"angle_index", "sample_path", "reference_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Static pairing CSV must contain {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Static pairing CSV is empty")
    indices = [int(row["angle_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("Static pairing CSV contains duplicate angle_index values")
    sample_paths = [_resolve(image_dir, row["sample_path"], "sample_path") for row in rows]
    reference_paths = [_resolve(image_dir, row["reference_path"], "reference_path") for row in rows]
    missing = [path for path in sample_paths + reference_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Static ODT holograms are missing: {', '.join(path.name for path in missing)}")
    return sample_paths, reference_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="static ODT reconstruction YAML")
    parser.add_argument("--validate-only", action="store_true", help="validate configuration and TIFF inputs only")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing reconstruction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, optical, algorithm, pairing_csv, image_dir, output_path = _load_config(args.config.resolve())
    validate_muscopy_version()
    sample_paths, reference_paths = _load_pairs(pairing_csv, image_dir)
    for path in sample_paths + reference_paths:
        load_hologram(path, optical.image_size_px)
    print(f"Validated {len(sample_paths)} static ODT sample/reference pairs")
    if args.validate_only:
        return
    volume = reconstruct_static_odt_volume(optical, algorithm, sample_paths, reference_paths)
    _save_npy_atomic(output_path, volume, overwrite=args.overwrite)
    manifest = {
        "schema_version": 1,
        "dataset_id": raw["dataset_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quantity": "delta_n = n_sample - n_medium",
        "axis_order": "axis0, axis1, z",
        "dtype": "float32",
        "shape": list(volume.shape),
        "pair_count": len(sample_paths),
        "configuration": args.config.name,
        "output": output_path.name,
        "software": {"muscopy": version("muscopy")},
    }
    _write_yaml_atomic(output_path.with_name("run_manifest.yaml"), manifest)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
