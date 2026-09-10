"""Reproduce Supplementary Fig. 2 (MIP-ODT noise STD without MIR irradiation).

Usage:
    uv run figures/suppfig2_noise_std/run.py --validate-only
    uv run figures/suppfig2_noise_std/run.py
    uv run figures/suppfig2_noise_std/run.py --compute-std   # from the MIR-off volumes
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import yaml

from mipodt_paper import figure_output_dir
from mipodt_paper.config import load_config
from mipodt_paper.noise import Region, region_means, voxel_std, write_region_means_csv
from mipodt_paper.paths import DATA_DIR, data_path
from mipodt_paper.zslices import ZSliceSpec, ZSliceStyle, render_z_slices

OUT = figure_output_dir(__file__)
CFG = load_config(__file__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--compute-std",
        action="store_true",
        help="recompute the noise STD from data/suppfig2_noise_std/mipodt/ instead of loading the distributed NPY",
    )
    parser.add_argument(
        "--std-output",
        type=Path,
        help="where --compute-std writes std_result.npy (default: output/std_result.npy)",
    )
    parser.add_argument("--validate-only", action="store_true", help="validate inputs without writing outputs")
    return parser.parse_args()


def _expected_shape() -> tuple[int, ...]:
    return tuple(int(value) for value in CFG["data"]["expected_shape"])


def _regions() -> list[Region]:
    return [Region.from_mapping(raw) for raw in CFG["noise_region"]["regions"]]


def _mipodt_volume_paths() -> list[Path]:
    settings = CFG["data"]["mipodt"]
    directory = data_path(settings["directory"])
    paths = sorted(directory.glob(settings["filename_glob"]))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least two MIR-off volumes ({settings['filename_glob']}) in {directory}")
    return paths


def load_noise_std() -> tuple[np.ndarray, Path]:
    """Memory-map the distributed noise STD volume and validate its layout."""
    path = data_path(CFG["data"]["noise_std"]["path"])
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    if volume.shape != _expected_shape():
        raise ValueError(f"Expected {_expected_shape()}, got {volume.shape}: {path}")
    if volume.dtype != np.float32:
        raise ValueError(f"Expected float32, got {volume.dtype}: {path}")
    return volume, path


def compute_noise_std(output_path: Path) -> tuple[np.ndarray, Path]:
    """Recompute the voxel-wise noise STD from the MIR-off MIP-ODT volumes."""
    paths = _mipodt_volume_paths()
    dtype = np.dtype(CFG["data"]["mipodt"]["accumulation_dtype"]).type
    print(f"Computing voxel STD over {len(paths)} volumes from {paths[0].parent}")
    std, count = voxel_std(
        paths,
        expected_shape=_expected_shape(),
        dtype=dtype,
        progress=lambda path: print(f"  {path.name}"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, std, allow_pickle=False)
    print(f"Saved {output_path} from {count} volumes")
    return std, output_path


def _z_slice_spec(input_label: str) -> tuple[ZSliceSpec, ZSliceStyle]:
    settings = CFG["z_slices"]
    font = settings["font"]
    spec = ZSliceSpec(
        input_label=input_label,
        z_indices=tuple(int(value) for value in settings["z_indices"]),
        crop_xy=tuple(int(value) for value in settings["crop_xy"]),
        cmap=str(settings["cmap"]),
        vmin=None if settings["vmin"] is None else float(settings["vmin"]),
        vmax=None if settings["vmax"] is None else float(settings["vmax"]),
        title=str(settings["title"]),
        colorbar_label=str(settings["colorbar_label"]),
        colorbar_scientific=bool(settings["colorbar_scientific"]),
        output_prefix=str(settings["output_prefix"]),
    )
    style = ZSliceStyle(
        pixel_size_xy_nm=float(settings["pixel_size_xy_nm"]),
        pixel_size_z_nm=float(settings["pixel_size_z_nm"]),
        figsize_width=float(settings["figsize_width"]),
        figsize_height=float(settings["figsize_height"]),
        dpi=int(settings["dpi"]),
        scalebar_um=float(settings["scalebar_um"]),
        scalebar_color=str(settings["scalebar_color"]),
        scalebar_linewidth=float(settings["scalebar_linewidth"]),
        font_family=str(font["family"]),
        size_title=int(font["size_title"]),
        size_colorbar_label=int(font["size_colorbar_label"]),
        size_colorbar_tick=int(font["size_colorbar_tick"]),
        size_scalebar=int(font["size_scalebar"]),
    )
    return spec, style


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(std_path: Path, computed: bool, rows: list[dict[str, Any]], outputs: list[Path]) -> Path:
    manifest = {
        "figure": "suppfig2_noise_std",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "noise_std": {
            "path": str(std_path),
            "sha256": _sha256(std_path),
            "source": "computed from MIR-off volumes" if computed else "distributed array",
            "expected_shape": list(_expected_shape()),
        },
        "noise_region_mean": rows,
        "outputs": [str(path.relative_to(OUT)) for path in outputs],
    }
    path = OUT / "run_manifest.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)
    return path


def main() -> None:
    args = parse_args()
    if args.validate_only:
        if args.compute_std:
            paths = _mipodt_volume_paths()
            for path in paths:
                volume = np.load(path, mmap_mode="r", allow_pickle=False)
                if volume.shape != _expected_shape():
                    raise ValueError(f"Expected {_expected_shape()}, got {volume.shape}: {path}")
            print(f"Validated {len(paths)} MIR-off volumes under {paths[0].parent}")
        else:
            _, path = load_noise_std()
            print(f"Validated {path}")
        for region in _regions():
            region.validate(_expected_shape())
        print("Input validation completed; no outputs were written")
        return

    if args.compute_std:
        std_path = args.std_output or (OUT / "std_result.npy")
        std, std_path = compute_noise_std(std_path)
    else:
        std, std_path = load_noise_std()

    rows = region_means(std, _regions())
    csv_path = OUT / CFG["noise_region"]["output_csv"]
    write_region_means_csv(csv_path, rows)
    for row in rows:
        print(f"{row['name']}: mean noise STD = {row['mean']:.6e} RI over {row['count']} voxels")

    try:
        input_label = str(std_path.relative_to(DATA_DIR))
    except ValueError:
        input_label = str(std_path)
    spec, style = _z_slice_spec(input_label)
    outputs = render_z_slices(std, spec, style, OUT)
    outputs.append(csv_path)
    manifest = _write_manifest(std_path, args.compute_std, rows, outputs)
    outputs.append(manifest)
    print(f"saved individual outputs under {OUT}")


if __name__ == "__main__":
    main()
