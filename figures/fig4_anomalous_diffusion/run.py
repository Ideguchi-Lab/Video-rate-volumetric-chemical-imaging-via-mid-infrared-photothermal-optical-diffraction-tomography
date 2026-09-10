#!/usr/bin/env python3
"""Validate data and reproduce the individual Figure 4 source images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import Normalize

from mipodt_paper import figure_output_dir
from mipodt_paper.config import load_config
from mipodt_paper.diffusion import (
    analyze_cached_tracks,
    bootstrap_table,
    component_msd,
    export_normalized_crop,
    extract_detections,
    fit_power_law,
    link_and_analyze,
    particle_track,
    regenerate_probabilities,
    validate_probability_file,
    validate_volume,
    volume_paths,
)
from mipodt_paper.paths import DATA_DIR

OUT = figure_output_dir(__file__)
CFG = load_config(__file__)


def _data_path(value: str) -> Path:
    root = Path(os.path.abspath(DATA_DIR))
    path = Path(os.path.abspath(root / value))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Configured data path escapes MIPODT_DATA_DIR: {value}") from exc
    return path


def _paths() -> dict[str, Path]:
    data = CFG["data"]
    return {
        "volumes": _data_path(data["volume_directory"]),
        "probability": _data_path(data["probability_path"]),
        "labels": _data_path(data["labels_path"]),
        "manifest": _data_path(data["subset_manifest_path"]),
    }


def _frame_ids() -> list[int]:
    settings = CFG["tracking"]["frame_ids"]
    return list(range(int(settings["start"]), int(settings["stop"])))


def _cache_paths() -> dict[str, Path]:
    return {key: OUT / value for key, value in CFG["output"].items()}


def validate_inputs(*, plot_only: bool, recompute_probabilities: bool) -> dict[str, Any]:
    """Validate the selected input files without running analysis."""
    if CFG.get("schema_version") != 1:
        raise ValueError("Unsupported Figure 4 configuration schema")
    paths = _paths()
    cache = _cache_paths()
    if plot_only:
        for key in ("slice_cache", "tracks_cache", "particle_results_cache", "bootstrap_cache"):
            if not cache[key].is_file():
                raise FileNotFoundError(f"Plot cache is missing: {cache[key]}")
        return {"paths": paths, "cache": cache}

    data = CFG["data"]
    slice_volume = paths["volumes"] / data["volume_pattern"].format(int(CFG["mipodt_slices"]["volume_id"]))
    if not slice_volume.is_file():
        raise FileNotFoundError(f"Figure 4 slice volume is missing: {slice_volume}")
    validate_volume(slice_volume, data["expected_volume_shape"])

    if recompute_probabilities:
        resolved_volumes = volume_paths(paths["volumes"], data["volume_pattern"], _frame_ids())
        for path in resolved_volumes:
            validate_volume(path, data["expected_volume_shape"])
        for key in ("labels", "manifest"):
            if not paths[key].is_file():
                raise FileNotFoundError(f"Probability-regeneration input is missing: {paths[key]}")
    else:
        if not paths["probability"].is_file():
            raise FileNotFoundError(f"Supplied probability map is missing: {paths['probability']}")
        validate_probability_file(
            paths["probability"],
            data["probability_dataset"],
            data["probability_shape"],
        )
        resolved_volumes = [slice_volume]
    return {"paths": paths, "cache": cache, "volumes": resolved_volumes, "slice_volume": slice_volume}


def _extract_slices(path: Path) -> dict[str, np.ndarray]:
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        f"slice_z_{z}": np.asarray(volume[:, :, int(z)], dtype=np.float32) for z in CFG["mipodt_slices"]["z_indices"]
    }


def compute(validated: Mapping[str, Any], *, recompute_probabilities: bool) -> dict[str, Any]:
    """Detect particles in probability maps, link tracks with pt3d, and analyze diffusion."""
    paths = validated["paths"]
    data = CFG["data"]
    if recompute_probabilities:
        normalized_path = OUT / "normalized_region_1.h5"
        export_normalized_crop(
            validated["volumes"],
            normalized_path,
            CFG["tracking"]["crop"],
            CFG["tracking"]["normalization_percentiles"],
            CFG["probability_regeneration"]["input_dataset"],
        )
        probability_path = regenerate_probabilities(
            normalized_path,
            paths["labels"],
            paths["manifest"],
            OUT / "region_1_regenerated.h5",
            CFG["probability_regeneration"],
        )
    else:
        probability_path = paths["probability"]

    tracking = CFG["tracking"]
    detections = extract_detections(
        probability_path,
        data["probability_dataset"],
        int(tracking["probability_class_channel"]),
        float(tracking["probability_threshold"]),
        int(tracking["minimum_component_voxels"]),
        int(tracking["component_connectivity"]),
    )
    settings = {**CFG["sampling"], "tracking": tracking, "diffusion": CFG["diffusion"]}
    tracks, analysis = link_and_analyze(detections, settings)
    case_bootstrap = bootstrap_table(analysis, {**settings, "bootstrap": CFG["bootstrap"]})
    selected = CFG["selected_particles"]
    required_ids = {int(selected["trajectory_and_msd"]), int(selected["centroid_shift_and_component_msd"])}
    available_ids = set(analysis.particle_results["particle"].astype(int))
    if not required_ids.issubset(available_ids):
        raise RuntimeError(f"Selected particles are absent after tracking: {sorted(required_ids - available_ids)}")
    if len(analysis.particle_results) != 50:
        raise RuntimeError(f"Expected 50 accepted particles, got {len(analysis.particle_results)}")
    return {
        "slices": _extract_slices(validated["slice_volume"]),
        "detections": detections,
        "tracks": tracks,
        "analysis": analysis,
        "bootstrap": case_bootstrap,
        "probability_path": probability_path,
    }


def write_cache(result: Mapping[str, Any]) -> None:
    """Write portable caches; Python pickle is intentionally not used."""
    cache = _cache_paths()
    np.savez_compressed(cache["slice_cache"], **result["slices"])
    result["detections"].to_csv(cache["detections_cache"], index=False)
    result["tracks"].to_csv(cache["tracks_cache"], index=False)
    result["analysis"].particle_results.to_csv(cache["particle_results_cache"], index=False)
    result["bootstrap"].to_csv(cache["bootstrap_cache"], index=False)


def read_cache() -> dict[str, Any]:
    """Load portable caches and reconstruct the small pt3d analysis object."""
    cache = _cache_paths()
    with np.load(cache["slice_cache"], allow_pickle=False) as archive:
        slices = {key: np.asarray(archive[key]) for key in archive.files}
    tracks = pd.read_csv(cache["tracks_cache"])
    settings = {**CFG["sampling"], "tracking": CFG["tracking"], "diffusion": CFG["diffusion"]}
    analysis = analyze_cached_tracks(tracks, settings)
    return {
        "slices": slices,
        "tracks": tracks,
        "analysis": analysis,
        "bootstrap": pd.read_csv(cache["bootstrap_cache"]),
    }


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for file_format in CFG["plot"]["formats"]:
        path = OUT / f"{stem}.{file_format}"
        kwargs = {"format": file_format} if file_format == "eps" else {"dpi": int(CFG["plot"]["dpi"])}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _trajectory_coordinates(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    voxel = CFG["sampling"]["voxel_size_um"]
    positions = track[["x", "y", "z"]].to_numpy(float)
    centered = positions - positions.mean(axis=0)
    centered *= 1000.0 * np.asarray([voxel["x"], voxel["y"], voxel["z"]])
    time = np.arange(len(centered), dtype=float) * float(CFG["sampling"]["frame_interval_s"])
    return centered, time


def _plot_slices(slices: Mapping[str, np.ndarray]) -> plt.Figure:
    settings = CFG["mipodt_slices"]
    z_indices = [int(value) for value in settings["z_indices"]]
    fig, axes = plt.subplots(len(z_indices), 1, figsize=(4.0, 3.5 * len(z_indices)), squeeze=False)
    pixel = float(settings["xy_pixel_size_um"])
    image = None
    for row, z_index in enumerate(z_indices):
        data = slices[f"slice_z_{z_index}"]
        height, width = data.shape
        ax = axes[row, 0]
        image = ax.imshow(
            data,
            cmap=settings["colormap"],
            vmin=float(settings["display_range"][0]),
            vmax=float(settings["display_range"][1]),
            extent=[0, width * pixel, height * pixel, 0],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        depth = (z_index - z_indices[0]) * float(settings["z_pixel_size_um"])
        ax.set_ylabel(f"z = {depth:.1f} µm")
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bar = float(settings["scalebar_um"])
        x_start = x0 + 0.05 * (x1 - x0)
        y_position = y1 - 0.10 * (y1 - y0)
        ax.plot([x_start, x_start + bar], [y_position, y_position], color="white", linewidth=3)
        ax.text(x_start + bar / 2, y_position - 0.03 * (y1 - y0), f"{bar:.0f} µm", color="white", ha="center")
    fig.subplots_adjust(right=0.85)
    colorbar_axis = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(image, cax=colorbar_axis).set_label("RI Change")
    fig.suptitle("MIP-ODT Slices")
    return fig


def _plot_trajectory(
    track: pd.DataFrame, particle_id: int, boundary_frame: int | None = None
) -> tuple[plt.Figure, float | None]:
    positions, time = _trajectory_coordinates(track)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*positions.T, color=CFG["plot"]["trajectory_line_color"], linewidth=1)
    scatter = ax.scatter(*positions.T, c=time, cmap=CFG["plot"]["trajectory_colormap"], s=30)
    shift = None
    if boundary_frame is not None:
        frames = track["frame"].to_numpy(int)
        first = positions[frames < boundary_frame].mean(axis=0)
        second = positions[frames >= boundary_frame].mean(axis=0)
        shift = float(np.linalg.norm(second - first) / 1000.0)
        ax.scatter(*first, c="#d62728", marker="*", s=260, edgecolors="black", label="Centroid 1")
        ax.scatter(*second, c="#1f77b4", marker="P", s=210, edgecolors="black", label="Centroid 2")
        ax.plot(*np.vstack([first, second]).T, color="black", linestyle="--", label="Centroid shift")
        ax.legend()
        ax.set_title(f"3D Trajectory with Centroids (particle {particle_id})")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    colorbar.set_label("Time (s)")
    if boundary_frame is None:
        scatter.set_clim(0.0, 3.6)
        colorbar.set_ticks([1.2, 2.4, 3.6])
    fig.tight_layout()
    return fig, shift


def _plot_single_msd(table: pd.DataFrame) -> tuple[plt.Figure, dict[str, float | int]]:
    time = table["lag_s"].to_numpy(float)
    msd = table["msd_3d_um2"].to_numpy(float)
    fit = fit_power_law(time, msd, CFG["diffusion"]["fit_range_fraction"], 6.0)
    linear_d = float(np.polyfit(time[:5], msd[:5], 1)[0] / 6.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(time, msd, yerr=table["sem_3d_um2"], fmt="o", markersize=8, alpha=0.7, label="MSD data")
    ax.plot(
        time,
        6.0 * float(fit["D"]) * time ** float(fit["alpha"]),
        color="red",
        linewidth=2,
        label=rf"Fit: $\alpha$={float(fit['alpha']):.2f}, D={float(fit['D']):.2e} $\mu$m$^2$/s",
    )
    ax.plot(time, 6.0 * linear_d * time, color="green", linestyle="--", alpha=0.5, label=r"$\alpha$=1 reference")
    ax.set(xscale="log", yscale="log", xlabel="Lag time (s)", ylabel=r"MSD ($\mu$m$^2$)")
    ax.legend()
    return fig, fit


def _plot_component_msd(table: pd.DataFrame, particle_id: int) -> tuple[plt.Figure, dict[str, dict[str, float | int]]]:
    time = table["lag_s"].to_numpy(float)
    components = {
        "X": ("msd_x_um2", "tab:blue", "o", 2.0),
        "Y": ("msd_y_um2", "tab:green", "o", 2.0),
        "Z": ("msd_z_um2", "tab:red", "o", 2.0),
        "3D": ("msd_3d_um2", "black", "D", 6.0),
    }
    fits = {}
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, (column, color, marker, divisor) in components.items():
        values = table[column].to_numpy(float)
        fit = fit_power_law(time, values, CFG["diffusion"]["fit_range_fraction"], divisor)
        fits[name] = fit
        label = (
            f"{name} MSD (alpha={float(fit['alpha']):.3f})"
            if name != "3D"
            else f"3D total MSD (alpha={float(fit['alpha']):.3f})"
        )
        ax.plot(time, values, color=color, marker=marker, label=label, linewidth=2.5 if name == "3D" else 1.5)
    start = float(CFG["diffusion"]["fit_range_fraction"][0])
    stop = float(CFG["diffusion"]["fit_range_fraction"][1])
    ax.axvspan(
        time[max(0, int((len(time) + 1) * start) - 1)],
        time[min(len(time) - 1, int((len(time) + 1) * stop) - 1)],
        color="gray",
        alpha=0.12,
        label="fit range",
    )
    ax.set(xscale="log", yscale="log", xlabel="Lag time (s)", ylabel=r"MSD ($\mu$m$^2$)")
    ax.set_title(f"Particle {particle_id} component-wise and total 3D MSD, log-log")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    return fig, fits


def _plot_diffusion_scatter(analysis: Any, bootstrap: pd.DataFrame) -> plt.Figure:
    merged = analysis.particle_results[["particle", "D"]].merge(
        bootstrap[["particle", "alpha_case_boot_median"]], on="particle", validate="one_to_one"
    )
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(4.0, 1.4), wspace=0.05)
    scatter_axis = fig.add_subplot(grid[0, 0])
    histogram_axis = fig.add_subplot(grid[0, 1], sharey=scatter_axis)
    color = CFG["plot"]["diffusion_color"]
    scatter_axis.scatter(merged["D"] * 1e3, merged["alpha_case_boot_median"], s=75, color=color)
    histogram_axis.hist(
        merged["alpha_case_boot_median"],
        bins=20,
        orientation="horizontal",
        color=color,
        edgecolor=CFG["plot"]["diffusion_edge_color"],
        alpha=0.9,
    )
    scatter_axis.set_xlabel(r"$D_\alpha$ ($10^{-3}\,\mu$m$^2$/s$^\alpha$)")
    scatter_axis.set_ylabel(r"$\alpha$ (diffusion exponent)")
    histogram_axis.set_xlabel("Count")
    histogram_axis.tick_params(axis="y", labelleft=False)
    scatter_axis.set_xlim(left=0)
    scatter_axis.set_ylim(bottom=-0.025)
    return fig


def _plot_alpha_spatial(analysis: Any, bootstrap: pd.DataFrame) -> plt.Figure:
    alpha = dict(zip(bootstrap["particle"].astype(int), bootstrap["alpha_case_boot_median"], strict=True))
    positions, values = [], []
    for particle_id, trajectory in analysis.positions_um.items():
        valid = trajectory[~np.any(np.isnan(trajectory), axis=1)]
        if len(valid) and int(particle_id) in alpha:
            positions.append(valid.mean(axis=0))
            values.append(alpha[int(particle_id)])
    positions_array = np.asarray(positions)
    values_array = np.asarray(values)
    z, y, x = positions_array.T
    z = 4.4 - z
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=float(values_array.min()), vmax=float(values_array.max()))
    scatter = ax.scatter(
        x,
        y,
        z,
        c=values_array,
        cmap=CFG["plot"]["spatial_colormap"],
        norm=norm,
        s=float(CFG["plot"]["spatial_point_size"]),
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set(
        xlabel=r"X ($\mu$m)", ylabel=r"Y ($\mu$m)", zlabel=r"Z ($\mu$m)", xlim=(0, 37.4), ylim=(0, 37.4), zlim=(0, 4.4)
    )
    ax.set_box_aspect((1, 1, 0.35))
    fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1).set_label(r"Diffusion Exponent ($\alpha$)")
    fig.tight_layout()
    return fig


def plot(result: Mapping[str, Any]) -> list[Path]:
    """Generate the seven independent source figures and their compact tables."""
    outputs = _save(_plot_slices(result["slices"]), "mipodt_z_slices")
    tracks, analysis, bootstrap = result["tracks"], result["analysis"], result["bootstrap"]
    selected = CFG["selected_particles"]
    first_id = int(selected["trajectory_and_msd"])
    second_id = int(selected["centroid_shift_and_component_msd"])
    first_track, second_track = particle_track(tracks, first_id), particle_track(tracks, second_id)

    figure, _ = _plot_trajectory(first_track, first_id)
    outputs.extend(_save(figure, f"particle_{first_id}_trajectory_3d"))
    max_lag = int(len(first_track) * float(CFG["diffusion"]["max_lag_fraction"]))
    first_msd = component_msd(
        first_track, CFG["sampling"]["voxel_size_um"], float(CFG["sampling"]["frame_interval_s"]), max_lag
    )
    first_msd.to_csv(OUT / f"particle_{first_id}_msd.csv", index=False)
    figure, first_fit = _plot_single_msd(first_msd)
    outputs.extend(_save(figure, f"particle_{first_id}_msd_loglog"))

    figure, centroid_shift = _plot_trajectory(second_track, second_id, int(selected["centroid_boundary_frame"]))
    outputs.extend(_save(figure, f"particle_{second_id}_trajectory_centroid_shift"))
    max_lag = int(len(second_track) * float(CFG["diffusion"]["max_lag_fraction"]))
    second_msd = component_msd(
        second_track, CFG["sampling"]["voxel_size_um"], float(CFG["sampling"]["frame_interval_s"]), max_lag
    )
    second_msd.to_csv(OUT / f"particle_{second_id}_component_msd.csv", index=False)
    figure, component_fits = _plot_component_msd(second_msd, second_id)
    outputs.extend(_save(figure, f"particle_{second_id}_msd_xyz3d_loglog"))

    outputs.extend(_save(_plot_diffusion_scatter(analysis, bootstrap), "diffusion_scatter"))
    outputs.extend(_save(_plot_alpha_spatial(analysis, bootstrap), "alpha_spatial"))
    summary = {
        "accepted_particles": int(len(analysis.particle_results)),
        f"particle_{first_id}": {"alpha": float(first_fit["alpha"]), "D_um2_per_s_alpha": float(first_fit["D"])},
        f"particle_{second_id}": {
            "centroid_boundary_frame": int(selected["centroid_boundary_frame"]),
            "centroid_shift_um": centroid_shift,
            "component_alpha": {name: float(values["alpha"]) for name, values in component_fits.items()},
        },
    }
    with _cache_paths()["summary"].open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true", help="Re-plot from portable output caches")
    mode.add_argument("--validate-only", action="store_true", help="Validate selected inputs and stop")
    parser.add_argument(
        "--recompute-probabilities",
        action="store_true",
        help="Regenerate probability maps from reconstructed volumes and public labels before tracking",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validated = validate_inputs(plot_only=args.plot_only, recompute_probabilities=args.recompute_probabilities)
    if args.validate_only:
        print("Figure 4 inputs are valid.")
        return 0
    if args.plot_only:
        result = read_cache()
    else:
        result = compute(validated, recompute_probabilities=args.recompute_probabilities)
        write_cache(result)
    outputs = plot(result)
    print(f"Generated {len(outputs)} figure files in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
