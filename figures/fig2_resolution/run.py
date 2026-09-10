"""Reproduce the individual Fig. 2 source outputs from the 2025-12-02 arrays.

The output contract follows ``z-slice-visualizer/results_fig1`` and
``resolution_evaluation/results-fig1`` rather than assembling a new composite.

Usage:
    uv run figures/fig2_resolution/run.py
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from PIL import Image

from mipodt_paper import figure_output_dir
from mipodt_paper.config import load_config
from mipodt_paper.paths import data_path
from mipodt_paper.resolution import GaussianFitResult, ProfileData, extract_profiles, fit_profiles

OUT = figure_output_dir(__file__)
BMP_OUT = OUT / "bmp"
CFG = load_config(__file__)
DIRECTIONS = ("X", "Y", "Z")
DATASETS = ("static_odt", "mipodt")


def load_volume(name: str) -> np.ndarray:
    """Memory-map one distributed reconstruction and validate its layout."""
    path = data_path(CFG["data"][name]["path"])
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(CFG["data"]["expected_shape"])
    if volume.shape != expected_shape:
        raise ValueError(f"Expected {expected_shape}, got {volume.shape}: {path}")
    return volume


def _real_values(data: np.ndarray, selection: tuple[object, object, object], offset: float) -> np.ndarray:
    values = np.asarray(np.real(data[selection]), dtype=np.float32)
    return values + np.float32(offset) if offset else values


def _save_fig_as_bmp(fig: plt.Figure, output_path: Path, *, dpi: int = 300) -> None:
    """Save a Matplotlib figure as the legacy 24-bit BMP output."""
    with BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        with Image.open(buffer) as image:
            image.convert("RGB").save(output_path, format="BMP")


def _add_scalebar(ax: plt.Axes, settings: dict[str, Any]) -> list[Any]:
    """Add the scalebar used by z-slice-visualizer and return its artists."""
    before = set(ax.get_children())
    pixel_size_nm = float(CFG["z_slices"]["pixel_size_xy_nm"])
    length_um = float(CFG["z_slices"]["scalebar_um"])
    length_px = length_um * 1000.0 / pixel_size_nm
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_start = xlim[1] - length_px - (xlim[1] - xlim[0]) * 0.05
    y_position = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot(
        [x_start, x_start + length_px],
        [y_position, y_position],
        color=CFG["z_slices"]["scalebar_color"],
        linewidth=CFG["z_slices"]["scalebar_linewidth"],
        solid_capstyle="butt",
    )
    ax.text(
        x_start + length_px / 2,
        y_position + (ylim[1] - ylim[0]) * 0.02,
        f"{length_um:.0f} µm",
        color=CFG["z_slices"]["scalebar_color"],
        fontsize=CFG["z_slices"]["font"]["size_scalebar"],
        fontfamily=CFG["z_slices"]["font"]["family"],
        ha="center",
        va="bottom",
    )
    return [artist for artist in ax.get_children() if artist not in before]


def _add_z_slice_annotations(ax: plt.Axes, settings: dict[str, Any], z_index: int) -> None:
    x_start, _, y_start, _ = settings["crop_xy"]
    for roi in settings["roi_annotations"]:
        if roi["z_index"] is None or roi["z_index"] == z_index:
            ax.add_patch(
                Rectangle(
                    (roi["y"] - y_start, roi["x"] - x_start),
                    roi["height"],
                    roi["width"],
                    linewidth=roi["linewidth"],
                    edgecolor=roi["color"],
                    facecolor="none",
                )
            )
    for arrow in settings["arrow_annotations"]:
        if arrow["z_index"] is None or arrow["z_index"] == z_index:
            ax.annotate(
                "",
                xy=(arrow["y_end"] - y_start, arrow["x_end"] - x_start),
                xytext=(arrow["y_start"] - y_start, arrow["x_start"] - x_start),
                arrowprops={
                    "arrowstyle": (f"-|>,head_width={arrow['head_width']},head_length={arrow['head_length']}"),
                    "color": arrow["color"],
                    "lw": arrow["linewidth"],
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )


def _save_z_slice_bmp(
    fig: plt.Figure,
    ax: plt.Axes,
    title_artist: Any,
    scalebar_artists: list[Any],
    output_path: Path,
    dpi: int,
) -> None:
    """Save one annotated z slice without its title or scalebar."""
    fig.canvas.draw()
    title_artist.set_visible(False)
    for artist in scalebar_artists:
        artist.set_visible(False)
    try:
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            raise RuntimeError("Could not determine z-slice subplot bounds")
        with BytesIO() as buffer:
            fig.savefig(
                buffer,
                format="png",
                dpi=dpi,
                bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()),
                pad_inches=0,
            )
            buffer.seek(0)
            with Image.open(buffer) as image:
                image.convert("RGB").save(output_path, format="BMP")
    finally:
        title_artist.set_visible(True)
        for artist in scalebar_artists:
            artist.set_visible(True)


def _z_slice_config(name: str) -> dict[str, Any]:
    settings = CFG["z_slices"][name]
    offset = float(CFG["data"][name]["value_offset"])
    return {
        "data_source": {
            "input_file": CFG["data"][name]["path"],
            "z_slice_positions": CFG["z_slices"]["z_indices"],
            "value_offset": offset,
        },
        "layout": {
            "orientation": "horizontal",
            "figsize_width": CFG["z_slices"]["figsize_width"],
            "figsize_height_per_slice": CFG["z_slices"]["figsize_height"],
        },
        "crop": {
            "enabled": True,
            "x_start": settings["crop_xy"][0],
            "x_end": settings["crop_xy"][1],
            "y_start": settings["crop_xy"][2],
            "y_end": settings["crop_xy"][3],
        },
        "roi_annotations": settings["roi_annotations"],
        "arrow_annotations": settings["arrow_annotations"],
        "colormap": {
            "name": settings["cmap"],
            "vmin": settings["vmin_raw"],
            "vmax": settings["vmax_raw"],
        },
        "output": {
            "output_dir": "output",
            "output_prefix": settings["output_prefix"],
            "dpi": CFG["z_slices"]["dpi"],
            "save_png": True,
        },
        "physical_constants": {
            "xy_pixel_size_nm": CFG["z_slices"]["pixel_size_xy_nm"],
            "z_pixel_size_nm": CFG["z_slices"]["pixel_size_z_nm"],
        },
        "font": {
            "family": CFG["z_slices"]["font"]["family"],
            "size_title": CFG["z_slices"]["font"]["size_title"],
            "size_ylabel": 24,
            "size_colorbar_label": CFG["z_slices"]["font"]["size_colorbar_label"],
            "size_colorbar_tick": CFG["z_slices"]["font"]["size_colorbar_tick"],
            "size_scalebar": CFG["z_slices"]["font"]["size_scalebar"],
        },
        "scalebar": {
            "length_um": CFG["z_slices"]["scalebar_um"],
            "color": CFG["z_slices"]["scalebar_color"],
            "linewidth": CFG["z_slices"]["scalebar_linewidth"],
        },
        "figure": {
            "title": settings["title"],
            "colorbar_label": settings["colorbar_label"],
            "colorbar_scientific": settings["colorbar_scientific"],
        },
    }


def create_z_slice_outputs(name: str, volume: np.ndarray) -> None:
    """Create one legacy z-slice figure, EPS, four BMPs, and its config."""
    global_settings = CFG["z_slices"]
    settings = global_settings[name]
    indices = global_settings["z_indices"]
    offset = float(CFG["data"][name]["value_offset"])
    x_start, x_end, y_start, y_end = settings["crop_xy"]
    slices = [
        _real_values(volume, (slice(x_start, x_end), slice(y_start, y_end), z_index), offset) for z_index in indices
    ]
    vmin = float(settings["vmin_raw"]) + offset
    vmax = float(settings["vmax_raw"]) + offset
    width = float(global_settings["figsize_width"]) * len(indices)
    height = float(global_settings["figsize_height"])
    fig, axes = plt.subplots(1, len(indices), figsize=(width, height), squeeze=False)
    axes_flat = list(axes[0])
    titles: list[Any] = []
    scalebars: list[list[Any]] = []
    image = None
    for ax, slice_data, z_index in zip(axes_flat, slices, indices, strict=True):
        image = ax.imshow(slice_data, cmap=settings["cmap"], vmin=vmin, vmax=vmax, origin="lower")
        z_um = (z_index - indices[0]) * float(global_settings["pixel_size_z_nm"]) / 1000.0
        ax.set_title(
            f"Z = {z_um:.1f} µm",
            fontsize=global_settings["font"]["size_title"],
            fontfamily=global_settings["font"]["family"],
        )
        titles.append(ax.title)
        ax.set_xticks([])
        ax.set_yticks([])
        scalebars.append(_add_scalebar(ax, settings))
        _add_z_slice_annotations(ax, settings, z_index)

    if image is None:
        raise RuntimeError("No z slices were configured")
    colorbar = fig.colorbar(image, ax=axes_flat, shrink=0.8, pad=0.02)
    colorbar.set_label(
        settings["colorbar_label"],
        fontsize=global_settings["font"]["size_colorbar_label"],
        fontfamily=global_settings["font"]["family"],
    )
    colorbar.ax.tick_params(labelsize=global_settings["font"]["size_colorbar_tick"])
    if settings["colorbar_scientific"]:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        colorbar.ax.yaxis.set_major_formatter(formatter)
        fig.canvas.draw()
        offset_text = colorbar.ax.yaxis.get_offset_text()
        offset_text.set_visible(False)
        offset_string = offset_text.get_text()
        if offset_string:
            colorbar.ax.text(
                1.5,
                1.02,
                offset_string,
                transform=colorbar.ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=global_settings["font"]["size_colorbar_tick"],
                fontfamily=global_settings["font"]["family"],
            )
    fig.suptitle(
        settings["title"],
        fontsize=global_settings["font"]["size_title"] + 2,
        fontfamily=global_settings["font"]["family"],
    )

    prefix = settings["output_prefix"]
    dpi = int(global_settings["dpi"])
    fig.savefig(OUT / f"{prefix}.png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(
        OUT / f"{prefix}.eps",
        format="eps",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    BMP_OUT.mkdir(parents=True, exist_ok=True)
    for index, (ax, title, artists, z_index) in enumerate(zip(axes_flat, titles, scalebars, indices, strict=True)):
        path = BMP_OUT / f"{prefix}_slice{index:02d}_z{z_index}.bmp"
        _save_z_slice_bmp(fig, ax, title, artists, path, dpi)
    plt.close(fig)

    with (OUT / f"{prefix}_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_z_slice_config(name), handle, sort_keys=True, allow_unicode=True)


def _fit_curve(positions_m: np.ndarray, fit: GaussianFitResult) -> tuple[np.ndarray, np.ndarray]:
    dense_m = np.linspace(float(positions_m[0]), float(positions_m[-1]), 200)
    values = fit.amplitude * np.exp(-0.5 * ((dense_m - fit.center_m) / fit.sigma_m) ** 2) + fit.offset
    return dense_m * 1e9, values


def create_profile_outputs(
    name: str,
    profiles: tuple[ProfileData, ...],
    fits: dict[str, GaussianFitResult],
) -> None:
    """Create the legacy three-panel profile plot and three panel-only BMPs."""
    resolution = CFG["resolution"]
    settings = resolution[name]
    profile_settings = resolution["profile"]
    font = resolution["font"]
    figure_size = tuple(profile_settings["figsize"])
    fig, axes = plt.subplots(1, 3, figsize=figure_size)
    for ax, profile, direction, color in zip(
        axes,
        profiles,
        DIRECTIONS,
        profile_settings["colors"],
        strict=True,
    ):
        position_nm = np.asarray(profile.positions_m) * 1e9
        ax.scatter(
            position_nm,
            np.asarray(profile.values),
            s=profile_settings["marker_size"],
            alpha=0.7,
            color=color,
            label="Data",
        )
        fit = fits[direction]
        dense_nm, fitted = _fit_curve(np.asarray(profile.positions_m), fit)
        ax.plot(
            dense_nm,
            fitted,
            color=profile_settings["fit_color"],
            linewidth=profile_settings["linewidth"],
            label="Gaussian fit",
        )
        ax.axhline(
            fit.amplitude / 2 + fit.offset,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Half max",
        )
        ax.set_title(
            f"{direction}-direction\nFWHM = {fit.fwhm_nm:.1f} nm",
            fontsize=profile_settings["font_size_subtitle"],
            fontfamily=font["family"],
        )
        ax.set_xlabel(
            profile_settings["xlabel"],
            fontsize=profile_settings["font_size_label"],
            fontfamily=font["family"],
        )
        ax.set_ylabel(
            settings["ylabel"],
            fontsize=profile_settings["font_size_label"],
            fontfamily=font["family"],
        )
        ax.tick_params(axis="both", labelsize=profile_settings["font_size_tick"])
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontfamily(font["family"])
        if profile_settings["show_legend"]:
            ax.legend(
                loc="upper right",
                prop={"family": font["family"], "size": profile_settings["font_size_legend"]},
            )
        if profile_settings["show_grid"]:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.suptitle(
        settings["title"],
        fontsize=profile_settings["font_size_title"],
        fontfamily=font["family"],
    )
    fig.tight_layout()
    prefix = settings["prefix"]
    output_path = OUT / f"{prefix}profiles_with_fit.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".eps"), format="eps", bbox_inches="tight")

    BMP_OUT.mkdir(parents=True, exist_ok=True)
    subfigure_size = (figure_size[0] / 3, figure_size[1])
    for profile, direction, color in zip(profiles, DIRECTIONS, profile_settings["colors"], strict=True):
        panel, ax = plt.subplots(1, 1, figsize=subfigure_size)
        position_m = np.asarray(profile.positions_m)
        ax.scatter(
            position_m * 1e9,
            np.asarray(profile.values),
            s=profile_settings["marker_size"],
            alpha=0.7,
            color=color,
        )
        dense_nm, fitted = _fit_curve(position_m, fits[direction])
        ax.plot(
            dense_nm,
            fitted,
            color=profile_settings["fit_color"],
            linewidth=profile_settings["linewidth"],
        )
        ax.set_axis_off()
        panel.tight_layout()
        _save_fig_as_bmp(panel, BMP_OUT / f"{prefix}profiles_with_fit_{direction}.bmp")
        plt.close(panel)
    plt.close(fig)


def _add_crosshair(
    ax: plt.Axes,
    center: tuple[int, int],
    half_range: tuple[int, int],
    color: str,
    linewidth: float,
) -> None:
    cx, cy = center
    rx, ry = half_range
    ax.plot([cx, cx], [cy - ry, cy + ry], color=color, linestyle="--", linewidth=linewidth, alpha=0.7)
    ax.plot([cx - rx, cx + rx], [cy, cy], color=color, linestyle="--", linewidth=linewidth, alpha=0.7)


def create_cross_section_outputs(name: str, volume: np.ndarray) -> None:
    """Create the legacy XY/XZ plot and two panel-only BMPs."""
    resolution = CFG["resolution"]
    settings = resolution[name]
    font = resolution["font"]
    cx, cy, cz = resolution["center_px"]
    rx, ry, rz = resolution["display_half_range_px"]
    erx, ery, erz = resolution["half_range_px"]
    x_min, x_max = cx - rx, cx + rx + 1
    y_min, y_max = cy - ry, cy + ry + 1
    z_min, z_max = cz - rz, cz + rz + 1
    offset = float(CFG["data"][name]["value_offset"])
    xy_slice = _real_values(
        volume,
        (slice(x_min, x_max), slice(y_min, y_max), cz),
        offset,
    )
    xz_slice = _real_values(
        volume,
        (slice(x_min, x_max), cy, slice(z_min, z_max)),
        offset,
    )
    extent_xy = (x_min, x_max - 1, y_min, y_max - 1)
    extent_xz = (x_min, x_max - 1, z_min, z_max - 1)
    aspect_xy = 1.0
    aspect_xz = float(resolution["pixel_size_z_m"]) / float(resolution["pixel_size_xy_m"])

    fig, axes = plt.subplots(2, 1, figsize=(7, 10), constrained_layout=True, sharex=True)
    image = axes[0].imshow(
        xy_slice.T,
        origin="lower",
        aspect=aspect_xy,
        cmap=settings["cmap"],
        extent=extent_xy,
        vmin=settings["vmin"],
        vmax=settings["vmax"],
    )
    _add_crosshair(
        axes[0],
        (cx, cy),
        (erx, ery),
        settings["crosshair_color"],
        settings["crosshair_linewidth"],
    )
    axes[0].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    axes[0].set_ylabel("Y", fontsize=font["size_label"], fontfamily=font["family"])
    axes[0].set_title(settings["title_xy"], fontsize=font["size_title"], fontfamily=font["family"])

    axes[1].imshow(
        xz_slice.T,
        origin="lower",
        aspect=aspect_xz,
        cmap=settings["cmap"],
        extent=extent_xz,
        vmin=settings["vmin"],
        vmax=settings["vmax"],
    )
    _add_crosshair(
        axes[1],
        (cx, cz),
        (erx, erz),
        settings["crosshair_color"],
        settings["crosshair_linewidth"],
    )
    axes[1].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    axes[1].set_xlabel("X", fontsize=font["size_label"], fontfamily=font["family"])
    axes[1].set_ylabel("Z", fontsize=font["size_label"], fontfamily=font["family"])
    axes[1].set_title(settings["title_xz"], fontsize=font["size_title"], fontfamily=font["family"])

    colorbar = fig.colorbar(image, ax=axes, shrink=1.0)
    colorbar.set_label(
        settings["colorbar_label"],
        fontsize=font["size_colorbar_label"],
        fontfamily=font["family"],
    )
    colorbar.ax.tick_params(labelsize=font["size_colorbar_tick"])
    for label in colorbar.ax.get_yticklabels():
        label.set_fontfamily(font["family"])
    if settings["colorbar_scientific"]:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        colorbar.ax.yaxis.set_major_formatter(formatter)
        colorbar.ax.yaxis.get_offset_text().set_fontsize(font["size_colorbar_tick"])
        colorbar.ax.yaxis.get_offset_text().set_fontfamily(font["family"])

    prefix = settings["prefix"]
    output_path = OUT / f"{prefix}cross_sections.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".eps"), format="eps", bbox_inches="tight")

    BMP_OUT.mkdir(parents=True, exist_ok=True)
    for plane, plane_data, extent, aspect, center, half_range in (
        ("XY", xy_slice.T, extent_xy, aspect_xy, (cx, cy), (erx, ery)),
        ("XZ", xz_slice.T, extent_xz, aspect_xz, (cx, cz), (erx, erz)),
    ):
        panel, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.imshow(
            plane_data,
            origin="lower",
            aspect=aspect,
            cmap=settings["cmap"],
            extent=extent,
            vmin=settings["vmin"],
            vmax=settings["vmax"],
        )
        _add_crosshair(
            ax,
            center,
            half_range,
            settings["crosshair_color"],
            settings["crosshair_linewidth"],
        )
        ax.set_axis_off()
        panel.tight_layout()
        _save_fig_as_bmp(panel, BMP_OUT / f"{prefix}cross_sections_{plane}.bmp")
        plt.close(panel)
    plt.close(fig)


def _resolution_config(name: str) -> dict[str, Any]:
    resolution = CFG["resolution"]
    settings = resolution[name]
    profile = resolution["profile"]
    font = resolution["font"]
    return {
        "data_source": {
            "input_file": CFG["data"][name]["path"],
            "value_offset": CFG["data"][name]["value_offset"],
        },
        "evaluation": {
            "center": resolution["center_px"],
            "range": resolution["half_range_px"],
        },
        "physical_constants": {
            "pixel_size_xy": resolution["pixel_size_xy_m"],
            "pixel_size_z": resolution["pixel_size_z_m"],
        },
        "processing": {"do_gaussian_fit": True},
        "visualization": {
            "display_range": resolution["display_half_range_px"],
            "colormap": settings["cmap"],
            "crosshair_color": settings["crosshair_color"],
            "crosshair_linewidth": settings["crosshair_linewidth"],
            "vmin": settings["vmin"],
            "vmax": settings["vmax"],
        },
        "font": {
            "family": font["family"],
            "size_title": font["size_title"],
            "size_label": font["size_label"],
            "size_colorbar_label": font["size_colorbar_label"],
            "size_colorbar_tick": font["size_colorbar_tick"],
        },
        "figure": {
            "title_xy": settings["title_xy"],
            "title_xz": settings["title_xz"],
            "colorbar_label": settings["colorbar_label"],
            "colorbar_scientific": settings["colorbar_scientific"],
        },
        "profile": {
            "figsize": profile["figsize"],
            "title": settings["title"],
            "xlabel": profile["xlabel"],
            "ylabel": settings["ylabel"],
            "font_size_title": profile["font_size_title"],
            "font_size_subtitle": profile["font_size_subtitle"],
            "font_size_label": profile["font_size_label"],
            "font_size_tick": profile["font_size_tick"],
            "font_size_legend": profile["font_size_legend"],
            "colors": profile["colors"],
            "fit_color": profile["fit_color"],
            "marker_size": profile["marker_size"],
            "linewidth": profile["linewidth"],
            "show_grid": profile["show_grid"],
            "show_legend": profile["show_legend"],
        },
        "output": {
            "output_dir": "output",
            "prefix": settings["prefix"],
            "save_eps": True,
            "save_bmp": True,
        },
    }


def save_resolution_results(name: str, fits: dict[str, GaussianFitResult]) -> None:
    """Save the legacy per-dataset JSON result and YAML configuration."""
    resolution = CFG["resolution"]
    settings = resolution[name]
    pixel_xy = float(resolution["pixel_size_xy_m"])
    pixel_z = float(resolution["pixel_size_z_m"])
    result = {
        "fwhm": {f"{direction}_nm": fits[direction].fwhm_nm for direction in DIRECTIONS},
        "fwhm_px": {
            "X_px": fits["X"].fwhm_m / pixel_xy,
            "Y_px": fits["Y"].fwhm_m / pixel_xy,
            "Z_px": fits["Z"].fwhm_m / pixel_z,
        },
        "fit_parameters": {
            direction: {
                "amplitude": fit.amplitude,
                "center_m": fit.center_m,
                "sigma_m": fit.sigma_m,
                "offset": fit.offset,
                "fwhm_m": fit.fwhm_m,
                "success": fit.success,
                "residual": fit.residual,
            }
            for direction, fit in fits.items()
        },
        "config": {
            "data_path": CFG["data"][name]["path"],
            "center": resolution["center_px"],
            "range": resolution["half_range_px"],
            "pixel_size_xy_m": pixel_xy,
            "pixel_size_z_m": pixel_z,
            "do_gaussian_fit": True,
            "save_eps": True,
        },
    }
    prefix = settings["prefix"]
    with (OUT / f"{prefix}results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    with (OUT / f"{prefix}config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_resolution_config(name), handle, sort_keys=False, allow_unicode=True)


def create_resolution_outputs(name: str, volume: np.ndarray) -> dict[str, GaussianFitResult]:
    """Evaluate resolution and write all files produced by the legacy script."""
    resolution = CFG["resolution"]
    pixel_sizes = (
        resolution["pixel_size_xy_m"],
        resolution["pixel_size_xy_m"],
        resolution["pixel_size_z_m"],
    )
    profiles = extract_profiles(
        volume,
        tuple(resolution["center_px"]),
        tuple(resolution["half_range_px"]),
        pixel_sizes,
        value_offset=float(CFG["data"][name]["value_offset"]),
    )
    fits = fit_profiles(profiles)
    create_profile_outputs(name, profiles, fits)
    create_cross_section_outputs(name, volume)
    save_resolution_results(name, fits)
    return fits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    BMP_OUT.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        volume = load_volume(name)
        create_z_slice_outputs(name, volume)
        fits = create_resolution_outputs(name, volume)
        values = ", ".join(f"{direction}={fits[direction].fwhm_nm:.6f} nm" for direction in DIRECTIONS)
        print(f"{name}: {values}")
    print(f"saved individual outputs under {OUT}")


if __name__ == "__main__":
    main()
