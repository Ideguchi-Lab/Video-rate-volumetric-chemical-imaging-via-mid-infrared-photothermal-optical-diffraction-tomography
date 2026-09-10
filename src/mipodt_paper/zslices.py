"""Z-slice panel used by the paper figures.

One horizontal row of XY planes with a shared scientific colorbar, a scalebar
on every plane, PNG/EPS of the whole panel, and a YAML snapshot of the settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter


@dataclass(frozen=True, slots=True)
class ZSliceStyle:
    """Fonts, geometry and scalebar shared by every z-slice panel."""

    pixel_size_xy_nm: float = 110.0
    pixel_size_z_nm: float = 220.0
    figsize_width: float = 6.0
    figsize_height: float = 5.0
    dpi: int = 300
    scalebar_um: float = 10.0
    scalebar_color: str = "white"
    scalebar_linewidth: float = 3.0
    font_family: str = "Arial"
    size_title: int = 26
    size_colorbar_label: int = 24
    size_colorbar_tick: int = 20
    size_scalebar: int = 22


@dataclass(frozen=True, slots=True)
class ZSliceSpec:
    """One panel: which planes to draw and how to color them."""

    input_label: str
    z_indices: tuple[int, ...]
    crop_xy: tuple[int, int, int, int]  # x_start, x_end, y_start, y_end
    cmap: str
    vmin: float | None
    vmax: float | None
    title: str
    colorbar_label: str
    colorbar_scientific: bool = True
    value_offset: float = 0.0
    output_prefix: str = "z_slices"
    roi_annotations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    arrow_annotations: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _add_scalebar(ax: Axes, style: ZSliceStyle) -> list[Any]:
    before = set(ax.get_children())
    length_px = style.scalebar_um * 1000.0 / style.pixel_size_xy_nm
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_start = xlim[1] - length_px - (xlim[1] - xlim[0]) * 0.05
    y_position = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot(
        [x_start, x_start + length_px],
        [y_position, y_position],
        color=style.scalebar_color,
        linewidth=style.scalebar_linewidth,
        solid_capstyle="butt",
    )
    ax.text(
        x_start + length_px / 2,
        y_position + (ylim[1] - ylim[0]) * 0.02,
        f"{style.scalebar_um:.0f} µm",
        color=style.scalebar_color,
        fontsize=style.size_scalebar,
        fontfamily=style.font_family,
        ha="center",
        va="bottom",
    )
    return [artist for artist in ax.get_children() if artist not in before]


def panel_config(spec: ZSliceSpec, style: ZSliceStyle) -> dict[str, Any]:
    """Settings of one panel as a plain mapping."""
    x_start, x_end, y_start, y_end = spec.crop_xy
    return {
        "data_source": {
            "input_file": spec.input_label,
            "z_slice_positions": list(spec.z_indices),
            "value_offset": spec.value_offset,
        },
        "layout": {
            "orientation": "horizontal",
            "figsize_width": style.figsize_width,
            "figsize_height_per_slice": style.figsize_height,
        },
        "crop": {"enabled": True, "x_start": x_start, "x_end": x_end, "y_start": y_start, "y_end": y_end},
        "roi_annotations": [dict(roi) for roi in spec.roi_annotations],
        "arrow_annotations": [dict(arrow) for arrow in spec.arrow_annotations],
        "colormap": {"name": spec.cmap, "vmin": spec.vmin, "vmax": spec.vmax},
        "output": {"output_dir": "output", "output_prefix": spec.output_prefix, "dpi": style.dpi, "save_png": True},
        "physical_constants": {
            "xy_pixel_size_nm": style.pixel_size_xy_nm,
            "z_pixel_size_nm": style.pixel_size_z_nm,
        },
        "font": {
            "family": style.font_family,
            "size_title": style.size_title,
            "size_ylabel": 24,
            "size_colorbar_label": style.size_colorbar_label,
            "size_colorbar_tick": style.size_colorbar_tick,
            "size_scalebar": style.size_scalebar,
        },
        "scalebar": {
            "length_um": style.scalebar_um,
            "color": style.scalebar_color,
            "linewidth": style.scalebar_linewidth,
        },
        "figure": {
            "title": spec.title,
            "colorbar_label": spec.colorbar_label,
            "colorbar_scientific": spec.colorbar_scientific,
        },
    }


def render_z_slices(volume: np.ndarray, spec: ZSliceSpec, style: ZSliceStyle, out_dir: Path) -> list[Path]:
    """Write ``<prefix>.png``, ``<prefix>.eps`` and ``<prefix>_config.yaml``; return the paths."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    invalid = [index for index in spec.z_indices if not 0 <= index < volume.shape[2]]
    if invalid:
        raise ValueError(f"z indices out of range for shape {volume.shape}: {invalid}")
    x_start, x_end, y_start, y_end = spec.crop_xy
    offset = np.float32(spec.value_offset)
    planes = [
        np.asarray(np.real(volume[x_start:x_end, y_start:y_end, z_index]), dtype=np.float32) + offset
        for z_index in spec.z_indices
    ]
    vmin = float(spec.vmin) + float(offset) if spec.vmin is not None else min(float(plane.min()) for plane in planes)
    vmax = float(spec.vmax) + float(offset) if spec.vmax is not None else max(float(plane.max()) for plane in planes)

    count = len(spec.z_indices)
    fig, axes = plt.subplots(1, count, figsize=(style.figsize_width * count, style.figsize_height), squeeze=False)
    axes_flat = list(axes[0])
    image = None
    for ax, plane, z_index in zip(axes_flat, planes, spec.z_indices, strict=True):
        image = ax.imshow(plane, cmap=spec.cmap, vmin=vmin, vmax=vmax, origin="lower")
        z_um = (z_index - spec.z_indices[0]) * style.pixel_size_z_nm / 1000.0
        ax.set_title(f"Z = {z_um:.1f} µm", fontsize=style.size_title, fontfamily=style.font_family)
        ax.set_xticks([])
        ax.set_yticks([])
        _add_scalebar(ax, style)
    if image is None:
        raise RuntimeError("No z slices were configured")

    colorbar = fig.colorbar(image, ax=axes_flat, shrink=0.8, pad=0.02)
    colorbar.set_label(spec.colorbar_label, fontsize=style.size_colorbar_label, fontfamily=style.font_family)
    colorbar.ax.tick_params(labelsize=style.size_colorbar_tick)
    if spec.colorbar_scientific:
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
                fontsize=style.size_colorbar_tick,
                fontfamily=style.font_family,
            )
    fig.suptitle(spec.title, fontsize=style.size_title + 2, fontfamily=style.font_family)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    png_path = out_dir / f"{spec.output_prefix}.png"
    eps_path = out_dir / f"{spec.output_prefix}.eps"
    fig.savefig(png_path, dpi=style.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(eps_path, format="eps", bbox_inches="tight", facecolor="white", edgecolor="none")
    written.extend([png_path, eps_path])
    plt.close(fig)

    config_path = out_dir / f"{spec.output_prefix}_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(panel_config(spec, style), handle, sort_keys=True, allow_unicode=True)
    written.append(config_path)
    return written
