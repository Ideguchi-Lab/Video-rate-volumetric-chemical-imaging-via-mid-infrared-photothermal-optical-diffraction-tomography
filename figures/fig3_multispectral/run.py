#!/usr/bin/env python3
"""Validate data and reproduce the individual Fig. 3 multispectral outputs.

Usage:
    uv run figures/fig3_multispectral/run.py
    uv run figures/fig3_multispectral/run.py --plot-only
    uv run figures/fig3_multispectral/run.py --validate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
from PIL import Image

from mipodt_paper import figure_output_dir
from mipodt_paper.config import load_config
from mipodt_paper.multispectral import (
    NnlsResult,
    Roi,
    apply_wavenumber_correction,
    area_normalize,
    broaden_to_fwhm,
    discover_volume_paths,
    extract_power_normalized_spectra,
    fit_nonnegative_components,
    load_reference_spectra_from_sources,
    load_volume_metadata,
    make_common_grid,
    portable_file_info,
    preprocess_native_spectrum,
    validate_real_float32_volume,
    validate_volume_files,
)
from mipodt_paper.nnls_plotting import NNLSFigureRow, plot_combined_nnls_paper_figure, plot_nnls_paper_figure
from mipodt_paper.paths import DATA_DIR, REPO_ROOT

OUT = figure_output_dir(__file__)
CFG = load_config(__file__)
CACHE_PATH = OUT / CFG["output"]["cache"]


def _under(root: Path, value: str, label: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root: {value}") from exc
    return path


def _configured_paths() -> dict[str, Path]:
    data = CFG["data"]
    return {
        "mipodt_directory": _under(DATA_DIR, data["mipodt_directory"], "mipodt_directory"),
        "static_odt": _under(DATA_DIR, data["static_odt_path"], "static_odt_path"),
        "reference_sources": _under(DATA_DIR, data["reference_sources_path"], "reference_sources_path"),
        "volume_metadata": _under(
            REPO_ROOT,
            data["volume_metadata_repository_path"],
            "volume_metadata_repository_path",
        ),
    }


def _range_ids(settings: Mapping[str, Any]) -> list[int]:
    return list(range(int(settings["start"]), int(settings["stop"])))


def _rois() -> dict[str, Roi]:
    return {name: Roi.from_mapping(value) for name, value in CFG["rois"].items()}


def _load_reference_sources(path: Path, required_names: set[str]) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    references = raw.get("references")
    if not isinstance(references, dict):
        raise ValueError(f"Reference source metadata must contain a 'references' mapping: {path}")
    missing = sorted(required_names - set(references))
    if missing:
        raise ValueError(f"Reference source metadata missing entries: {missing}")
    result: dict[str, dict[str, Any]] = {}
    for name in required_names:
        entry = references[name]
        if not isinstance(entry, dict) or "native_fwhm_cm_inv" not in entry:
            raise ValueError(f"Reference source metadata '{name}' requires native_fwhm_cm_inv")
        fwhm = float(entry["native_fwhm_cm_inv"])
        if fwhm < 0 or fwhm > float(CFG["nnls"]["common_resolution_fwhm_cm_inv"]):
            raise ValueError(f"Reference source metadata '{name}' has invalid native_fwhm_cm_inv={fwhm}")
        result[name] = dict(entry)
        result[name]["native_fwhm_cm_inv"] = fwhm
    return result


def validate_inputs() -> dict[str, Any]:
    """Validate every distributed input and return resolved handles."""
    if CFG.get("schema_version") != 1:
        raise ValueError("Unsupported Figure 3 configuration schema; expected schema_version: 1")
    paths = _configured_paths()
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {label} input not found: {path}")
    expected_ids = _range_ids(CFG["data"]["expected_volume_ids"])
    if expected_ids != list(range(42)):
        raise ValueError("Figure 3 requires exactly the calibrated volume IDs 0-41")
    metadata = load_volume_metadata(paths["volume_metadata"], expected_ids)
    volume_paths = discover_volume_paths(paths["mipodt_directory"], CFG["data"]["mipodt_filename_regex"], expected_ids)
    expected_shape = tuple(int(value) for value in CFG["data"]["expected_shape"])
    if CFG["data"]["expected_dtype"] != "float32":
        raise ValueError("Figure 3 public volumes must use expected_dtype: float32")
    validate_volume_files(volume_paths, expected_shape)
    static_volume = np.load(paths["static_odt"], mmap_mode="r", allow_pickle=False)
    validate_real_float32_volume(static_volume, expected_shape, label="static ODT")
    rois = _rois()
    for name, roi in rois.items():
        roi.validate(expected_shape, name=name)
    for section_name in ("odt_slices", "mipodt_slices"):
        section = CFG[section_name]
        for axis, size in (("axis0", expected_shape[0]), ("axis1", expected_shape[1])):
            start, stop = (int(value) for value in section["crop"][axis])
            if start < 0 or start >= stop or stop > size:
                raise ValueError(f"{section_name} crop {axis} is outside the volume shape")
        if any(int(z_index) < 0 or int(z_index) >= expected_shape[2] for z_index in section["z_indices"]):
            raise ValueError(f"{section_name} contains an out-of-range z index")
    mip_slice_ids = [int(value) for value in CFG["mipodt_slices"]["volume_ids"]]
    spectrum_ids = _range_ids(CFG["spectra"]["volume_ids"])
    if not set(mip_slice_ids + spectrum_ids).issubset(expected_ids):
        raise ValueError("Configured MIP-ODT slice or spectrum volume IDs are outside IDs 0-41")
    required_references = {name for names in CFG["nnls"]["regions"].values() for name in names}
    reference_metadata = _load_reference_sources(paths["reference_sources"], required_references)
    analysis_start, analysis_stop = (float(value) for value in CFG["nnls"]["analysis_range_cm_inv"])
    reference_spectra = load_reference_spectra_from_sources(
        reference_metadata,
        required_references,
        data_root=DATA_DIR,
        cache_directory=OUT / ".nist_reference_cache",
        bounds=(analysis_start, analysis_stop),
    )
    if CFG["mipodt_slices"]["source"] != "reconstructed_volumes":
        raise ValueError("mipodt_slices.source must be 'reconstructed_volumes'")
    metadata_by_id = {row.volume_id: row for row in metadata}
    corrected_spectrum_x = apply_wavenumber_correction(
        np.asarray([metadata_by_id[volume_id].wavenumber_cm_inv for volume_id in spectrum_ids]),
        CFG["spectra"]["wavenumber_correction_cm_inv"],
    )
    fit_start, fit_stop = (float(value) for value in CFG["nnls"]["area_and_fit_range_cm_inv"])
    if np.min(corrected_spectrum_x) > fit_start or np.max(corrected_spectrum_x) < fit_stop:
        raise ValueError("Corrected ROI spectrum wavenumbers do not cover the configured fit range")
    return {
        "paths": paths,
        "volume_paths": volume_paths,
        "metadata": metadata,
        "static_volume": static_volume,
        "rois": rois,
        "reference_spectra": reference_spectra,
        "reference_metadata": reference_metadata,
    }


def _crop_slice(volume: np.ndarray, z_index: int, crop: Mapping[str, list[int]]) -> np.ndarray:
    a0 = slice(*crop["axis0"])
    a1 = slice(*crop["axis1"])
    return np.asarray(volume[a0, a1, z_index], dtype=np.float32)


def compute(validated: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Compute image panels, ROI spectra, processed references, and NNLS fits."""
    metadata = validated["metadata"]
    metadata_by_id = {row.volume_id: row for row in metadata}
    volume_paths = validated["volume_paths"]
    spectrum_ids = _range_ids(CFG["spectra"]["volume_ids"])
    ids, raw_wavenumbers, spectra = extract_power_normalized_spectra(
        volume_paths, metadata, validated["rois"], spectrum_ids
    )
    corrected_wavenumbers = apply_wavenumber_correction(raw_wavenumbers, CFG["spectra"]["wavenumber_correction_cm_inv"])

    odt_settings = CFG["odt_slices"]
    static_slices = np.stack(
        [
            _crop_slice(validated["static_volume"], int(z_index), odt_settings["crop"])
            for z_index in odt_settings["z_indices"]
        ]
    )
    mip_settings = CFG["mipodt_slices"]
    mip_slices = []
    for volume_id in mip_settings["volume_ids"]:
        volume = np.load(volume_paths[int(volume_id)], mmap_mode="r", allow_pickle=False)
        power = metadata_by_id[int(volume_id)].power_mw
        mip_slices.append(
            np.stack([_crop_slice(volume, int(z), mip_settings["crop"]) / power for z in mip_settings["z_indices"]])
        )
    mip_color_limits = [
        [
            float(mip_settings["vmin"]),
            max(
                float(np.percentile(values, float(mip_settings["vmax_percentile"]))),
                float(mip_settings["row_vmax_floors"].get(str(volume_id), 0.0)),
            ),
        ]
        for volume_id, values in zip(mip_settings["volume_ids"], mip_slices, strict=True)
    ]

    nnls = CFG["nnls"]
    grid_step = float(nnls["common_grid_step_cm_inv"])
    asls = nnls["asls"]
    preparation = {
        "asls_lambda": float(asls["lambda"]),
        "asls_p": float(asls["p"]),
        "asls_iterations": int(asls["iterations"]),
        "analysis_bounds": nnls["analysis_range_cm_inv"],
        "area_bounds": nnls["area_and_fit_range_cm_inv"],
    }
    # Preprocess the nominal measurement points before shifting
    # the sample axis. Cropping after the shift changes the baseline endpoints.
    native_samples = {
        name: preprocess_native_spectrum(raw_wavenumbers, values, **preparation) for name, values in spectra.items()
    }
    sample_x = apply_wavenumber_correction(
        next(iter(native_samples.values()))[0], CFG["spectra"]["wavenumber_correction_cm_inv"]
    )
    processed_samples = {name: values for name, (_, values) in native_samples.items()}
    dense_references = {}
    for name, (x, y) in validated["reference_spectra"].items():
        native_x, native_y = preprocess_native_spectrum(x, y, **preparation)
        dense_x = np.arange(native_x[0], native_x[-1] + grid_step * 0.5, grid_step)
        dense_y = broaden_to_fwhm(
            dense_x,
            np.interp(dense_x, native_x, native_y),
            float(nnls["common_resolution_fwhm_cm_inv"]),
            float(validated["reference_metadata"][name]["native_fwhm_cm_inv"]),
        )
        dense_references[name] = (dense_x, dense_y)
    # The shared dense grid displays references; fitting uses measured sample points.
    common_start = max(x[0] for x, _ in dense_references.values())
    common_stop = min(x[-1] for x, _ in dense_references.values())
    grid = make_common_grid(
        (np.ceil(common_start / grid_step) * grid_step, np.floor(common_stop / grid_step) * grid_step), grid_step
    )
    processed_references = {name: np.interp(grid, x, y) for name, (x, y) in dense_references.items()}
    fit_low, fit_high = (float(value) for value in nnls["area_and_fit_range_cm_inv"])
    fit_mask = (sample_x >= fit_low) & (sample_x <= fit_high)
    fit_grid = sample_x[fit_mask]
    fit_references = {
        name: np.interp(fit_grid, x, y, left=np.nan, right=np.nan) for name, (x, y) in dense_references.items()
    }
    if fit_grid.size < int(nnls["minimum_fit_points"]) or not all(
        np.all(np.isfinite(values)) for values in fit_references.values()
    ):
        raise ValueError("NNLS requires enough measured points covered by every reference")
    fits = {
        region: fit_nonnegative_components(
            fit_grid,
            processed_samples[region][fit_mask],
            {name: fit_references[name] for name in component_names},
        )
        for region, component_names in nnls["regions"].items()
    }

    result: dict[str, np.ndarray] = {
        "cache_schema_version": np.asarray(3),
        "volume_ids": ids,
        "measurement_time_ms": np.asarray([metadata_by_id[int(i)].time_ms for i in ids]),
        "sample_grid_cm_inv": sample_x,
        "mipodt_display_wavenumbers": np.asarray(
            [
                round(metadata_by_id[int(i)].wavenumber_cm_inv) + CFG["spectra"]["wavenumber_correction_cm_inv"]
                for i in mip_settings["volume_ids"]
            ]
        ),
        "wavenumber_uncorrected_cm_inv": raw_wavenumbers,
        "wavenumber_corrected_cm_inv": corrected_wavenumbers,
        "odt_slices": static_slices,
        "mipodt_slices": np.stack(mip_slices),
        "mipodt_color_limits": np.asarray(mip_color_limits),
        "mipodt_slice_source": np.asarray(mip_settings["source"]),
        "common_grid_cm_inv": grid,
        "fit_grid_cm_inv": fit_grid,
    }
    for name, values in spectra.items():
        result[f"roi_spectrum__{name}"] = values
        result[f"processed_sample__{name}"] = processed_samples[name]
    for name, values in processed_references.items():
        result[f"processed_reference__{name}"] = values
        result[f"fit_reference__{name}"] = area_normalize(fit_grid, fit_references[name], (fit_low, fit_high))
    for region, fit in fits.items():
        result[f"nnls_components__{region}"] = np.asarray(fit.component_names)
        result[f"nnls_coefficients__{region}"] = fit.coefficients
        result[f"nnls_ratios__{region}"] = fit.ratios
        result[f"nnls_intercept_slope__{region}"] = np.asarray([fit.intercept, fit.slope])
        result[f"nnls_observed__{region}"] = fit.observed
        result[f"nnls_fitted__{region}"] = fit.fitted
        result[f"nnls_baseline__{region}"] = fit.baseline
        result[f"nnls_residual__{region}"] = fit.residual
    np.savez_compressed(CACHE_PATH, **result)
    return result


def _load_cache() -> dict[str, np.ndarray]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Analysis cache not found: {CACHE_PATH}. Run without --plot-only first.")
    with np.load(CACHE_PATH, allow_pickle=False) as archive:
        if "cache_schema_version" not in archive or int(archive["cache_schema_version"]) != 3:
            raise ValueError("Outdated Fig. 3 cache: rerun without --plot-only")
        if str(archive["mipodt_slice_source"]) != "reconstructed_volumes":
            raise ValueError("Cached mipodt_slice_source must be 'reconstructed_volumes'")
        return {name: archive[name] for name in archive.files}


def _save_figure(fig: plt.Figure, name: str, *, bmp: bool = False) -> list[Path]:
    saved: list[Path] = []
    for suffix in CFG["output"]["formats"]:
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=int(CFG["plot"]["dpi"]), bbox_inches="tight", facecolor="white")
        saved.append(path)
    if bmp:
        path = OUT / f"{name}.bmp"
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=int(CFG["plot"]["dpi"]), bbox_inches="tight")
            buffer.seek(0)
            with Image.open(buffer) as image:
                image.convert("RGB").save(path, format="BMP")
        saved.append(path)
    plt.close(fig)
    return saved


def _add_scalebar(ax: plt.Axes, width_px: int, settings: Mapping[str, Any]) -> None:
    """Place the upper-left scale bar with its label below it."""
    length_um = float(settings["scalebar_um"])
    length_px = length_um / float(CFG["sampling"]["xy_um_per_pixel"])
    x_start = width_px * 0.05
    y = width_px * 0.10
    ax.plot([x_start, x_start + length_px], [y, y], color="white", linewidth=3)
    ax.text(
        x_start + length_px / 2,
        y + width_px * 0.03,
        f"{length_um:g} µm",
        color="white",
        ha="center",
        va="top",
        fontsize=14,
        fontfamily="Arial",
    )


def _plot_odt_slices(result: Mapping[str, np.ndarray]) -> list[Path]:
    """Plot vertically arranged ODT slices with arrows at the spectral ROIs."""
    settings = CFG["odt_slices"]
    slices = result["odt_slices"] + float(CFG["sampling"]["medium_refractive_index"])
    xy_um = float(CFG["sampling"]["xy_um_per_pixel"])
    z_um = float(CFG["sampling"]["z_um_per_pixel"])
    crop = settings["crop"]
    extent = [crop["axis1"][0] * xy_um, crop["axis1"][1] * xy_um, crop["axis0"][1] * xy_um, crop["axis0"][0] * xy_um]
    width, height = settings["figsize_per_slice"]
    arrows = settings["arrows"]
    length = float(arrows["length_um"])
    angle = np.radians(float(arrows["offset_angle_deg"]))
    family = settings["font_family"]
    with plt.rc_context({"font.family": family}):
        fig, axes = plt.subplots(len(slices), 1, figsize=(width, height * len(slices)), squeeze=False)
        for ax, values, z_index in zip(axes[:, 0], slices, settings["z_indices"], strict=True):
            image = ax.imshow(
                values,
                cmap=settings["cmap"],
                vmin=settings["display_range"][0],
                vmax=settings["display_range"][1],
                extent=extent,
                aspect="equal",
                origin="upper",
            )
            depth = (z_index - settings["z_indices"][0]) * z_um
            ax.set_ylabel(f"z = {depth:.1f} μm", fontsize=settings["z_label_fontsize"], fontname=family)
            ax.set_xticks([])
            ax.set_yticks([])
            for name, color in arrows["colors"].items():
                roi = CFG["rois"][name]
                if not roi["z"][0] <= z_index < roi["z"][1]:
                    continue
                center_x = np.mean(roi["axis1"]) * xy_um
                center_y = np.mean(roi["axis0"]) * xy_um
                ax.annotate(
                    "",
                    xy=(center_x, center_y),
                    xytext=(center_x + length * np.cos(angle), center_y + length * np.sin(angle)),
                    arrowprops={
                        "arrowstyle": f"-|>,head_width={arrows['head_width_um'] / length},"
                        f"head_length={arrows['head_length_um'] / length}",
                        "color": color,
                        "lw": arrows["linewidth"],
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            start_x = xlim[0] + (xlim[1] - xlim[0]) * 0.05
            bar_y = ylim[1] - (ylim[1] - ylim[0]) * 0.10
            bar_length = float(settings["scalebar_um"])
            ax.plot([start_x, start_x + bar_length], [bar_y, bar_y], color="white", linewidth=3)
            ax.text(
                start_x + bar_length / 2,
                bar_y - (ylim[1] - ylim[0]) * 0.03,
                f"{bar_length:g} μm",
                color="white",
                ha="center",
                va="top",
                fontsize=settings["scalebar_fontsize"],
                fontname=family,
            )
        cbar = fig.colorbar(image, cax=fig.add_axes((0.88, 0.15, 0.03, 0.7)))
        cbar.set_label("RI", fontsize=settings["colorbar_title_fontsize"], fontname=family)
        cbar.ax.tick_params(labelsize=settings["colorbar_fontsize"])
        for label in cbar.ax.get_yticklabels():
            label.set_fontname(family)
        fig.suptitle(settings["title"], fontsize=settings["title_fontsize"], fontname=family, y=0.98)
        fig.tight_layout(rect=(0, 0, 0.85, 0.95))
        return _save_figure(fig, "odt_roi_slices", bmp=True)


def _plot_mipodt_slices(result: Mapping[str, np.ndarray]) -> list[Path]:
    """Plot MIP-ODT slices with one row per wavenumber and one column per depth."""
    settings = CFG["mipodt_slices"]
    data = result["mipodt_slices"]
    nrows, ncols = data.shape[:2]
    fig, axes = plt.subplots(
        nrows,
        ncols + 1,
        figsize=(3 * ncols + 0.9, 3 * nrows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0] * ncols + [0.12]},
    )
    z_ref = min(settings["z_indices"])
    for row, volume_id in enumerate(settings["volume_ids"]):
        limits = result["mipodt_color_limits"][row]
        for column, z_index in enumerate(settings["z_indices"]):
            ax = axes[row, column]
            values = data[row, column]
            im = ax.imshow(
                values,
                cmap=settings["cmap"],
                vmin=limits[0],
                vmax=limits[1],
                origin="upper",
                interpolation="nearest",
                aspect="equal",
            )
            if row == 0:
                z_um = (z_index - z_ref) * float(CFG["sampling"]["z_um_per_pixel"])
                ax.set_title(f"z = {z_um:.1f} µm", fontsize=16, fontfamily="Arial")
            if column == 0:
                wn = result["mipodt_display_wavenumbers"][row]
                ax.set_ylabel(f"MIP({wn:,.0f} cm$^{{-1}}$)", fontsize=16, fontfamily="Arial")
            ax.set_xticks([])
            ax.set_yticks([])
            _add_scalebar(ax, values.shape[1], settings)
        cbar = fig.colorbar(im, cax=axes[row, -1])
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.yaxis.set_offset_position("left")
        offset = cbar.ax.yaxis.get_offset_text()
        offset.set_fontsize(14)
        offset.set_fontfamily("Arial")
        offset.set_position((0, 1.02))
        offset.set_verticalalignment("bottom")
        offset.set_horizontalalignment("left")
        for label in cbar.ax.get_yticklabels():
            label.set_fontfamily("Arial")
    fig.subplots_adjust(left=0.07, right=0.96, top=0.96, bottom=0.06, wspace=0.25, hspace=0.28)
    return _save_figure(fig, "mipodt_slices", bmp=True)


def _plot_roi_spectra(result: Mapping[str, np.ndarray]) -> list[Path]:
    """Draw the paper's lipid/nucleolus spectrum with its measurement-time axis."""
    x = result["wavenumber_corrected_cm_inv"]
    times = result["measurement_time_ms"] - np.min(result["measurement_time_ms"])
    styles = CFG["plot"]["spectrum_styles"]
    with plt.rc_context({"font.family": "Arial"}):
        fig, ax = plt.subplots(figsize=(5.4, 3.5))
        for name in ("lipid_droplet", "nucleolus"):
            style = styles[name]
            ax.plot(
                x,
                result[f"roi_spectrum__{name}"],
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=1.3,
                markersize=3,
                label=style["label"],
            )
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        ax.set_ylim(CFG["plot"]["lipid_nucleolus_ylim"])
        ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=11)
        ax.set_ylabel("$\\Delta n$ / MIR power (/mW)", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(10)
        ax2 = ax.twinx()
        ax2.scatter(x, times, color="red", s=8, label="Measurement time point")
        ax2.set_ylim(CFG["plot"]["measurement_time_ylim_ms"])
        ax2.set_ylabel("Measurement time point (ms)", fontsize=11, color="red")
        ax2.tick_params(labelcolor="red", labelsize=10)
        ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        fig.tight_layout()
        return _save_figure(fig, "spectrum_analysis_lipid_nucleolus")


def _fit_from_cache(result: Mapping[str, np.ndarray], region: str) -> NnlsResult:
    components = tuple(str(value) for value in result[f"nnls_components__{region}"].tolist())
    intercept, slope = result[f"nnls_intercept_slope__{region}"]
    return NnlsResult(
        component_names=components,
        coefficients=result[f"nnls_coefficients__{region}"],
        ratios=result[f"nnls_ratios__{region}"],
        intercept=float(intercept),
        slope=float(slope),
        observed=result[f"nnls_observed__{region}"],
        fitted=result[f"nnls_fitted__{region}"],
        baseline=result[f"nnls_baseline__{region}"],
        residual=result[f"nnls_residual__{region}"],
    )


def _plot_nnls(result: Mapping[str, np.ndarray]) -> list[Path]:
    """Render fit/residual, component, and coefficient-fraction panels."""
    reference_names = sorted({name for names in CFG["nnls"]["regions"].values() for name in names})
    metadata = pd.DataFrame({"sample_id": reference_names, "display_name": reference_names})
    rows = []
    saved = []
    with plt.rc_context({"font.size": 9, "font.family": "sans-serif", "ps.fonttype": 42}):
        for region, components in CFG["nnls"]["regions"].items():
            fit = _fit_from_cache(result, region)
            grid = result["fit_grid_cm_inv"]
            # Component curves use the normalized reference basis times its coefficient;
            # the fitted baseline belongs to the total fit.
            basis = {
                name: pd.DataFrame({"wavenumber_cm1": grid, "absorbance_normalized": result[f"fit_reference__{name}"]})
                for name in components
            }
            coefficients = pd.DataFrame(
                {
                    "roi_id": region,
                    "reference_id": components,
                    "coefficient": fit.coefficients,
                    "coefficient_fraction": fit.ratios,
                }
            )
            curves = {
                region: pd.DataFrame(
                    {"wavenumber_cm1": grid, "observed": fit.observed, "fitted": fit.fitted, "residual": fit.residual}
                )
            }
            style = CFG["plot"]["nnls_styles"][region]
            row = NNLSFigureRow(
                dense_basis=basis,
                coeffs=coefficients,
                curves=curves,
                title=style["title"],
                roi_id=region,
                observed_marker=style["marker"],
                observed_marker_size=style["markersize"],
                observed_color=style["color"],
                reference_colors=style["reference_colors"],
                ratio_reference_order=tuple(style["ratio_order"]),
            )
            rows.append(row)
            stem = f"nnls_{region}"
            plot_nnls_paper_figure(
                basis,
                coefficients,
                curves,
                metadata,
                CFG["nnls"]["area_and_fit_range_cm_inv"],
                OUT,
                False,
                stem,
                row.title,
                region,
                row.observed_marker,
                row.observed_marker_size,
                row.observed_color,
                row.reference_colors,
                row.ratio_reference_order,
            )
            saved.extend(OUT / f"{stem}.{suffix}" for suffix in CFG["output"]["formats"])
        plot_combined_nnls_paper_figure(
            rows, metadata, CFG["nnls"]["area_and_fit_range_cm_inv"], OUT, False, "nnls_combined"
        )
        saved.extend(OUT / f"nnls_combined.{suffix}" for suffix in CFG["output"]["formats"])
    return saved


def write_csv_outputs(result: Mapping[str, np.ndarray]) -> list[Path]:
    """Write raw/corrected ROI spectra and all NNLS numerical results."""
    saved: list[Path] = []
    roi_path = OUT / "roi_spectra.csv"
    roi_names = list(CFG["rois"])
    with roi_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["volume_id", "wavenumber_uncorrected_cm_inv", "wavenumber_corrected_cm_inv"]
            + [f"{name}_mean_ri_change_per_mw" for name in roi_names]
        )
        for index, volume_id in enumerate(result["volume_ids"]):
            writer.writerow(
                [
                    int(volume_id),
                    result["wavenumber_uncorrected_cm_inv"][index],
                    result["wavenumber_corrected_cm_inv"][index],
                    *[result[f"roi_spectrum__{name}"][index] for name in roi_names],
                ]
            )
    saved.append(roi_path)

    coefficient_path = OUT / "nnls_coefficients.csv"
    ratio_path = OUT / "nnls_coefficient_ratios.csv"
    with (
        coefficient_path.open("w", newline="", encoding="utf-8") as coefficients_handle,
        ratio_path.open("w", newline="", encoding="utf-8") as ratios_handle,
    ):
        coefficient_writer = csv.writer(coefficients_handle)
        ratio_writer = csv.writer(ratios_handle)
        coefficient_writer.writerow(["region", "component", "coefficient", "baseline_intercept", "baseline_slope"])
        ratio_writer.writerow(["region", "component", "coefficient_ratio"])
        for region in CFG["nnls"]["regions"]:
            fit = _fit_from_cache(result, region)
            for name, coefficient, ratio in zip(fit.component_names, fit.coefficients, fit.ratios, strict=True):
                coefficient_writer.writerow([region, name, coefficient, fit.intercept, fit.slope])
                ratio_writer.writerow([region, name, ratio])
    saved.extend([coefficient_path, ratio_path])

    residual_path = OUT / "nnls_residuals.csv"
    with residual_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region", "wavenumber_cm_inv", "observed", "fitted", "baseline", "residual"])
        for region in CFG["nnls"]["regions"]:
            fit = _fit_from_cache(result, region)
            for values in zip(
                result["fit_grid_cm_inv"], fit.observed, fit.fitted, fit.baseline, fit.residual, strict=True
            ):
                writer.writerow([region, *values])
    saved.append(residual_path)
    return saved


def plot_all(result: Mapping[str, np.ndarray]) -> list[Path]:
    """Regenerate every figure and CSV solely from cached numerical arrays."""
    plt.rcParams.update({"font.size": CFG["plot"]["font_size"], "font.family": "sans-serif"})
    saved = _plot_odt_slices(result)
    saved.extend(_plot_mipodt_slices(result))
    saved.extend(_plot_roi_spectra(result))
    saved.extend(_plot_nnls(result))
    saved.extend(write_csv_outputs(result))
    return saved


def _snapshot_config() -> Path:
    destination = OUT / "config_snapshot.yaml"
    source = Path(__file__).resolve().with_name("config.yaml")
    shutil.copyfile(source, destination)
    return destination


def _write_manifest(validated: Mapping[str, Any], outputs: list[Path]) -> Path:
    paths = validated["paths"]
    volume_paths = validated["volume_paths"]
    input_files = [portable_file_info(paths["static_odt"], DATA_DIR)]
    input_files.extend(portable_file_info(volume_paths[volume_id], DATA_DIR) for volume_id in sorted(volume_paths))
    input_files.extend(
        [
            portable_file_info(paths["reference_sources"], DATA_DIR),
            portable_file_info(paths["volume_metadata"], REPO_ROOT),
        ]
    )
    seed_provenance = paths["mipodt_directory"] / "run_manifest.yaml"
    if seed_provenance.exists():
        input_files.append(portable_file_info(seed_provenance, DATA_DIR))
    remote_references = []
    for name, entry in sorted(validated["reference_metadata"].items()):
        if entry.get("source_type") == "local_jasco_csv":
            input_files.append(portable_file_info(_under(DATA_DIR, entry["path"], f"references.{name}.path"), DATA_DIR))
        elif entry.get("source_type") == "nist_webbook_jcamp":
            remote_references.append({"name": name, "url": entry["url"], "sha256": entry["sha256"]})
    manifest = {
        "schema_version": 1,
        "figure_id": CFG["figure_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "axis_convention": CFG["array_convention"],
        "mipodt_slice_source": CFG["mipodt_slices"]["source"],
        "seed_reconstruction_provenance": (
            portable_file_info(seed_provenance, DATA_DIR)["path"] if seed_provenance.exists() else None
        ),
        "nnls_fit_sampling": "measured_points_after_nominal_axis_preprocessing",
        "volume_ids": _range_ids(CFG["data"]["expected_volume_ids"]),
        "input_files": input_files,
        "remote_reference_sources": remote_references,
        "configuration_snapshot": "config_snapshot.yaml",
        "analysis_cache": CACHE_PATH.name,
        "outputs": sorted(path.name for path in outputs),
    }
    path = OUT / "run_manifest.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Write analysis and plots to a separate directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-only",
        action="store_true",
        help="replot and rewrite CSVs from output/analysis_cache.npz",
    )
    group.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all inputs without analysis or output generation",
    )
    return parser.parse_args()


def main() -> None:
    global OUT, CACHE_PATH
    args = parse_args()
    if args.output_dir is not None:
        OUT = args.output_dir.expanduser().resolve()
        OUT.mkdir(parents=True, exist_ok=True)
        CACHE_PATH = OUT / CFG["output"]["cache"]
    if args.plot_only:
        result = _load_cache()
        outputs = plot_all(result)
        print(f"Recreated {len(outputs)} outputs from {CACHE_PATH.name}")
        return
    validated = validate_inputs()
    print("Validated volume IDs 0-41, 42 NPY files, static ODT, ROIs, and reference spectra")
    if args.validate_only:
        return
    result = compute(validated)
    outputs = plot_all(result)
    outputs.extend([CACHE_PATH, _snapshot_config()])
    manifest = _write_manifest(validated, outputs)
    print(json.dumps({"output_directory": str(OUT), "manifest": manifest.name, "file_count": len(outputs) + 1}))


if __name__ == "__main__":
    main()
