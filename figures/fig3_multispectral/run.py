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
import yaml
from PIL import Image

from mipodt_paper import figure_output_dir
from mipodt_paper.config import load_config
from mipodt_paper.multispectral import (
    NnlsResult,
    Roi,
    apply_wavenumber_correction,
    discover_volume_paths,
    extract_power_normalized_spectra,
    fit_nonnegative_components,
    load_reference_spectra,
    load_volume_metadata,
    make_common_grid,
    portable_file_info,
    prepare_spectrum,
    validate_real_float32_volume,
    validate_volume_files,
)
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
        "reference_spectra": _under(DATA_DIR, data["reference_spectra_path"], "reference_spectra_path"),
        "reference_metadata": _under(DATA_DIR, data["reference_metadata_path"], "reference_metadata_path"),
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


def _load_reference_metadata(path: Path, required_names: set[str]) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    references = raw.get("references")
    if not isinstance(references, dict):
        raise ValueError(f"Reference metadata must contain a 'references' mapping: {path}")
    missing = sorted(required_names - set(references))
    if missing:
        raise ValueError(f"Reference metadata missing entries: {missing}")
    result: dict[str, dict[str, Any]] = {}
    for name in required_names:
        entry = references[name]
        if not isinstance(entry, dict) or "native_fwhm_cm_inv" not in entry:
            raise ValueError(f"Reference metadata '{name}' requires native_fwhm_cm_inv")
        fwhm = float(entry["native_fwhm_cm_inv"])
        if fwhm < 0 or fwhm > float(CFG["nnls"]["common_resolution_fwhm_cm_inv"]):
            raise ValueError(f"Reference metadata '{name}' has invalid native_fwhm_cm_inv={fwhm}")
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
    reference_x, reference_values = load_reference_spectra(paths["reference_spectra"], required_references)
    reference_metadata = _load_reference_metadata(paths["reference_metadata"], required_references)
    analysis_start, analysis_stop = (float(value) for value in CFG["nnls"]["analysis_range_cm_inv"])
    if np.min(reference_x) > analysis_start or np.max(reference_x) < analysis_stop:
        raise ValueError("Reference spectra do not cover the configured 2800-3100 cm^-1 analysis range")
    if np.unique(reference_x).size != reference_x.size:
        raise ValueError("Reference spectrum wavenumbers must be unique")
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
        "reference_x": reference_x,
        "reference_values": reference_values,
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
            np.stack(
                [
                    _crop_slice(volume, int(z_index), mip_settings["crop"]) / power
                    for z_index in mip_settings["z_indices"]
                ]
            )
        )

    nnls = CFG["nnls"]
    analysis_start, analysis_stop = (float(value) for value in nnls["analysis_range_cm_inv"])
    grid_step = float(nnls["common_grid_step_cm_inv"])
    measured_in_range = corrected_wavenumbers[
        (corrected_wavenumbers >= analysis_start) & (corrected_wavenumbers <= analysis_stop)
    ]
    reference_in_range = validated["reference_x"][
        (validated["reference_x"] >= analysis_start) & (validated["reference_x"] <= analysis_stop)
    ]
    overlap_start = max(float(np.min(measured_in_range)), float(np.min(reference_in_range)))
    overlap_stop = min(float(np.max(measured_in_range)), float(np.max(reference_in_range)))
    common_start = np.ceil(overlap_start / grid_step) * grid_step
    common_stop = np.floor(overlap_stop / grid_step) * grid_step
    grid = make_common_grid((common_start, common_stop), grid_step)
    asls = nnls["asls"]
    preparation = {
        "asls_lambda": float(asls["lambda"]),
        "asls_p": float(asls["p"]),
        "asls_iterations": int(asls["iterations"]),
        "analysis_bounds": nnls["analysis_range_cm_inv"],
        "area_bounds": nnls["area_and_fit_range_cm_inv"],
    }
    processed_samples = {
        name: prepare_spectrum(corrected_wavenumbers, values, grid, **preparation) for name, values in spectra.items()
    }
    processed_references = {
        name: prepare_spectrum(
            validated["reference_x"],
            values,
            grid,
            target_fwhm_cm_inv=float(nnls["common_resolution_fwhm_cm_inv"]),
            native_fwhm_cm_inv=float(validated["reference_metadata"][name]["native_fwhm_cm_inv"]),
            **preparation,
        )
        for name, values in validated["reference_values"].items()
    }
    fit_low, fit_high = (float(value) for value in nnls["area_and_fit_range_cm_inv"])
    fit_mask = (grid >= fit_low) & (grid <= fit_high)
    fit_grid = grid[fit_mask]
    fits = {
        region: fit_nonnegative_components(
            fit_grid,
            processed_samples[region][fit_mask],
            {name: processed_references[name][fit_mask] for name in component_names},
        )
        for region, component_names in nnls["regions"].items()
    }

    result: dict[str, np.ndarray] = {
        "volume_ids": ids,
        "wavenumber_uncorrected_cm_inv": raw_wavenumbers,
        "wavenumber_corrected_cm_inv": corrected_wavenumbers,
        "odt_slices": static_slices,
        "mipodt_slices": np.stack(mip_slices),
        "common_grid_cm_inv": grid,
        "fit_grid_cm_inv": fit_grid,
    }
    for name, values in spectra.items():
        result[f"roi_spectrum__{name}"] = values
        result[f"processed_sample__{name}"] = processed_samples[name]
    for name, values in processed_references.items():
        result[f"processed_reference__{name}"] = values
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
    length_um = float(settings["scalebar_um"])
    length_px = length_um / float(CFG["sampling"]["xy_um_per_pixel"])
    x_end = width_px * 0.94
    y = width_px * 0.94
    ax.plot([x_end - length_px, x_end], [y, y], color="white", linewidth=3, solid_capstyle="butt")
    ax.text(x_end - length_px / 2, y - width_px * 0.025, f"{length_um:g} µm", color="white", ha="center", va="top")


def _plot_odt_slices(result: Mapping[str, np.ndarray]) -> list[Path]:
    settings = CFG["odt_slices"]
    slices = result["odt_slices"] + float(CFG["sampling"]["medium_refractive_index"])
    fig, axes = plt.subplots(1, slices.shape[0], figsize=(8, 4), constrained_layout=True)
    image = None
    crop = settings["crop"]
    for ax, values, z_index in zip(np.atleast_1d(axes), slices, settings["z_indices"], strict=True):
        image = ax.imshow(
            values,
            cmap=settings["cmap"],
            vmin=settings["display_range"][0],
            vmax=settings["display_range"][1],
        )
        ax.set_title(f"ODT, z={z_index}")
        ax.set_axis_off()
        _add_scalebar(ax, values.shape[1], settings)
        for arrow in settings["arrows"]:
            if int(arrow["z_index"]) != int(z_index):
                continue
            target_a0, target_a1 = arrow["target"]
            label_a0, label_a1 = arrow["label"]
            ax.annotate(
                arrow["name"].replace("_", " "),
                xy=(target_a1 - crop["axis1"][0], target_a0 - crop["axis0"][0]),
                xytext=(label_a1 - crop["axis1"][0], label_a0 - crop["axis0"][0]),
                color=arrow["color"],
                fontsize=8,
                arrowprops={"arrowstyle": "->", "color": arrow["color"], "lw": 1.5},
            )
    if image is None:
        raise RuntimeError("No ODT slices configured")
    fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), label="Refractive index")
    return _save_figure(fig, "odt_roi_slices", bmp=True)


def _plot_mipodt_slices(result: Mapping[str, np.ndarray]) -> list[Path]:
    settings = CFG["mipodt_slices"]
    data = result["mipodt_slices"]
    fig, axes = plt.subplots(data.shape[0], data.shape[1], figsize=(7, 9), constrained_layout=True, squeeze=False)
    for row, volume_id in enumerate(settings["volume_ids"]):
        limits = settings["row_display_ranges"][str(volume_id)]
        row_image = None
        for column, z_index in enumerate(settings["z_indices"]):
            ax = axes[row, column]
            values = data[row, column]
            row_image = ax.imshow(values, cmap=settings["cmap"], vmin=limits[0], vmax=limits[1])
            ax.set_title(f"volume {volume_id}, z={z_index}")
            ax.set_axis_off()
            _add_scalebar(ax, values.shape[1], settings)
        if row_image is not None:
            fig.colorbar(row_image, ax=axes[row, :].tolist(), label="RI change / mW")
    return _save_figure(fig, "mipodt_slices", bmp=True)


def _plot_roi_spectra(result: Mapping[str, np.ndarray]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = result["wavenumber_corrected_cm_inv"]
    for name, color in CFG["plot"]["spectrum_colors"].items():
        ax.plot(x, result[f"roi_spectrum__{name}"], "o-", label=name.replace("_", " "), color=color)
    ax.set_xlabel("Corrected wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Mean RI change / laser power [mW$^{-1}$]")
    ax.invert_xaxis()
    ax.legend(frameon=False)
    return _save_figure(fig, "roi_spectra")


def _plot_comparison(result: Mapping[str, np.ndarray], region: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    grid = result["common_grid_cm_inv"]
    ax.plot(grid, result[f"processed_sample__{region}"], color="black", linewidth=2, label=region.replace("_", " "))
    for component in CFG["nnls"]["regions"][region]:
        ax.plot(grid, result[f"processed_reference__{component}"], label=component.replace("_", " "))
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Area-normalized signal")
    ax.invert_xaxis()
    ax.legend(frameon=False)
    return _save_figure(fig, f"{region}_comparison")


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


def _draw_nnls(ax: plt.Axes, result: Mapping[str, np.ndarray], region: str) -> None:
    fit = _fit_from_cache(result, region)
    grid = result["fit_grid_cm_inv"]
    ax.plot(grid, fit.observed, color="black", linewidth=2, label="observed")
    ax.plot(grid, fit.fitted, color="#d62728", linewidth=2, label="fit")
    for name, coefficient in zip(fit.component_names, fit.coefficients, strict=True):
        contribution = (
            coefficient
            * result[f"processed_reference__{name}"][
                (result["common_grid_cm_inv"] >= grid[0]) & (result["common_grid_cm_inv"] <= grid[-1])
            ]
        )
        ax.plot(grid, contribution + fit.baseline, linestyle="--", label=name.replace("_", " "))
    ax.set_title(region.replace("_", " "))
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Area-normalized signal")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)


def _plot_nnls(result: Mapping[str, np.ndarray]) -> list[Path]:
    regions = list(CFG["nnls"]["regions"])
    saved: list[Path] = []
    fig, axes = plt.subplots(1, len(regions), figsize=(11, 4), constrained_layout=True, squeeze=False)
    for ax, region in zip(axes[0], regions, strict=True):
        _draw_nnls(ax, result, region)
    saved.extend(_save_figure(fig, "nnls_combined"))
    for region in regions:
        fig, (fit_ax, residual_ax) = plt.subplots(
            2, 1, figsize=(6, 6), constrained_layout=True, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        _draw_nnls(fit_ax, result, region)
        residual_ax.axhline(0, color="gray", linewidth=1)
        residual_ax.plot(result["fit_grid_cm_inv"], result[f"nnls_residual__{region}"], color="#555555")
        residual_ax.set_xlabel("Wavenumber [cm$^{-1}$]")
        residual_ax.set_ylabel("Residual")
        saved.extend(_save_figure(fig, f"nnls_{region}"))
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
    saved.extend(_plot_comparison(result, "lipid_droplet"))
    saved.extend(_plot_comparison(result, "nucleolus"))
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
            portable_file_info(paths["reference_spectra"], DATA_DIR),
            portable_file_info(paths["reference_metadata"], DATA_DIR),
            portable_file_info(paths["volume_metadata"], REPO_ROOT),
        ]
    )
    manifest = {
        "schema_version": 1,
        "figure_id": CFG["figure_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "axis_convention": CFG["array_convention"],
        "volume_ids": _range_ids(CFG["data"]["expected_volume_ids"]),
        "input_files": input_files,
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
    args = parse_args()
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
