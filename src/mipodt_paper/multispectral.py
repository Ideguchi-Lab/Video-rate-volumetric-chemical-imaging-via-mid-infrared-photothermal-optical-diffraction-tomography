"""Numerical helpers for the Fig. 3 multispectral analysis."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True, slots=True)
class Roi:
    """A half-open three-dimensional ROI in ``(axis0, axis1, z)`` order."""

    axis0: tuple[int, int]
    axis1: tuple[int, int]
    z: tuple[int, int]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Sequence[int]]) -> Roi:
        """Create an ROI from a YAML-compatible mapping."""
        intervals: list[tuple[int, int]] = []
        for key in ("axis0", "axis1", "z"):
            bounds = tuple(int(item) for item in value[key])
            if len(bounds) != 2 or bounds[0] < 0 or bounds[0] >= bounds[1]:
                raise ValueError(f"ROI {key} must be a non-empty half-open [start, stop] interval")
            intervals.append(bounds)
        return cls(*intervals)

    @property
    def selection(self) -> tuple[slice, slice, slice]:
        """Return NumPy slices preserving the documented half-open bounds."""
        return tuple(slice(start, stop) for start, stop in (self.axis0, self.axis1, self.z))  # type: ignore[return-value]

    def validate(self, shape: Sequence[int], *, name: str = "ROI") -> None:
        """Raise when any half-open interval lies outside ``shape``."""
        if len(shape) != 3:
            raise ValueError(f"{name} requires a three-dimensional volume, got shape {tuple(shape)}")
        for axis, ((_, stop), size) in enumerate(zip((self.axis0, self.axis1, self.z), shape, strict=True)):
            if stop > size:
                raise ValueError(f"{name} exceeds axis {axis}: stop {stop}, size {size}")


@dataclass(frozen=True, slots=True)
class VolumeMetadata:
    """Calibrated spectral coordinate and laser power for one volume."""

    volume_id: int
    wavenumber_cm_inv: float
    power_mw: float


@dataclass(frozen=True, slots=True)
class NnlsResult:
    """One constrained spectral fit on the common wavenumber grid."""

    component_names: tuple[str, ...]
    coefficients: np.ndarray
    ratios: np.ndarray
    intercept: float
    slope: float
    observed: np.ndarray
    fitted: np.ndarray
    baseline: np.ndarray
    residual: np.ndarray


def validate_real_float32_volume(volume: np.ndarray, expected_shape: Sequence[int], *, label: str) -> None:
    """Validate the public reconstruction contract without copying an array."""
    if tuple(volume.shape) != tuple(expected_shape):
        raise ValueError(f"{label}: expected shape {tuple(expected_shape)}, got {volume.shape}")
    if volume.dtype != np.dtype(np.float32):
        raise ValueError(f"{label}: expected dtype float32, got {volume.dtype}")
    if np.iscomplexobj(volume):
        raise ValueError(f"{label}: expected a real array")


def load_volume_metadata(path: Path, expected_ids: Sequence[int]) -> list[VolumeMetadata]:
    """Load and validate the calibrated volume-to-wavenumber manifest."""
    required = {"volume_id", "wavenumber_cm_inv", "power_mw"}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Volume metadata must contain {sorted(required)}: {path}")
        rows = [
            VolumeMetadata(int(row["volume_id"]), float(row["wavenumber_cm_inv"]), float(row["power_mw"]))
            for row in reader
        ]
    ids = [row.volume_id for row in rows]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate volume IDs in metadata: {duplicates}")
    expected = list(expected_ids)
    missing = sorted(set(expected) - set(ids))
    unexpected = sorted(set(ids) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Volume metadata ID mismatch; missing={missing}, unexpected={unexpected}")
    by_id = {row.volume_id: row for row in rows}
    ordered = [by_id[volume_id] for volume_id in expected]
    if any(not np.isfinite(row.wavenumber_cm_inv) for row in ordered):
        raise ValueError("Wavenumbers must be finite")
    if any(not np.isfinite(row.power_mw) or row.power_mw <= 0 for row in ordered):
        raise ValueError("Laser powers must be finite and positive")
    return ordered


def discover_volume_paths(directory: Path, filename_regex: str, expected_ids: Sequence[int]) -> dict[int, Path]:
    """Find exactly one NPY for every expected ID and reject duplicates/extras."""
    pattern = re.compile(filename_regex)
    discovered: dict[int, Path] = {}
    duplicates: list[int] = []
    unexpected: list[int] = []
    unmatched: list[str] = []
    npy_paths = sorted(directory.glob("*.npy"))
    for path in npy_paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            unmatched.append(path.name)
            continue
        volume_id = int(match.group("volume_id"))
        if volume_id not in expected_ids:
            unexpected.append(volume_id)
        elif volume_id in discovered:
            duplicates.append(volume_id)
        else:
            discovered[volume_id] = path
    missing = sorted(set(expected_ids) - set(discovered))
    if missing or duplicates or unexpected or unmatched or len(npy_paths) != len(expected_ids):
        raise ValueError(
            f"Volume file ID mismatch; missing={missing}, duplicate={sorted(duplicates)}, "
            f"unexpected={sorted(unexpected)}, unmatched={unmatched}, "
            f"file_count={len(npy_paths)} (expected {len(expected_ids)})"
        )
    return discovered


def validate_volume_files(paths: Mapping[int, Path], expected_shape: Sequence[int]) -> None:
    """Memory-map all volume headers and validate shape and dtype."""
    for volume_id, path in sorted(paths.items()):
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        validate_real_float32_volume(volume, expected_shape, label=f"volume {volume_id} ({path.name})")


def roi_mean(volume: np.ndarray, roi: Roi) -> float:
    """Return the real-valued mean inside a validated half-open ROI."""
    roi.validate(volume.shape)
    return float(np.mean(np.real(volume[roi.selection]), dtype=np.float64))


def extract_power_normalized_spectra(
    paths: Mapping[int, Path],
    metadata: Sequence[VolumeMetadata],
    rois: Mapping[str, Roi],
    selected_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Extract ROI means sequentially and divide each value by laser power."""
    by_id = {row.volume_id: row for row in metadata}
    ids = np.asarray(list(selected_ids), dtype=np.int64)
    if any(int(volume_id) not in paths or int(volume_id) not in by_id for volume_id in ids):
        raise ValueError("Selected spectrum volume IDs are unavailable")
    wavenumbers = np.asarray([by_id[int(item)].wavenumber_cm_inv for item in ids], dtype=np.float64)
    powers = np.asarray([by_id[int(item)].power_mw for item in ids], dtype=np.float64)
    spectra = {name: np.empty(ids.size, dtype=np.float64) for name in rois}
    for position, volume_id in enumerate(ids):
        volume = np.load(paths[int(volume_id)], mmap_mode="r", allow_pickle=False)
        for name, roi in rois.items():
            spectra[name][position] = roi_mean(volume, roi) / powers[position]
    return ids, wavenumbers, spectra


def apply_wavenumber_correction(wavenumbers: np.ndarray, correction_cm_inv: float) -> np.ndarray:
    """Apply the fixed calibration correction exactly once to a new array."""
    return np.asarray(wavenumbers, dtype=np.float64) + float(correction_cm_inv)


def asls_baseline(values: np.ndarray, lam: float = 1e5, p: float = 0.01, iterations: int = 20) -> np.ndarray:
    """Estimate a baseline with asymmetric least squares smoothing."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("AsLS input must be a finite one-dimensional array with at least three values")
    if lam <= 0 or not 0 < p < 1 or iterations < 1:
        raise ValueError("AsLS requires lambda > 0, 0 < p < 1, and iterations >= 1")
    difference = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(y.size - 2, y.size), format="csc")
    penalty = lam * (difference.T @ difference)
    weights = np.ones(y.size, dtype=np.float64)
    baseline = np.zeros_like(y)
    for _ in range(iterations):
        weight_matrix = sparse.spdiags(weights, 0, y.size, y.size)
        baseline = np.asarray(spsolve(weight_matrix + penalty, weights * y))
        weights = np.where(y > baseline, p, 1.0 - p)
    return baseline


def interpolate_spectrum(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate a spectrum after sorting and rejecting duplicate x values."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    order = np.argsort(x_values)
    x_sorted = x_values[order]
    y_sorted = y_values[order]
    if np.any(np.diff(x_sorted) <= 0):
        raise ValueError("Spectrum wavenumbers must be unique")
    grid_values = np.asarray(grid, dtype=np.float64)
    if grid_values[0] < x_sorted[0] or grid_values[-1] > x_sorted[-1]:
        raise ValueError("Common grid lies outside the measured spectrum range")
    return np.interp(grid_values, x_sorted, y_sorted)


def broaden_to_fwhm(
    wavenumbers: np.ndarray,
    values: np.ndarray,
    target_fwhm_cm_inv: float,
    native_fwhm_cm_inv: float = 0.0,
) -> np.ndarray:
    """Gaussian-broaden uniformly sampled data to a target spectral FWHM."""
    x = np.asarray(wavenumbers, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    spacing = np.diff(x)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2 or not np.allclose(spacing, spacing[0]):
        raise ValueError("Broadening requires equal-length data on a uniform grid")
    if target_fwhm_cm_inv <= 0 or native_fwhm_cm_inv < 0 or native_fwhm_cm_inv > target_fwhm_cm_inv:
        raise ValueError("Resolution FWHM values must satisfy 0 <= native <= target")
    additional_fwhm = np.sqrt(target_fwhm_cm_inv**2 - native_fwhm_cm_inv**2)
    if additional_fwhm == 0:
        return y.copy()
    sigma_points = additional_fwhm / (2 * np.sqrt(2 * np.log(2))) / spacing[0]
    return gaussian_filter1d(y, sigma_points, mode="nearest")


def area_normalize(wavenumbers: np.ndarray, values: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    """Normalize by signed trapezoidal area inside a closed fit interval."""
    x = np.asarray(wavenumbers, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    mask = (x >= float(bounds[0])) & (x <= float(bounds[1]))
    if np.count_nonzero(mask) < 2:
        raise ValueError("Area-normalization interval contains fewer than two points")
    area = float(np.trapezoid(y[mask], x[mask]))
    if not np.isfinite(area) or abs(area) <= np.finfo(float).eps:
        raise ValueError("Spectrum has zero or non-finite normalization area")
    return y / area


def load_reference_spectra(path: Path, required_names: Iterable[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load wide-form reference spectra from CSV."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    if data.dtype.names is None or "wavenumber_cm_inv" not in data.dtype.names:
        raise ValueError(f"Reference CSV lacks wavenumber_cm_inv: {path}")
    if data.ndim == 0:
        data = data.reshape(1)
    names = tuple(required_names)
    missing = sorted(set(names) - set(data.dtype.names))
    if missing:
        raise ValueError(f"Reference spectra missing columns: {missing}")
    wavenumbers = np.asarray(data["wavenumber_cm_inv"], dtype=np.float64)
    spectra = {name: np.asarray(data[name], dtype=np.float64) for name in names}
    if wavenumbers.size < 3 or not np.all(np.isfinite(wavenumbers)):
        raise ValueError("Reference wavenumbers must contain at least three finite values")
    for name, values in spectra.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Reference spectrum '{name}' contains non-finite values")
    return wavenumbers, spectra


def prepare_spectrum(
    wavenumbers: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    *,
    asls_lambda: float,
    asls_p: float,
    asls_iterations: int,
    analysis_bounds: Sequence[float],
    area_bounds: Sequence[float],
    target_fwhm_cm_inv: float | None = None,
    native_fwhm_cm_inv: float = 0.0,
) -> np.ndarray:
    """Crop, baseline-correct, interpolate, optionally broaden, and normalize."""
    x = np.asarray(wavenumbers, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    analysis_mask = (x >= float(analysis_bounds[0])) & (x <= float(analysis_bounds[1]))
    if np.count_nonzero(analysis_mask) < 3:
        raise ValueError("Analysis interval contains fewer than three spectrum points")
    x = x[analysis_mask]
    y = y[analysis_mask]
    order = np.argsort(x)
    corrected = y[order] - asls_baseline(y[order], asls_lambda, asls_p, asls_iterations)
    prepared = interpolate_spectrum(x[order], corrected, grid)
    if target_fwhm_cm_inv is not None:
        prepared = broaden_to_fwhm(grid, prepared, target_fwhm_cm_inv, native_fwhm_cm_inv)
    return area_normalize(grid, prepared, area_bounds)


def fit_nonnegative_components(
    wavenumbers: np.ndarray,
    observed: np.ndarray,
    references: Mapping[str, np.ndarray],
) -> NnlsResult:
    """Fit nonnegative references plus a signed linear residual baseline."""
    x = np.asarray(wavenumbers, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    names = tuple(references)
    if not names:
        raise ValueError("At least one reference component is required")
    centered_x = x - float(np.mean(x))
    scale = float(np.ptp(x))
    if scale == 0:
        raise ValueError("Fit wavenumbers must span a non-zero interval")
    scaled_x = centered_x / scale
    reference_matrix = np.column_stack([np.asarray(references[name], dtype=np.float64) for name in names])
    design = np.column_stack((reference_matrix, np.ones_like(x), scaled_x))
    lower = np.concatenate((np.zeros(len(names)), [-np.inf, -np.inf]))
    upper = np.full(len(names) + 2, np.inf)
    solution = lsq_linear(design, y, bounds=(lower, upper), method="trf")
    if not solution.success:
        raise RuntimeError(f"NNLS fit failed: {solution.message}")
    coefficients = solution.x[: len(names)]
    total = float(np.sum(coefficients))
    ratios = np.zeros_like(coefficients) if total <= np.finfo(float).eps else coefficients / total
    baseline = design[:, -2:] @ solution.x[-2:]
    fitted = design @ solution.x
    return NnlsResult(
        component_names=names,
        coefficients=coefficients,
        ratios=ratios,
        intercept=float(solution.x[-2]),
        slope=float(solution.x[-1]),
        observed=y,
        fitted=fitted,
        baseline=baseline,
        residual=y - fitted,
    )


def make_common_grid(bounds: Sequence[float], step_cm_inv: float) -> np.ndarray:
    """Return an inclusive uniform wavenumber grid."""
    start, stop = (float(value) for value in bounds)
    if start >= stop or step_cm_inv <= 0:
        raise ValueError("Grid requires start < stop and a positive step")
    count = int(round((stop - start) / step_cm_inv))
    grid = start + np.arange(count + 1, dtype=np.float64) * step_cm_inv
    if not np.isclose(grid[-1], stop):
        raise ValueError("Grid bounds must be an integer multiple of the step")
    return grid


def portable_file_info(path: Path, root: Path) -> dict[str, Any]:
    """Describe an input file without leaking paths outside its declared root."""
    try:
        portable = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        portable = path.name
    stat = path.stat()
    return {"path": portable, "size_bytes": stat.st_size}
