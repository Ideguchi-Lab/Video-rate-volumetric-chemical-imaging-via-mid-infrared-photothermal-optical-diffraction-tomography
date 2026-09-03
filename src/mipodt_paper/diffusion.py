"""Figure 4 particle detection, tracking, and diffusion helpers.

Particle linking and total 3D MSD analysis are delegated to the commit-pinned
:mod:`pt3d` submodule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import yaml
from pt3d import DetectionConfig, InputConfig, PipelineConfig, PostprocessConfig, TrackingConfig, VoxelSize
from pt3d.analysis import DiffusionAnalysisConfig, MSDConfig, analyze_diffusion
from pt3d.pipeline import PipelineResult
from pt3d.postprocess import compute_track_stats, postprocess
from pt3d.track import link_detections
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class SubsetMapping:
    """Map annotation-subset coordinates to the full cropped time series."""

    region_name: str
    frame_indices: np.ndarray
    z_indices: np.ndarray


def read_subset_mapping(path: Path, region_name: str) -> SubsetMapping:
    """Read the frozen annotation subset manifest."""
    with path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    regions = [entry for entry in manifest.get("regions", []) if entry.get("name") == region_name]
    if len(regions) != 1:
        raise ValueError(f"Expected exactly one region named {region_name!r} in {path}")
    frames = np.asarray(manifest.get("frame_indices", []), dtype=np.int32)
    z_indices = np.asarray(regions[0].get("subset_z_indices_relative", []), dtype=np.int32)
    if frames.size == 0 or z_indices.size == 0:
        raise ValueError(f"Empty frame or z mapping in {path}")
    return SubsetMapping(region_name, frames, z_indices)


def volume_paths(directory: Path, pattern: str, frame_ids: Sequence[int]) -> list[Path]:
    """Resolve reconstructed volume paths and report every missing frame."""
    paths = [directory / pattern.format(frame_id) for frame_id in frame_ids]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Missing {len(missing)} reconstructed volume(s): {preview}")
    return paths


def validate_volume(path: Path, expected_shape: Sequence[int]) -> None:
    """Validate the public real-float32 ``(x, y, z)`` volume contract."""
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    if volume.shape != tuple(expected_shape):
        raise ValueError(f"Unexpected shape for {path}: {volume.shape}, expected {tuple(expected_shape)}")
    if volume.dtype != np.float32 or np.iscomplexobj(volume):
        raise ValueError(f"Expected a real float32 array at {path}, got {volume.dtype}")


def crop_xyz_to_zyx(volume: np.ndarray, crop: Mapping[str, Sequence[int]]) -> np.ndarray:
    """Transpose an ``(x, y, z)`` reconstruction and apply the frozen crop."""
    zyx = np.transpose(np.real(volume), (2, 1, 0))
    return zyx[
        slice(*crop["z"]),
        slice(*crop["y"]),
        slice(*crop["x"]),
    ]


def export_normalized_crop(
    paths: Sequence[Path],
    output_path: Path,
    crop: Mapping[str, Sequence[int]],
    percentiles: Sequence[float],
    dataset_name: str = "data",
) -> Path:
    """Refactor the original two-pass percentile normalization into one helper."""
    low, high = float("inf"), float("-inf")
    for path in paths:
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        frame = crop_xyz_to_zyx(volume, crop)
        frame_low, frame_high = np.percentile(frame, percentiles)
        low, high = min(low, float(frame_low)), max(high, float(frame_high))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid normalization range [{low}, {high}]")

    shape = crop_xyz_to_zyx(np.load(paths[0], mmap_mode="r", allow_pickle=False), crop).shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        dataset = handle.create_dataset(
            dataset_name,
            shape=(len(paths), *shape),
            dtype=np.float32,
            chunks=(1, *shape),
            compression="gzip",
        )
        for frame_index, path in enumerate(paths):
            volume = np.load(path, mmap_mode="r", allow_pickle=False)
            frame = crop_xyz_to_zyx(volume, crop)
            dataset[frame_index] = np.clip((frame - low) / (high - low), 0.0, 1.0).astype(np.float32)
        dataset.attrs["axis_order"] = "tzyx"
        handle.attrs["normalization_percentiles"] = np.asarray(percentiles, dtype=np.float64)
        handle.attrs["normalization_range"] = np.asarray([low, high], dtype=np.float64)
    return output_path


def build_feature_volume(volume_zyx: np.ndarray, scales: Sequence[float]) -> np.ndarray:
    """Build the raw/Gaussian/LoG/gradient feature stack used in the original RF fallback."""
    volume = volume_zyx.astype(np.float32, copy=False)
    features = [volume]
    for sigma in scales:
        features.extend(
            [
                ndi.gaussian_filter(volume, sigma=sigma).astype(np.float32, copy=False),
                ndi.gaussian_laplace(volume, sigma=sigma).astype(np.float32, copy=False),
                ndi.gaussian_gradient_magnitude(volume, sigma=sigma).astype(np.float32, copy=False),
            ]
        )
    return np.stack(features, axis=-1)


def _sample_training_coordinates(
    labels: np.ndarray,
    positive_label: int,
    max_positive: int,
    max_negative: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive = np.argwhere(labels == positive_label)
    if positive.size == 0:
        raise ValueError(f"No positive voxels with label {positive_label}")
    positive_frames = np.unique(positive[:, 0])
    negative_all = np.argwhere(labels != positive_label)
    negative = negative_all[np.isin(negative_all[:, 0], positive_frames)]
    if negative.size == 0:
        raise ValueError("No negative training voxels")
    positive = positive[rng.choice(len(positive), min(max_positive, len(positive)), replace=False)]
    negative = negative[rng.choice(len(negative), min(max_negative, len(negative)), replace=False)]
    coordinates = np.vstack([positive, negative])
    targets = np.concatenate([np.ones(len(positive), dtype=np.uint8), np.zeros(len(negative), dtype=np.uint8)])
    order = rng.permutation(len(targets))
    return coordinates[order], targets[order]


def regenerate_probabilities(
    normalized_path: Path,
    labels_path: Path,
    manifest_path: Path,
    output_path: Path,
    settings: Mapping[str, Any],
) -> Path:
    """Regenerate the frozen probability-map format from public labels and volumes."""
    mapping = read_subset_mapping(manifest_path, str(settings["region_name"]))
    with h5py.File(labels_path, "r") as labels_handle:
        labels = np.asarray(labels_handle[str(settings["labels_dataset"])])
    expected_labels_shape = (len(mapping.frame_indices), len(mapping.z_indices))
    if labels.shape[:2] != expected_labels_shape:
        raise ValueError(f"Labels shape {labels.shape} does not match subset mapping {expected_labels_shape}")

    scales = [float(value) for value in settings["feature_scales"]]
    seed = int(settings["random_seed"])
    coordinates, targets = _sample_training_coordinates(
        labels,
        int(settings["positive_label"]),
        int(settings["max_positive_samples"]),
        int(settings["max_negative_samples"]),
        seed,
    )
    with h5py.File(normalized_path, "r") as normalized_handle:
        source = normalized_handle[str(settings["input_dataset"])]
        groups: dict[int, list[tuple[int, int, int, int]]] = {}
        for index, (t_subset, z_subset, y, x) in enumerate(coordinates):
            t_full = int(mapping.frame_indices[int(t_subset)])
            z_full = int(mapping.z_indices[int(z_subset)])
            groups.setdefault(t_full, []).append((z_full, int(y), int(x), int(targets[index])))
        feature_parts, target_parts = [], []
        for frame in sorted(groups):
            features = build_feature_volume(np.asarray(source[frame]), scales)
            rows = np.asarray(groups[frame], dtype=np.int32)
            feature_parts.append(features[rows[:, 0], rows[:, 1], rows[:, 2]])
            target_parts.append(rows[:, 3].astype(np.uint8))
        classifier = RandomForestClassifier(
            n_estimators=int(settings["n_estimators"]),
            max_depth=settings.get("max_depth"),
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        classifier.fit(np.concatenate(feature_parts), np.concatenate(target_parts))
        positive_column = list(classifier.classes_).index(1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as output_handle:
            out = output_handle.create_dataset(
                str(settings["output_dataset"]),
                shape=(*source.shape, 2),
                dtype=np.float32,
                chunks=(1, min(4, source.shape[1]), source.shape[2], source.shape[3], 2),
                compression="gzip",
            )
            for frame in range(source.shape[0]):
                features = build_feature_volume(np.asarray(source[frame]), scales)
                probability = classifier.predict_proba(features.reshape(-1, features.shape[-1]))[:, positive_column]
                probability = probability.reshape(source.shape[1:]).astype(np.float32)
                out[frame, ..., 1] = probability
                out[frame, ..., 0] = 1.0 - probability
            output_handle.attrs["axis_order"] = "tzyxc"
            output_handle.attrs["channel_meaning"] = "[background, lipid_droplet]"
            output_handle.attrs["prediction_type"] = "rf_from_manual_labels"
    return output_path


def validate_probability_file(path: Path, dataset_name: str, expected_shape: Sequence[int]) -> None:
    """Validate the frozen probability HDF5 contract without loading it in full."""
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise ValueError(f"Dataset {dataset_name!r} not found in {path}")
        dataset = handle[dataset_name]
        if dataset.shape != tuple(expected_shape):
            raise ValueError(f"Unexpected probability shape {dataset.shape}; expected {tuple(expected_shape)}")
        if dataset.dtype != np.float32:
            raise ValueError(f"Expected float32 probabilities, got {dataset.dtype}")


def extract_detections(
    path: Path,
    dataset_name: str,
    class_channel: int,
    threshold: float,
    min_voxels: int,
    connectivity: int,
) -> pd.DataFrame:
    """Extract probability-weighted connected-component centroids frame by frame."""
    structure = ndi.generate_binary_structure(3, connectivity)
    rows: list[dict[str, float | int]] = []
    with h5py.File(path, "r") as handle:
        source = handle[dataset_name]
        for frame_index in range(source.shape[0]):
            probability = np.asarray(source[frame_index, ..., class_channel], dtype=np.float32)
            labels, count = ndi.label(probability >= threshold, structure=structure)
            if count == 0:
                continue
            ids = np.arange(1, count + 1)
            sizes = ndi.sum(np.ones_like(probability), labels, ids)
            masses = ndi.sum(probability, labels, ids)
            maxima = ndi.maximum(probability, labels, ids)
            centroids = ndi.center_of_mass(probability, labels, ids)
            for index, centroid in enumerate(centroids):
                size = int(sizes[index])
                if size < min_voxels or not np.all(np.isfinite(centroid)):
                    continue
                rows.append(
                    {
                        "frame": frame_index,
                        "z": float(centroid[0]),
                        "y": float(centroid[1]),
                        "x": float(centroid[2]),
                        "mass": float(masses[index]),
                        "size": size,
                        "mean_prob": float(masses[index]) / size,
                        "max_prob": float(maxima[index]),
                    }
                )
    return pd.DataFrame(rows, columns=["frame", "z", "y", "x", "mass", "size", "mean_prob", "max_prob"])


def make_pipeline_config(settings: Mapping[str, Any]) -> PipelineConfig:
    """Build the commit-pinned pt3d configuration from the Figure 4 YAML."""
    voxel = settings["voxel_size_um"]
    return PipelineConfig(
        input=InputConfig(
            axis_order="tzyx",
            voxel_size=VoxelSize(z_um=float(voxel["z"]), y_um=float(voxel["y"]), x_um=float(voxel["x"])),
        ),
        detection=DetectionConfig(diameter=(5, 7, 7)),
        tracking=TrackingConfig(
            search_range_um=float(settings["tracking"]["search_range_um"]),
            memory=int(settings["tracking"]["memory"]),
        ),
        postprocess=PostprocessConfig(min_track_length=int(settings["tracking"]["min_track_length"])),
    )


def link_and_analyze(detections: pd.DataFrame, settings: Mapping[str, Any]) -> tuple[pd.DataFrame, Any]:
    """Run the unchanged pt3d linker, postprocessing, and 3D MSD implementation."""
    config = make_pipeline_config(settings)
    tracks = link_detections(detections, config.tracking, config.input.voxel_size)
    tracks = postprocess(tracks, config.postprocess, config.input.voxel_size)
    result = PipelineResult(
        detections=detections,
        tracks=tracks,
        track_stats=compute_track_stats(tracks, config.input.voxel_size),
        config=config,
        input_info={"source": "frozen probability map", "axis_order": "tzyx"},
    )
    diffusion = settings["diffusion"]
    analysis = analyze_diffusion(
        result,
        DiffusionAnalysisConfig(
            min_track_length=int(settings["tracking"]["min_track_length"]),
            dt=float(settings["frame_interval_s"]),
            interpolate_gaps=False,
            msd=MSDConfig(
                max_lag_fraction=float(diffusion["max_lag_fraction"]),
                fit_range_fraction=tuple(float(value) for value in diffusion["fit_range_fraction"]),
            ),
        ),
    )
    valid = analysis.particle_results["alpha"] >= float(diffusion["minimum_alpha"])
    valid_ids = set(analysis.particle_results.loc[valid, "particle"].astype(int))
    analysis.particle_results = analysis.particle_results.loc[valid].reset_index(drop=True)
    analysis.msd_per_particle = {
        pid: value for pid, value in analysis.msd_per_particle.items() if int(pid) in valid_ids
    }
    analysis.positions_um = {pid: value for pid, value in analysis.positions_um.items() if int(pid) in valid_ids}
    return tracks, analysis


def analyze_cached_tracks(tracks: pd.DataFrame, settings: Mapping[str, Any]) -> Any:
    """Analyze already linked tracks without relinking them."""
    config = make_pipeline_config(settings)
    result = PipelineResult(
        detections=pd.DataFrame(),
        tracks=tracks,
        track_stats=compute_track_stats(tracks, config.input.voxel_size),
        config=config,
        input_info={"source": "tracks.csv"},
    )
    diffusion = settings["diffusion"]
    analysis = analyze_diffusion(
        result,
        DiffusionAnalysisConfig(
            min_track_length=int(settings["tracking"]["min_track_length"]),
            dt=float(settings["frame_interval_s"]),
            interpolate_gaps=False,
            msd=MSDConfig(
                max_lag_fraction=float(diffusion["max_lag_fraction"]),
                fit_range_fraction=tuple(float(value) for value in diffusion["fit_range_fraction"]),
            ),
        ),
    )
    valid_ids = set(
        analysis.particle_results.loc[
            analysis.particle_results["alpha"] >= float(diffusion["minimum_alpha"]), "particle"
        ].astype(int)
    )
    analysis.particle_results = analysis.particle_results[
        analysis.particle_results["particle"].astype(int).isin(valid_ids)
    ].reset_index(drop=True)
    analysis.msd_per_particle = {
        pid: value for pid, value in analysis.msd_per_particle.items() if int(pid) in valid_ids
    }
    analysis.positions_um = {pid: value for pid, value in analysis.positions_um.items() if int(pid) in valid_ids}
    return analysis


def particle_track(tracks: pd.DataFrame, particle_id: int) -> pd.DataFrame:
    """Return one frame-sorted track and reject ambiguous duplicate frames."""
    track = tracks.loc[tracks["particle"].astype(int) == particle_id].sort_values("frame").copy()
    if track.empty:
        raise ValueError(f"Particle {particle_id} is absent from the tracking result")
    if track["frame"].duplicated().any():
        raise ValueError(f"Particle {particle_id} has duplicate frame indices")
    return track


def component_msd(track: pd.DataFrame, voxel_size_um: Mapping[str, float], dt: float, max_lag: int) -> pd.DataFrame:
    """Compute the component-wise and total MSD table used by the original selected-track script."""
    coordinates = track[["x", "y", "z"]].to_numpy(dtype=float)
    coordinates = coordinates * np.asarray([voxel_size_um["x"], voxel_size_um["y"], voxel_size_um["z"]])
    records = []
    for lag in range(1, min(max_lag, len(coordinates) - 1) + 1):
        displacement = coordinates[lag:] - coordinates[:-lag]
        squared = displacement**2
        records.append(
            {
                "lag_frame": lag,
                "lag_s": lag * dt,
                "n_pairs": len(displacement),
                "msd_x_um2": np.mean(squared[:, 0]),
                "msd_y_um2": np.mean(squared[:, 1]),
                "msd_z_um2": np.mean(squared[:, 2]),
                "msd_xy_um2": np.mean(squared[:, 0] + squared[:, 1]),
                "msd_3d_um2": np.mean(np.sum(squared, axis=1)),
                "sem_3d_um2": np.std(np.sum(squared, axis=1)) / np.sqrt(len(displacement)),
            }
        )
    return pd.DataFrame(records)


def fit_power_law(
    time: np.ndarray, msd: np.ndarray, fit_range: Sequence[float], divisor: float
) -> dict[str, float | int]:
    """Fit ``MSD = divisor * D * tau**alpha`` using the frozen pt3d lag convention."""
    n_with_zero = len(msd) + 1
    start_lag = max(1, int(n_with_zero * float(fit_range[0])))
    end_lag = min(n_with_zero, max(start_lag + 2, int(n_with_zero * float(fit_range[1]))))
    start, stop = start_lag - 1, end_lag - 1
    valid = np.isfinite(msd[start:stop]) & (msd[start:stop] > 0)
    coefficients = np.polyfit(np.log(time[start:stop][valid]), np.log(msd[start:stop][valid]), 1)
    return {
        "alpha": float(coefficients[0]),
        "D": float(np.exp(coefficients[1]) / divisor),
        "amplitude": float(np.exp(coefficients[1])),
        "fit_lag_start_inclusive": start_lag,
        "fit_lag_end_exclusive": end_lag,
        "n_fit_points": int(np.count_nonzero(valid)),
    }


def case_bootstrap(
    msd: np.ndarray,
    dt: float,
    fit_range: Sequence[float],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> dict[str, float | int]:
    """Case-bootstrap log-log fit points, preserving the original per-particle method."""
    n = len(msd)
    start = max(1, int(n * float(fit_range[0])))
    stop = min(n, max(start + 2, int(n * float(fit_range[1]))))
    lag = np.arange(start, stop, dtype=float)
    values = np.asarray(msd[start:stop], dtype=float)
    valid = np.isfinite(values) & (values > 0)
    x, y = np.log(lag[valid] * dt), np.log(values[valid])
    if len(x) < 3:
        raise ValueError("At least three fit points are required for case bootstrap")
    rng = np.random.default_rng(seed)
    alpha_samples, d_samples = [], []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(x), size=len(x))
        if np.allclose(x[indices], x[indices][0]):
            continue
        slope, intercept = np.polyfit(x[indices], y[indices], 1)
        coefficient = np.exp(intercept) / 6.0
        if np.isfinite(slope) and np.isfinite(coefficient) and coefficient > 0:
            alpha_samples.append(slope)
            d_samples.append(coefficient)
    low = 100.0 * (1.0 - ci_level) / 2.0
    high = 100.0 - low
    alpha_array, d_array = np.asarray(alpha_samples), np.asarray(d_samples)
    return {
        "n_boot_used": len(alpha_array),
        "alpha_case_boot_median": float(np.median(alpha_array)),
        "alpha_ci_lo_case_boot": float(np.percentile(alpha_array, low)),
        "alpha_ci_hi_case_boot": float(np.percentile(alpha_array, high)),
        "D_case_boot_median": float(np.median(d_array)),
        "D_ci_lo_case_boot": float(np.percentile(d_array, low)),
        "D_ci_hi_case_boot": float(np.percentile(d_array, high)),
    }


def bootstrap_table(analysis: Any, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Compute the frozen case-bootstrap summary for every accepted particle."""
    bootstrap = settings["bootstrap"]
    records = []
    for particle_id, result in sorted(analysis.msd_per_particle.items()):
        record = case_bootstrap(
            result.msd,
            float(settings["frame_interval_s"]),
            settings["diffusion"]["fit_range_fraction"],
            int(bootstrap["n_resamples"]),
            float(bootstrap["ci_level"]),
            int(bootstrap["random_seed"]) + int(particle_id),
        )
        records.append({"particle": int(particle_id), "D": result.D, "alpha": result.alpha, **record})
    return pd.DataFrame(records).sort_values("particle").reset_index(drop=True)
