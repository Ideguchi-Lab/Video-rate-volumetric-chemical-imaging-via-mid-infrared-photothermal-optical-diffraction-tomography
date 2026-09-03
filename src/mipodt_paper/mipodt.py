"""Configuration and reconstruction for the paper's sequential MIP-ODT data."""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tifffile
import yaml
from muscopy.cfg import ArrayPrecision
from muscopy.dh import get_spectrum
from muscopy.odt import EwaldEmbeddingMode, ODTConfig, ODTParameters, odt

from mipodt_paper.pairing import MIPPair
from mipodt_paper.paths import DATA_DIR, RECONSTRUCTION_DIR, REPO_ROOT

REQUIRED_MUSCOPY_VERSION = "1.0.1"


@dataclass(frozen=True, slots=True)
class OpticalConfig:
    """Physical acquisition parameters."""

    objective_na: float
    illumination_na: float
    wavelength_nm: float
    camera_pixel_size_um: float
    magnification: float
    medium_refractive_index: float
    image_size_px: int
    offaxis_center_px: tuple[int, int]

    @property
    def effective_pixel_size_m(self) -> float:
        return self.camera_pixel_size_um * 1e-6 / self.magnification


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    """Signal-fidelity settings used for the accepted-paper reconstruction."""

    approximation: str
    linear_approximation: bool
    hermite_symmetry: bool
    edge_size_px: int
    gradient_correction: bool
    offset_regions: Any
    use_skimage_unwrap: bool
    ewald_embedding_mode: str
    integer_precision_bits: int
    float_precision_bits: int
    verbose: bool

    def validate_paper_profile(self) -> None:
        """Validate explicitly recorded reconstruction settings."""
        if self.approximation != "Rytov":
            raise ValueError("The paper reconstruction pipeline requires the Rytov approximation")
        if self.gradient_correction:
            raise ValueError("Gradient correction must be disabled for the paper reconstruction")
        if self.offset_regions is not None:
            raise ValueError("Offset correction must be disabled (offset_regions: null)")
        if self.ewald_embedding_mode != EwaldEmbeddingMode.TRUNCATE.value:
            raise ValueError("The paper reconstruction pipeline requires legacy 'truncate' Ewald embedding")
        if self.edge_size_px < 0:
            raise ValueError("edge_size_px must be non-negative")


@dataclass(frozen=True, slots=True)
class PreviewConfig:
    """Default quick-look view used for the paper data."""

    volume_id: int
    z_indices: tuple[int, ...]
    colormap: str
    vmin: float | None
    vmax: float | None
    formats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MIPODTRunConfig:
    """Fully resolved reconstruction configuration."""

    source_path: Path
    dataset_id: str
    frame_interval_s: float | None
    image_dir: Path
    pairing_csv: Path
    volume_manifest_csv: Path | None
    default_volume_ids: tuple[int, ...] | None
    optical: OpticalConfig
    algorithm: AlgorithmConfig
    output_dir: Path
    filename_pattern: str
    ri_change_sign: float
    preview: PreviewConfig

    def output_path(self, volume_id: int) -> Path:
        return self.output_dir / self.filename_pattern.format(volume_id=volume_id)

    def make_odt_parameters(self) -> ODTParameters:
        return ODTParameters(
            na=self.optical.objective_na,
            wavelength_m=self.optical.wavelength_nm * 1e-9,
            img_size_px=self.optical.image_size_px,
            px_size_m=self.optical.effective_pixel_size_m,
            n_sol=self.optical.medium_refractive_index,
            na_illumination=self.optical.illumination_na,
        )

    def make_odt_config(self) -> ODTConfig:
        self.algorithm.validate_paper_profile()
        return ODTConfig(
            approx_type=self.algorithm.approximation,
            linear_approx=self.algorithm.linear_approximation,
            hermite_symmetry=self.algorithm.hermite_symmetry,
            precision=ArrayPrecision(
                int_length=self.algorithm.integer_precision_bits,
                float_length=self.algorithm.float_precision_bits,
            ),
            edge_size=self.algorithm.edge_size_px,
            offset_regions=self.algorithm.offset_regions,
            gradient_correction=self.algorithm.gradient_correction,
            use_skimage_unwrap=self.algorithm.use_skimage_unwrap,
            ewald_embedding_mode=EwaldEmbeddingMode(self.algorithm.ewald_embedding_mode),
            verbose=self.algorithm.verbose,
        )


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration key '{key}' must be a mapping")
    return value


def _resolve_under(root: Path, value: str, *, key: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Configuration path '{key}' escapes its root: {value}") from exc
    return resolved


def _load_volume_ids(path: Path) -> tuple[int, ...]:
    """Load the explicitly analyzed volume IDs from a dataset manifest."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "volume_id" not in reader.fieldnames:
            raise ValueError(f"Volume manifest must contain a 'volume_id' column: {path}")
        volume_ids = tuple(int(row["volume_id"]) for row in reader)
    if not volume_ids:
        raise ValueError(f"Volume manifest is empty: {path}")
    if any(volume_id < 0 for volume_id in volume_ids):
        raise ValueError(f"Volume IDs must be non-negative: {path}")
    if len(set(volume_ids)) != len(volume_ids):
        raise ValueError(f"Volume manifest contains duplicate volume IDs: {path}")
    return volume_ids


def load_mipodt_config(
    config_path: str | Path,
    *,
    data_root: Path = DATA_DIR,
    reconstruction_root: Path = RECONSTRUCTION_DIR,
) -> MIPODTRunConfig:
    """Load and validate a sequential MIP-ODT YAML configuration."""
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported or missing schema_version; expected 1")

    input_config = _mapping(raw, "input")
    acquisition_raw = _mapping(raw, "acquisition")
    optical_raw = _mapping(raw, "optical")
    algorithm_raw = _mapping(raw, "reconstruction")
    output_raw = _mapping(raw, "output")
    preview_raw = _mapping(raw, "preview")

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
    if len(optical.offaxis_center_px) != 2:
        raise ValueError("offaxis_center_px must contain exactly two integers")

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

    preview = PreviewConfig(
        volume_id=int(preview_raw["volume_id"]),
        z_indices=tuple(int(value) for value in preview_raw["z_indices"]),
        colormap=str(preview_raw["colormap"]),
        vmin=None if preview_raw.get("vmin") is None else float(preview_raw["vmin"]),
        vmax=None if preview_raw.get("vmax") is None else float(preview_raw["vmax"]),
        formats=tuple(str(value) for value in preview_raw.get("formats", ["png"])),
    )

    volume_manifest_value = input_config.get("volume_manifest_csv")
    volume_manifest_csv = (
        None
        if volume_manifest_value is None
        else _resolve_under(REPO_ROOT, str(volume_manifest_value), key="input.volume_manifest_csv")
    )
    frame_interval_value = acquisition_raw.get("frame_interval_s")

    config = MIPODTRunConfig(
        source_path=path,
        dataset_id=str(raw["dataset_id"]),
        frame_interval_s=None if frame_interval_value is None else float(frame_interval_value),
        image_dir=_resolve_under(data_root, str(input_config["image_dir"]), key="input.image_dir"),
        pairing_csv=_resolve_under(REPO_ROOT, str(input_config["pairing_csv"]), key="input.pairing_csv"),
        volume_manifest_csv=volume_manifest_csv,
        default_volume_ids=None if volume_manifest_csv is None else _load_volume_ids(volume_manifest_csv),
        optical=optical,
        algorithm=algorithm,
        output_dir=_resolve_under(reconstruction_root, str(output_raw["directory"]), key="output.directory"),
        filename_pattern=str(output_raw["filename_pattern"]),
        ri_change_sign=float(output_raw.get("ri_change_sign", -1.0)),
        preview=preview,
    )
    if config.output_path(0).suffix != ".npy":
        raise ValueError("output.filename_pattern must produce a .npy file")
    if config.ri_change_sign not in (-1.0, 1.0):
        raise ValueError("output.ri_change_sign must be -1 or 1")
    if config.frame_interval_s is not None and config.frame_interval_s <= 0:
        raise ValueError("acquisition.frame_interval_s must be positive")
    return config


def validate_muscopy_version() -> None:
    """Fail before a large reconstruction when muscopy is not v1.0.1."""
    installed = version("muscopy")
    if installed != REQUIRED_MUSCOPY_VERSION:
        raise RuntimeError(f"This reconstruction requires muscopy {REQUIRED_MUSCOPY_VERSION}, found {installed}")


def load_hologram(path: Path, expected_size_px: int) -> jax.Array:
    """Load one square TIFF hologram as a float32 JAX array."""
    image = np.asarray(tifffile.imread(path), dtype=np.float32)
    expected_shape = (expected_size_px, expected_size_px)
    if image.shape != expected_shape:
        raise ValueError(f"Expected hologram shape {expected_shape}, got {image.shape}: {path}")
    return jnp.asarray(image)


def _extract_spectrum(path: Path, config: MIPODTRunConfig, params: ODTParameters) -> jax.Array:
    hologram = load_hologram(path, config.optical.image_size_px)
    fourier_hologram = jnp.fft.fftshift(jnp.fft.fft2(hologram))
    return get_spectrum(fourier_hologram, params, config.optical.offaxis_center_px)


def reconstruct_mipodt_volume(config: MIPODTRunConfig, pairs: list[MIPPair]) -> np.ndarray:
    """Reconstruct one real float32 RI-change volume.

    ``muscopy.odt`` reconstructs the MIR-ON field relative to the MIR-OFF
    reference. ``output.ri_change_sign`` records which sign of
    ``n_reconstructed - n_medium`` is saved for each acquisition.
    """
    params = config.make_odt_parameters()
    odt_config = config.make_odt_config()
    on_spectra = [_extract_spectrum(pair.on_path, config, params) for pair in pairs]
    off_spectra = [_extract_spectrum(pair.off_path, config, params) for pair in pairs]
    refractive_index, _ = odt(on_spectra, off_spectra, params, odt_config)
    delta_n = config.ri_change_sign * (jnp.real(refractive_index) - params.n_sol)
    result = np.asarray(jax.device_get(delta_n), dtype=np.float32)
    expected_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1, params.freq_axial_extent_px)
    if result.shape != expected_shape:
        raise RuntimeError(f"Unexpected reconstruction shape: expected {expected_shape}, got {result.shape}")
    return result


def reconstruct_static_odt_volume(
    optical: OpticalConfig,
    algorithm: AlgorithmConfig,
    sample_paths: list[Path],
    reference_paths: list[Path],
) -> np.ndarray:
    """Reconstruct ``delta_n = n_sample - n_medium`` from paired TIFF holograms."""
    if not sample_paths or len(sample_paths) != len(reference_paths):
        raise ValueError("Static ODT requires equal non-zero numbers of sample and reference holograms")
    params = ODTParameters(
        na=optical.objective_na,
        wavelength_m=optical.wavelength_nm * 1e-9,
        img_size_px=optical.image_size_px,
        px_size_m=optical.effective_pixel_size_m,
        n_sol=optical.medium_refractive_index,
        na_illumination=optical.illumination_na,
    )
    algorithm.validate_paper_profile()
    odt_config = ODTConfig(
        approx_type=algorithm.approximation,
        linear_approx=algorithm.linear_approximation,
        hermite_symmetry=algorithm.hermite_symmetry,
        precision=ArrayPrecision(
            int_length=algorithm.integer_precision_bits,
            float_length=algorithm.float_precision_bits,
        ),
        edge_size=algorithm.edge_size_px,
        offset_regions=algorithm.offset_regions,
        gradient_correction=algorithm.gradient_correction,
        use_skimage_unwrap=algorithm.use_skimage_unwrap,
        ewald_embedding_mode=EwaldEmbeddingMode(algorithm.ewald_embedding_mode),
        verbose=algorithm.verbose,
    )

    def extract(path: Path) -> jax.Array:
        hologram = load_hologram(path, optical.image_size_px)
        fourier_hologram = jnp.fft.fftshift(jnp.fft.fft2(hologram))
        return get_spectrum(fourier_hologram, params, optical.offaxis_center_px)

    sample_spectra = [extract(path) for path in sample_paths]
    reference_spectra = [extract(path) for path in reference_paths]
    refractive_index, _ = odt(sample_spectra, reference_spectra, params, odt_config)
    result = np.asarray(jax.device_get(jnp.real(refractive_index) - params.n_sol), dtype=np.float32)
    expected_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1, params.freq_axial_extent_px)
    if result.shape != expected_shape:
        raise RuntimeError(f"Unexpected reconstruction shape: expected {expected_shape}, got {result.shape}")
    return result


def _portable_path(path: Path, root: Path) -> str:
    """Prefer a path relative to its configured root in run manifests."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def effective_manifest(config: MIPODTRunConfig) -> dict[str, Any]:
    """Return a serializable record of all signal-affecting settings."""
    params = config.make_odt_parameters()
    return {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "acquisition": (
            None
            if config.frame_interval_s is None
            else {
                "frame_interval_s": config.frame_interval_s,
                "volume_rate_hz": 1.0 / config.frame_interval_s,
            }
        ),
        "quantity": f"delta_n_mip = {config.ri_change_sign:+g} * (n_reconstructed - n_medium)",
        "axis_order": "x, y, z",
        "dtype": "float32",
        "expected_shape": [
            2 * params.aperturesize_px + 1,
            2 * params.aperturesize_px + 1,
            params.freq_axial_extent_px,
        ],
        "configuration": {
            "path": _portable_path(config.source_path, REPO_ROOT),
        },
        "input": {
            "image_dir": _portable_path(config.image_dir, DATA_DIR),
            "pairing_csv": _portable_path(config.pairing_csv, REPO_ROOT),
            "volume_manifest_csv": (
                None if config.volume_manifest_csv is None else _portable_path(config.volume_manifest_csv, REPO_ROOT)
            ),
            "legacy_pairing_semantics": "ImagePathA=MIR-ON, ImagePathB=MIR-OFF",
        },
        "optical": dataclasses.asdict(config.optical),
        "reconstruction": dataclasses.asdict(config.algorithm),
        "derived_sampling": {
            "lateral_m_per_px": float(params.imgpx_lateral_m_per_px),
            "axial_m_per_px": float(params.imgpx_axial_m_per_px),
        },
    }
