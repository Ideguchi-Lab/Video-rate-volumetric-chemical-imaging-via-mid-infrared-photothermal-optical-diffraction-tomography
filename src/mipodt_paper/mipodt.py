"""Configuration and reconstruction for the paper's sequential MIP-ODT data."""

from __future__ import annotations

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
    offset_regions: None
    use_skimage_unwrap: bool
    ewald_embedding_mode: str
    integer_precision_bits: int
    float_precision_bits: int
    verbose: bool

    def validate_paper_profile(self) -> None:
        """Reject settings that differ from the accepted-paper correction profile."""
        if self.approximation != "Rytov":
            raise ValueError("The live-cell paper pipeline requires the Rytov approximation")
        if not self.use_skimage_unwrap:
            raise ValueError("The live-cell paper pipeline requires scikit-image phase unwrapping")
        if self.gradient_correction:
            raise ValueError("Gradient correction must be disabled for the accepted-paper reconstruction")
        if self.offset_regions is not None:
            raise ValueError("Offset correction must be disabled (offset_regions: null)")
        if self.ewald_embedding_mode != EwaldEmbeddingMode.TRUNCATE.value:
            raise ValueError("The live-cell paper pipeline requires legacy 'truncate' Ewald embedding")
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
    frame_interval_s: float
    image_dir: Path
    pairing_csv: Path
    optical: OpticalConfig
    algorithm: AlgorithmConfig
    output_dir: Path
    filename_pattern: str
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
            offset_regions=None,
            gradient_correction=False,
            use_skimage_unwrap=True,
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

    config = MIPODTRunConfig(
        source_path=path,
        dataset_id=str(raw["dataset_id"]),
        frame_interval_s=float(acquisition_raw["frame_interval_s"]),
        image_dir=_resolve_under(data_root, str(input_config["image_dir"]), key="input.image_dir"),
        pairing_csv=_resolve_under(REPO_ROOT, str(input_config["pairing_csv"]), key="input.pairing_csv"),
        optical=optical,
        algorithm=algorithm,
        output_dir=_resolve_under(reconstruction_root, str(output_raw["directory"]), key="output.directory"),
        filename_pattern=str(output_raw["filename_pattern"]),
        preview=preview,
    )
    if config.output_path(0).suffix != ".npy":
        raise ValueError("output.filename_pattern must produce a .npy file")
    if config.frame_interval_s <= 0:
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
    """Reconstruct one volume as real float32 ``n_OFF - n_ON``.

    ``muscopy.odt`` reconstructs the MIR-ON field relative to the MIR-OFF
    reference. In the linear approximation used here, negating its RI change
    gives the paper's public MIP convention, ``n_OFF - n_ON``.
    """
    params = config.make_odt_parameters()
    odt_config = config.make_odt_config()
    on_spectra = [_extract_spectrum(pair.on_path, config, params) for pair in pairs]
    off_spectra = [_extract_spectrum(pair.off_path, config, params) for pair in pairs]
    refractive_index, _ = odt(on_spectra, off_spectra, params, odt_config)
    delta_n = -(jnp.real(refractive_index) - params.n_sol)
    result = np.asarray(jax.device_get(delta_n), dtype=np.float32)
    expected_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1, params.freq_axial_extent_px)
    if result.shape != expected_shape:
        raise RuntimeError(f"Unexpected reconstruction shape: expected {expected_shape}, got {result.shape}")
    return result


def effective_manifest(config: MIPODTRunConfig) -> dict[str, Any]:
    """Return a serializable record of all signal-affecting settings."""
    params = config.make_odt_parameters()
    return {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "acquisition": {
            "frame_interval_s": config.frame_interval_s,
            "volume_rate_hz": 1.0 / config.frame_interval_s,
        },
        "quantity": "delta_n_mip = n_off - n_on",
        "axis_order": "x, y, z",
        "dtype": "float32",
        "expected_shape": [
            2 * params.aperturesize_px + 1,
            2 * params.aperturesize_px + 1,
            params.freq_axial_extent_px,
        ],
        "configuration": {
            "path": str(config.source_path),
        },
        "input": {
            "image_dir": str(config.image_dir),
            "pairing_csv": str(config.pairing_csv),
            "legacy_pairing_semantics": "ImagePathA=MIR-ON, ImagePathB=MIR-OFF",
        },
        "optical": dataclasses.asdict(config.optical),
        "reconstruction": dataclasses.asdict(config.algorithm),
        "derived_sampling": {
            "lateral_m_per_px": float(params.imgpx_lateral_m_per_px),
            "axial_m_per_px": float(params.imgpx_axial_m_per_px),
        },
    }
