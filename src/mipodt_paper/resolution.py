"""Resolution-profile extraction and the Gaussian fit used for Fig. 2.

The fitting routine intentionally preserves the parameterization of the
original ``resolution_evaluation`` script.  Changing optimizers or bounds can
move the fitted FWHM by several nanometres for these short profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True, slots=True)
class ProfileData:
    positions_m: jax.Array
    values: jax.Array
    direction: str


@dataclass(frozen=True, slots=True)
class GaussianFitResult:
    amplitude: float
    center_m: float
    sigma_m: float
    offset: float
    fwhm_m: float
    success: bool
    residual: float

    @property
    def fwhm_nm(self) -> float:
        return self.fwhm_m * 1e9


@jit
def gaussian_1d(x: jax.Array, params: jax.Array) -> jax.Array:
    amplitude, center, sigma, offset = params
    return amplitude * jnp.exp(-0.5 * ((x - center) / sigma) ** 2) + offset


def extract_profiles(
    data: np.ndarray,
    center: tuple[int, int, int],
    ranges: tuple[int, int, int],
    pixel_sizes_m: tuple[float, float, float],
    *,
    value_offset: float = 0.0,
) -> tuple[ProfileData, ProfileData, ProfileData]:
    """Extract the inclusive X/Y/Z profiles used by the original analysis."""
    cx, cy, cz = center
    rx, ry, rz = ranges
    px_x, px_y, px_z = pixel_sizes_m

    def values(selection: tuple[object, object, object]) -> jax.Array:
        selected = jnp.asarray(np.real(data[selection]), dtype=jnp.float32)
        return selected + value_offset if value_offset else selected

    return (
        ProfileData(
            jnp.arange(-rx, rx + 1) * px_x,
            values((slice(cx - rx, cx + rx + 1), cy, cz)),
            "X",
        ),
        ProfileData(
            jnp.arange(-ry, ry + 1) * px_y,
            values((cx, slice(cy - ry, cy + ry + 1), cz)),
            "Y",
        ),
        ProfileData(
            jnp.arange(-rz, rz + 1) * px_z,
            values((cx, cy, slice(cz - rz, cz + rz + 1))),
            "Z",
        ),
    )


def fit_gaussian_1d(profile: ProfileData) -> GaussianFitResult:
    """Fit one profile using the exact bounded transform of the legacy code."""
    x = jnp.asarray(profile.positions_m, dtype=jnp.float64)
    y = jnp.asarray(profile.values, dtype=jnp.float64)

    scale = float(jnp.max(jnp.abs(x)))
    if scale < 1e-12:
        scale = 1.0
    x_work = x / scale

    y_min = float(jnp.min(y))
    y_max = float(jnp.max(y))
    amplitude_init = y_max - y_min
    offset_init = y_min
    max_idx = int(jnp.argmax(y))
    center_init = float(x_work[max_idx])

    if 0 < max_idx < len(x_work) - 1 and len(x_work) > 2:
        y0 = float(y[max_idx - 1])
        y1 = float(y[max_idx])
        y2 = float(y[max_idx + 1])
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-20:
            step = float(x_work[1] - x_work[0])
            center_init = float(x_work[max_idx] + 0.5 * (y0 - y2) / denom * step)

    def loss_fn(params: jax.Array) -> jax.Array:
        return jnp.sum((y - gaussian_1d(x_work, params)) ** 2)

    grid_step = float(jnp.min(jnp.abs(jnp.diff(x_work)))) if len(x_work) > 1 else 0.0
    sigma_min = max(grid_step * 0.25, 1e-12)
    span = float(jnp.abs(x_work[-1] - x_work[0]))
    sigma_max = max(span, sigma_min * 10.0)
    x_min = float(x_work[0])
    x_max = float(x_work[-1])

    half_max = 0.5 * (y_max + y_min)
    x_np = np.asarray(x_work)
    y_np = np.asarray(y)

    def interpolate_half_max(index0: int, index1: int) -> float:
        x0, x1 = float(x_np[index0]), float(x_np[index1])
        y0, y1 = float(y_np[index0]), float(y_np[index1])
        denominator = y1 - y0
        if abs(denominator) < 1e-20:
            return 0.5 * (x0 + x1)
        return x0 + (half_max - y0) / denominator * (x1 - x0)

    left_cross = None
    for index in range(max_idx - 1, -1, -1):
        if y_np[index] <= half_max:
            left_cross = interpolate_half_max(index, index + 1)
            break
    right_cross = None
    for index in range(max_idx, len(y_np) - 1):
        if y_np[index + 1] <= half_max:
            right_cross = interpolate_half_max(index, index + 1)
            break

    if left_cross is not None and right_cross is not None and right_cross > left_cross:
        sigma_init = (right_cross - left_cross) / FWHM_FACTOR
    else:
        above_half = np.where(y_np > half_max)[0]
        if above_half.size > 1:
            sigma_init = float(x_np[above_half[-1]] - x_np[above_half[0]]) / FWHM_FACTOR
        else:
            sigma_init = span / 6.0
    if not np.isfinite(sigma_init) or sigma_init <= 0.0:
        sigma_init = span / 6.0
    sigma_init = float(np.clip(sigma_init, sigma_min, sigma_max))

    def softplus_inv(value: float) -> float:
        return float(jnp.log(jnp.expm1(max(value, 1e-12))))

    def logit(value: float) -> float:
        value = min(max(value, 1e-6), 1.0 - 1e-6)
        return float(jnp.log(value) - jnp.log1p(-value))

    center_fraction = (center_init - x_min) / (x_max - x_min) if x_max != x_min else 0.5
    sigma_fraction = (sigma_init - sigma_min) / (sigma_max - sigma_min) if sigma_max != sigma_min else 0.5
    initial_raw = jnp.array([softplus_inv(amplitude_init), logit(center_fraction), logit(sigma_fraction), offset_init])

    def unpack(params_raw: jax.Array) -> jax.Array:
        amplitude_raw, center_raw, sigma_raw, offset = params_raw
        sigmoid_center = 1.0 / (1.0 + jnp.exp(-center_raw))
        sigmoid_sigma = 1.0 / (1.0 + jnp.exp(-sigma_raw))
        amplitude = jnp.log1p(jnp.exp(amplitude_raw))
        center = x_min + (x_max - x_min) * jnp.clip(sigmoid_center, 0.0, 1.0)
        sigma = sigma_min + (sigma_max - sigma_min) * jnp.clip(sigmoid_sigma, 0.0, 1.0)
        return jnp.array([amplitude, center, sigma, offset])

    base_loss = float(loss_fn(unpack(initial_raw)))
    result = minimize(
        lambda params_raw: loss_fn(unpack(params_raw)),
        initial_raw,
        method="BFGS",
        options={"maxiter": 500, "gtol": 1e-9},
    )
    amplitude, center_scaled, sigma_scaled, offset = unpack(result.x)
    center_m = center_scaled * scale
    sigma_m = sigma_scaled * scale
    fwhm_m = FWHM_FACTOR * sigma_m
    improved = float(result.fun) < base_loss
    return GaussianFitResult(
        amplitude=float(amplitude),
        center_m=float(center_m),
        sigma_m=float(sigma_m),
        offset=float(offset),
        fwhm_m=float(fwhm_m),
        success=bool(result.success) or improved,
        residual=float(result.fun),
    )


def fit_profiles(profiles: tuple[ProfileData, ...]) -> dict[str, GaussianFitResult]:
    return {profile.direction: fit_gaussian_1d(profile) for profile in profiles}


def fit_result_to_array(result: GaussianFitResult) -> np.ndarray:
    """Serialize a fit into cache.npz without pickled objects."""
    return np.asarray(
        [
            result.amplitude,
            result.center_m,
            result.sigma_m,
            result.offset,
            result.fwhm_m,
            float(result.success),
            result.residual,
        ],
        dtype=np.float64,
    )


def fit_result_from_array(values: np.ndarray) -> GaussianFitResult:
    return GaussianFitResult(
        amplitude=float(values[0]),
        center_m=float(values[1]),
        sigma_m=float(values[2]),
        offset=float(values[3]),
        fwhm_m=float(values[4]),
        success=bool(values[5]),
        residual=float(values[6]),
    )
