"""Legacy spectral placement for the pinned muscopy 1.0.2 API.

This placement is selected by the reconstruction configuration.
For the reconstruction parameters used here, the 2D spectra start at an array index one smaller
along both axes than in muscopy 1.0.2's standard placement before the inverse FFT.
Scattering calculations and volume synthesis use muscopy 1.0.2.
"""

import jax.numpy as jnp
from muscopy.odt import (
    ODTConfig,
    ODTParameters,
    ScatteringSpectrum,
    _calc_1st_scattering_spectrum,
    calc_refractive_index,
    calc_scattering_potential_from_spectrums,
)


def odt_with_legacy_placement(on_spectra, off_spectra, params: ODTParameters, config: ODTConfig):
    """Reconstruct with the configured spectrum placement and muscopy arithmetic."""
    params.verify_parameters()
    config.precision.validate()
    spectrums = []
    for on, off in zip(on_spectra, off_spectra, strict=True):
        peak = jnp.unravel_index(jnp.argmax(jnp.abs(off)), off.shape)
        vector = (int(peak[0]) - on.shape[0] // 2, int(peak[1]) - on.shape[1] // 2)
        fields = []
        for spectrum in (on, off):
            size = 2 * params.aperturesize_px + 1 + 2 * config.edge_size
            expanded = jnp.zeros((size, size), dtype=spectrum.dtype)
            x = params.aperturesize_px // 2 - vector[0] + config.edge_size
            y = params.aperturesize_px // 2 - vector[1] + config.edge_size
            expanded = expanded.at[x : x + spectrum.shape[0], y : y + spectrum.shape[1]].set(spectrum)
            expanded = jnp.asarray(expanded, dtype=config.precision.complex_precision())
            fields.append(jnp.fft.ifft2(jnp.fft.ifftshift(expanded), norm="ortho"))
        scattering = _calc_1st_scattering_spectrum(
            fields[0],
            fields[1],
            params,
            config.approx_type,
            vector,
            config.edge_size,
            config.offset_regions,
            config.gradient_correction,
            config.use_skimage_unwrap,
        )
        spectrums.append(ScatteringSpectrum(scattering, vector))
    potential, spectrum = calc_scattering_potential_from_spectrums(spectrums, params, config)
    if config.linear_approx:
        factor = -((params.light_freq_px * params.k_per_px) ** 2) * params.n_sol
        refractive_index = params.n_sol + potential / factor
    else:
        refractive_index = calc_refractive_index(potential, params)
    return refractive_index, spectrum
