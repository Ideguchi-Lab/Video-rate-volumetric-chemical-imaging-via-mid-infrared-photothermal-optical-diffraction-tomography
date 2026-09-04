# Pre-analysis manifests

These small files are frozen outputs of acquisition-specific pre-analysis and
are tracked with the code. Reconstruction starts here; it does not repeat MIR
state identification or failed-SLM-update detection.

The dataset names describe their role in the paper:

- `fig2_resolution`: static-ODT and sequential MIP-ODT source holograms and
  their distributed reconstructed arrays used for the resolution validation.
- `anomalous_diffusion_20251224`: video-rate sequence used for trajectory and
  anomalous-diffusion analysis.
- `hyperspectral_20251214`: wavelength-swept sequence used for hyperspectral
  analysis.

`anomalous_diffusion_20251224_pairings.csv` contains the valid MIR-ON/OFF
pairs used for the 2025-12-24 trajectory analysis.

`hyperspectral_20251214_pairings.csv` contains all valid pairings recovered
from the 2025-12-14 acquisition. `hyperspectral_20251214_volumes.csv` is the
paper-analysis manifest: each row explicitly maps a reconstructed `volume_id`
to its calibrated wavenumber, MIR power, and calibration timestamp. Only the
volumes listed in this table are selected by default. The remaining pairing
groups are not assigned calibrated wavenumbers and are therefore not part of
the hyperspectral paper analysis.

`fig2_resolution_static_pairs.csv` freezes the sample/reference ordering for
the 11-angle static ODT acquisition.

`fig3_static_odt_pairs.csv` defines the sample/reference TIFF ordering for the
Figure 3 static ODT reconstruction.

`fig2_resolution_20251202_pairings.csv` freezes the 11 same-angle MIR-ON/OFF
pairs used to regenerate the distributed `mipodt_r_000.npy`.

Both pairing CSVs use these legacy columns and fixed semantics:

- `ImagePathA`, `ImageIndexA`: MIR-ON hologram
- `ImagePathB`, `ImageIndexB`: MIR-OFF hologram
- `MipIndex`: zero-based reconstructed volume ID

The hyperspectral volume-manifest columns and units are:

- `volume_id`: zero-based reconstructed volume ID
- `wavenumber_cm_inv`: calibrated MIR wavenumber in cm⁻¹
- `power_mw`: calibrated MIR power in mW
- `time_ms`: calibration timestamp in ms
