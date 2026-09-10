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
- `noise_std_20251212`: sequence acquired with the MIR laser off, used for the
  Supplementary Fig. 2 noise-floor estimate.

`anomalous_diffusion_20251224_pairings.csv` contains the valid MIR-ON/OFF
pairs used for the 2025-12-24 trajectory analysis.

`hyperspectral_20251214_pairings.csv` contains the valid pairings for the 42
volumes used in the paper analysis. `hyperspectral_20251214_volumes.csv` maps
each `volume_id` to its calibrated wavenumber, MIR power, and calibration
timestamp.

`fig2_resolution_static_pairs.csv` freezes the sample/reference ordering for
the 11-angle static ODT acquisition.

`fig3_static_odt_pairs.csv` defines the sample/reference TIFF ordering for the
Figure 3 static ODT reconstruction.

`fig2_resolution_20251202_pairings.csv` freezes the 11 same-angle MIR-ON/OFF
pairs used to regenerate the distributed `mipodt_r_000.npy`.

`noise_std_20251212_pairings.csv` contains the same-angle hologram pairs of
the 50 MIR-off volumes (IDs 0–49) averaged into the Supplementary Fig. 2 noise
STD. The MIR laser was off for the whole run, so both columns are MIR-OFF
holograms; column A takes the "MIR-ON" role in the reconstruction.
`noise_std_20251212_volumes.csv` lists those 50 volume IDs as the default
reconstruction selection.

The sequential MIP-ODT pairing CSVs use these columns and fixed semantics:

- `ImagePathA`, `ImageIndexA`: MIR-ON hologram
- `ImagePathB`, `ImageIndexB`: MIR-OFF hologram
- `MipIndex`: zero-based reconstructed volume ID

The hyperspectral volume-manifest columns and units are:

- `volume_id`: zero-based reconstructed volume ID
- `wavenumber_cm_inv`: calibrated MIR wavenumber in cm⁻¹
- `power_mw`: calibrated MIR power in mW
- `time_ms`: calibration timestamp in ms
