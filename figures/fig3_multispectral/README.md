# Fig. 3 — Multispectral live-cell analysis

This directory reproduces the panels and numerical results used for Figure 3.
Reconstruction from TIFF holograms and analysis of the distributed NPY volumes
have separate entry points.

## Array and ROI convention

Every reconstruction is a real `float32` array with shape `(535, 535, 267)`
and is indexed as `volume[axis0, axis1, z]`. In displayed XY images, `axis1`
is horizontal and `axis0` is vertical. Every ROI is a NumPy-style half-open
interval:

- lipid droplet: `volume[325:328, 145:148, 135:137]`
- ER/mixed region: `volume[261:276, 230:237, 132:136]`
- nucleolus: `volume[300:321, 310:331, 136:141]`

## Expected Zenodo layout

Set `MIPODT_DATA_DIR` to the root of the unpacked archive. The archive is
expected to have this layout:

```text
fig3_multispectral/
├── hyperspectral_raw/                 # raw hyperspectral TIFF holograms
├── mipodt/
│   ├── mipodt_r_000.npy
│   ├── ...
│   └── mipodt_r_041.npy               # 42 calibrated float32 volumes only
├── metadata/
│   ├── hyperspectral_20251214_pairings.csv
│   ├── hyperspectral_20251214_volumes.csv
│   └── fig3_static_odt_pairs.csv
├── static_odt/
│   ├── odt_cell.npy                   # delta_n = n_sample - n_medium
│   └── raw/
│       ├── sample/
│       └── reference/
└── references/
    ├── bsa_ftir.csv
    └── reference_sources.yaml
```

`bsa_ftir.csv` is the locally measured BSA spectrum. The YAML file records its
checksum and the official NIST Chemistry WebBook URLs, identifiers, spectral
resolutions, and checksums for guanosine, oleic acid, and cholesterol. The NIST
spectra are not redistributed in the Zenodo archive; `run.py` downloads the
pinned JCAMP-DX files into `output/.nist_reference_cache/` when needed.
The DOI and archive checksums will be added at publication time.

## Validate and run the analysis

```bash
export MIPODT_DATA_DIR=/path/to/unpacked-data

uv run figures/fig3_multispectral/run.py --validate-only
uv run figures/fig3_multispectral/run.py
uv run figures/fig3_multispectral/run.py --plot-only
```

Validation requires volume IDs 0–41 exactly once, the expected shape and
real `float32` dtype, in-range half-open ROIs, the static ODT NPY, and all
reference spectra and metadata. Normal execution computes and caches the
analysis. `--plot-only` reads only `output/analysis_cache.npz` and regenerates
the figures and CSVs without loading the distributed volumes.

The ROI analysis uses volume IDs 14–35, divides each ROI mean by the volume's
laser power, and records nominal and corrected (`−12.5 cm⁻¹`) wavenumbers.
The lipid/nucleolus plot uses an ascending wavenumber axis, blue/green curves,
and red measurement-time points relative to the first selected frame.

NNLS preprocessing crops the **nominal** sample axis to 2800–3100 cm⁻¹,
applies AsLS `(lambda=1e5, p=0.01, 20 iterations)` on native points, and
normalizes by the positive area over 2830–3027 cm⁻¹ while retaining negative
signal values. The sample axis is then shifted by −12.5 cm⁻¹.
References are baseline-corrected and normalized on their own native axes,
then resampled at 0.5 cm⁻¹ and broadened to 16 cm⁻¹ FWHM. Fit their interpolated,
renormalized basis on the **14 measured sample points** in 2830–3027 cm⁻¹.
Reference coefficients are nonnegative; the residual linear baseline is signed.
Component curves exclude that baseline. Ratios are coefficient fractions, not
concentrations.

MIP-ODT slices are extracted from the same seed-reconstructed volumes used
for ROI spectra and divided by the recorded laser power. For each row, the
display range is zero to the maximum of its two visible crops; the off-resonance
row has a minimum
upper limit of `6e-5`. Labels follow nominal-channel rounding, then shift,
then display rounding: 2920 / 2950 / 3094 cm⁻¹; relative z is 0 / 1.1 µm.

## Outputs

Each plotted item is saved independently as PNG and EPS:

- `odt_roi_slices` (also BMP; vertical z = 0.0/1.1 µm panels, RI 1.327–1.360,
  with blue/orange/green arrows marking the lipid droplet, ER/mixed region and nucleolus ROIs)
- `mipodt_slices` (also BMP)
- `spectrum_analysis_lipid_nucleolus` (lipid/nucleolus spectra with measurement-time overlay)
- `nnls_combined` (two rows, three columns: fit/residual, components, fractions)
- `nnls_lipid_droplet` and `nnls_nucleolus` (the corresponding individual rows)

Numerical and provenance outputs are:

- `roi_spectra.csv`
- `nnls_coefficients.csv`
- `nnls_coefficient_ratios.csv`
- `nnls_residuals.csv`
- `analysis_cache.npz`
- `config_snapshot.yaml`
- `run_manifest.yaml`

## Reconstruction settings

The commands below reconstruct the 42 MIP-ODT volumes and static ODT from
944 TIFFs (461 ON/OFF pairs and 11 static sample/reference pairs).
The complete spectral analysis also requires the local BSA spectrum and three
pinned NIST spectra. `run.py` reuses its validated `output/.nist_reference_cache/`
or downloads these spectra on first use.

The settings for **MIP-ODT** are:

- `use_skimage_unwrap: false` (Poisson, preserving the mean phase with `keep_mean=True`)
- `spectrum_placement: legacy`
- `offset_regions: null`, `gradient_correction: false`
- `output.ri_change_sign: 1.0`

The original Figure 3 analysis used the **legacy spectrum placement**,
reproduced by [`legacy_odt.py`](../../src/mipodt_paper/legacy_odt.py) with
muscopy 1.0.2. For the Figure 3 parameters, the 2D spectra start one pixel
lower in array index along both axes than in muscopy 1.0.2's standard
placement before the inverse FFT. Scattering calculations and volume synthesis
use muscopy 1.0.2.

Static ODT uses a separate configuration with `use_skimage_unwrap: true`.
The reconstruction commands record their parameters and software versions in
`run_manifest.yaml` beside the generated volumes.

## Reconstruct the hyperspectral volumes

The common reconstruction entry point uses pinned muscopy 1.0.2, with the
Figure 3 legacy placement selected by its YAML configuration. The tracked
volume manifest limits the default reconstruction to calibrated IDs 0–41:

```bash
uv run scripts/reconstruct_mipodt.py \
  configs/reconstruction/hyperspectral_20251214.yaml --validate-only

uv run scripts/reconstruct_mipodt.py \
  configs/reconstruction/hyperspectral_20251214.yaml
```

Outputs default to `reconstructions/fig3_multispectral/mipodt`. To prepare the
archive layout consumed by the figure analysis, set
`MIPODT_RECONSTRUCTION_DIR="$MIPODT_DATA_DIR"` before running the same command.

## Reconstruct the static ODT volume

The archive contains 11 sample and 11 reference TIFF holograms. Their
illumination-angle ordering is fixed by `manifests/fig3_static_odt_pairs.csv`:

```bash
uv run scripts/reconstruct_static_odt.py \
  configs/reconstruction/fig3_static_odt.yaml --validate-only

uv run scripts/reconstruct_static_odt.py \
  configs/reconstruction/fig3_static_odt.yaml
```

The output is a real `float32` volume whose physical quantity is
`delta_n = n_sample - n_medium`. The initial optical conditions match the
Figure 3 hyperspectral reconstruction. Set
`MIPODT_RECONSTRUCTION_DIR="$MIPODT_DATA_DIR"` to write it at the archive path
consumed by the analysis.
