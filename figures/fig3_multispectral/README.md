# Fig. 3 — Multispectral live-cell analysis

This directory reproduces the individual analysis outputs used for Figure 3.
It does not assemble a composite Figure 3. Reconstruction from TIFF holograms
is kept separate from analysis of the distributed NPY volumes.

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
    ├── reference_spectra.csv
    └── reference_metadata.yaml
```

`reference_spectra.csv` is wide-form and must contain
`wavenumber_cm_inv,bsa,guanosine,oleic_acid,cholesterol`. The YAML metadata
contains a `references` mapping with `native_fwhm_cm_inv` for each reference.
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

The fixed analysis uses volume IDs 14–35 for ROI spectra, divides every ROI
mean and MIP-ODT slice by its volume's laser power, and stores both the raw
and the once-corrected (`-12.5 cm⁻¹`) wavenumber. NNLS uses the settings in
`config.yaml`: 2800–3100 cm⁻¹ analysis range, AsLS `(lambda=1e5, p=0.01,
20 iterations)`, 2830–3027 cm⁻¹ area normalization and fitting, 16 cm⁻¹
common FWHM, and a 0.5 cm⁻¹ grid over the measured/reference overlap inside
the analysis range (no spectral extrapolation). Reference coefficients are nonnegative;
the fitted linear residual baseline is signed. Reported ratios are normalized
reference coefficients, not concentrations.

## Outputs

Each plotted item is saved independently as PNG and EPS:

- `odt_roi_slices` (also BMP)
- `mipodt_slices` (also BMP)
- `roi_spectra`
- `lipid_droplet_comparison` and `nucleolus_comparison`
- `nnls_combined`
- `nnls_lipid_droplet` and `nnls_nucleolus`

Numerical and provenance outputs are:

- `roi_spectra.csv`
- `nnls_coefficients.csv`
- `nnls_coefficient_ratios.csv`
- `nnls_residuals.csv`
- `analysis_cache.npz`
- `config_snapshot.yaml`
- `run_manifest.yaml`

## Reconstruct the hyperspectral volumes

The existing common reconstruction entry point and the same `muscopy`
version contract used by Figure 2 are used unchanged. The tracked volume
manifest limits the default reconstruction to calibrated IDs 0–41:

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
