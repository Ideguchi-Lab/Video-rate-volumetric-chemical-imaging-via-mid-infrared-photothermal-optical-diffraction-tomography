# Video-rate volumetric chemical imaging via mid-infrared photothermal optical diffraction tomography

Scripts for reproducing Figures 2–4 and Supplementary Figure 2 of the paper.
Each figure has one entry point, with shared reconstruction and analysis
functions in `src/mipodt_paper/`.

## Layout

```text
.
├── figures/                     # each figure contains run.py, config.yaml, README.md and output/
│   ├── fig2_resolution/
│   ├── fig3_multispectral/
│   ├── fig4_anomalous_diffusion/
│   └── suppfig2_noise_std/
├── src/
│   └── mipodt_paper/            # shared reconstruction and analysis code
├── configs/
│   └── reconstruction/         # acquisition-specific reconstruction parameters
├── manifests/                  # frozen pairing and calibration CSVs
├── scripts/
│   ├── reconstruct_mipodt.py   # reconstruct sequential MIP-ODT volumes from TIFFs
│   ├── reconstruct_static_odt.py # reconstruct static ODT volumes from TIFFs
│   └── preview_mipodt.py       # preview reconstructed slices
├── submodules/                 # commit-pinned public dependencies
│   ├── ParticleTracker/
│   └── fastfilters/
├── data/                       # seed TIFFs, reconstructed arrays and reference spectra (ignored)
└── reconstructions/            # reconstruction outputs (ignored)
```

## Setup

```bash
git submodule update --init --recursive
uv sync --locked             # creates .venv with pinned deps (uv.lock)
```

Reconstruction uses `muscopy[cuda]==1.0.2` from PyPI. ParticleTracker (`pt3d`)
is installed from the public HTTPS submodule at its v1.0.0 release commit
`2ef7a912a1b42ee1ff86b0adc6cec6fb53d7b9bc` for reproducible Figure 4 tracking.

Place the data under `data/`, or set
`MIPODT_DATA_DIR=/path/to/unpacked-data`.

## Generate the main figure panels

Run each figure from the repository root:

```bash
uv run --locked figures/fig2_resolution/run.py
uv run --locked figures/fig3_multispectral/run.py
uv run --locked figures/fig4_anomalous_diffusion/run.py
uv run --locked figures/suppfig2_noise_std/run.py
```

PNG/EPS panels and numerical results are written to each `figures/<figure>/output/`.
These are the source panels for assembling Figures 2–4 and Supplementary Figure 2.

| Figure | Contents | Detailed instructions |
|---|---|---|
| 2 | ODT/MIP-ODT slices, profiles, cross sections and resolution fits | [Fig. 2](figures/fig2_resolution/README.md) |
| 3 | ODT ROIs, MIP-ODT slices, spectra and NNLS | [Fig. 3](figures/fig3_multispectral/README.md) |
| 4 | Particle trajectories, MSD fits and diffusion distributions | [Fig. 4](figures/fig4_anomalous_diffusion/README.md) |
| S2 | MIP-ODT noise STD slices without MIR irradiation and ROI noise floor | [Supp. Fig. 2](figures/suppfig2_noise_std/README.md) |

The figure scripts use the distributed NPY arrays and, for Figure 4, the supplied
probability map. Reconstruction from TIFF measurements is a separate step using
`scripts/reconstruct_mipodt.py` and `scripts/reconstruct_static_odt.py`.
The [Fig. 3 instructions](figures/fig3_multispectral/README.md) describe the
reconstruction settings and BSA/NIST reference spectra required for its analysis.

## Reconstruct the sequential MIP-ODT datasets

Sequential reconstruction uses the supplied pairing manifests to select
same-angle MIR-ON/OFF holograms. These manifests record the acquisition
pre-analysis, including exclusion of failed illumination updates.

Dataset names reflect their downstream role:

- `fig2_resolution_20251202`: one volume used for the Fig. 2 resolution validation
- `anomalous_diffusion_20251224`: video-rate trajectory and anomalous-diffusion analysis
- `hyperspectral_20251214`: wavelength-swept hyperspectral analysis
- `noise_std_20251212`: MIR-off sequence used for the Supplementary Fig. 2 noise floor

Arrange the downloaded data as described in `data/README.md`, then validate all
inputs without starting a GPU reconstruction:

```bash
uv run scripts/reconstruct_mipodt.py \
  configs/reconstruction/anomalous_diffusion_20251224.yaml \
  --validate-only

uv run scripts/reconstruct_mipodt.py \
  configs/reconstruction/hyperspectral_20251214.yaml \
  --validate-only
```

Reconstruct one volume first. Output is a real `float32` array in NumPy
`(axis0, axis1, z)` order. The sign convention is acquisition-specific and
recorded in each YAML and run manifest; Figure 3 uses `ri_change_sign: +1`.

```bash
uv run scripts/reconstruct_mipodt.py \
  configs/reconstruction/anomalous_diffusion_20251224.yaml \
  --volume 0
```

Omitting `--volume` processes the complete anomalous-diffusion sequence. For
the hyperspectral configuration it processes only the volume IDs explicitly
mapped to calibrated values in `manifests/hyperspectral_20251214_volumes.csv`.
Existing volumes are retained, so the same command resumes an interrupted run.
Pass `--overwrite` only when an intentional recomputation is required.

```bash
uv run scripts/reconstruct_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml
uv run scripts/reconstruct_mipodt.py configs/reconstruction/hyperspectral_20251214.yaml
```

Generate the default quick-look view (volume 9, z indices 125 and 131) after
that volume has been reconstructed:

```bash
uv run scripts/preview_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml
```

Large outputs default to dataset-named directories under `reconstructions/`.
Set `MIPODT_RECONSTRUCTION_DIR` to place them on another disk. Every run writes
a `run_manifest.yaml` containing the effective optical parameters, sampling,
software versions, pair counts, selected volume IDs, and correction settings.

## Development

```bash
uv sync --locked               # dev group (ruff, pre-commit) is included by default
uv run pre-commit install      # once; hooks run on every commit
uv run pre-commit run --all-files
```

## Citation and distribution

The data and source archives have not yet been published on Zenodo. The paper
citation, DOI and archive checksums will be recorded when assigned. Source
archives must include both pinned submodules; a top-level Git archive omits them.
See the [data layout](data/README.md).

## License

This project is licensed under the [MIT License](LICENSE).
