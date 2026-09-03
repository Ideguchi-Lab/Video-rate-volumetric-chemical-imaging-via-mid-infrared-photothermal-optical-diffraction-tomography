# Video-rate volumetric chemical imaging via mid-infrared photothermal optical diffraction tomography

Code to reproduce every figure of the paper. One directory per figure, one
entry point per directory, shared helpers in a small package.

## Layout

```
.
├── figures/                 # one sub-directory per (supplementary) figure
│   ├── _template/           # copied by scripts/new_figure.py
│   ├── fig1_system/
│   │   ├── README.md        # panels, required data, runtime
│   │   ├── config.yaml      # all parameters (no magic numbers in run.py)
│   │   ├── run.py           # compute() -> cache.npz -> plot() -> output/fig1.{pdf,png}
│   │   └── output/          # generated, git-ignored
│   ├── fig2_resolution/
│   ├── fig3_multispectral/
│   ├── fig4_anomalous_diffusion/
│   └── figS1_noise/
├── src/mipodt_paper/        # shared code: paths, config loader, plotting style
├── submodules/               # commit-pinned public dependencies
│   ├── ParticleTracker/
│   └── fastfilters/
├── scripts/
│   ├── download_data.py     # fetch dataset (URLs + sha256)
│   ├── new_figure.py        # scaffold a new figure directory
│   ├── reconstruct_mipodt.py # reconstruct sequential MIP-ODT volumes
│   └── preview_mipodt.py    # render quick-look XY planes
└── data/                    # raw data (git-ignored), see data/README.md
```

## Setup

```bash
git submodule update --init --recursive
uv sync                      # creates .venv with pinned deps (uv.lock)
uv run scripts/download_data.py
```

Raw data can live elsewhere: `export MIPODT_DATA_DIR=/path/to/data`.

## Reproduce

```bash
uv run figures/fig1_system/run.py          # one figure
uv run figures/fig1_system/run.py --plot-only   # re-plot from cached arrays
```

Each figure README records its output contract. Figure 2 preserves the
individual files produced by the original z-slice and resolution-evaluation
tools; other figure scripts may expose `--plot-only` for cached re-plotting.

Throughout this repository, 3D reconstruction from raw measurements is kept
separate from downstream analysis of the reconstructed arrays. Figure 3
follows this repository-wide convention and writes individual source figures
rather than a new composite; see `figures/fig3_multispectral/README.md`.
Figure 4 follows the same convention: released reconstructed volumes feed the
public, commit-pinned tracking code and each source image is saved
independently; see `figures/fig4_anomalous_diffusion/README.md`.

## Reconstruct the sequential MIP-ODT datasets

Sequential reconstruction starts from completed pre-analysis. The frozen
pairing manifests identify same-angle MIR-ON/OFF holograms and remove failed
illumination updates. The pre-analysis itself is intentionally outside the
public reconstruction pipeline.

Dataset names reflect their downstream role:

- `fig2_resolution_20251202`: one volume used for the Fig. 2 resolution validation
- `anomalous_diffusion_20251224`: video-rate trajectory and anomalous-diffusion analysis
- `hyperspectral_20251214`: wavelength-swept hyperspectral analysis

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

Reconstruct one volume first. Output is a real `float32` array in `(x, y, z)`
axis order and has the physical definition `delta_n_MIP = n_OFF - n_ON`.

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

## Adding a figure

```bash
uv run scripts/new_figure.py fig4_spectra --title "Hyperspectral imaging"
```

## Development

```bash
uv sync                        # dev group (ruff, pre-commit) is included by default
uv run pre-commit install      # once; hooks run on every commit
uv run pre-commit run --all-files
```

## Citation

TODO: bibtex once published.

## License

TODO
