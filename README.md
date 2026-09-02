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
│   ├── fig3_live_cell/
│   └── figS1_noise/
├── src/mipodt_paper/        # shared code: paths, config loader, plotting style
├── scripts/
│   ├── download_data.py     # fetch dataset (URLs + sha256)
│   ├── new_figure.py        # scaffold a new figure directory
│   └── run_all.py           # reproduce everything
└── data/                    # raw data (git-ignored), see data/README.md
```

## Setup

```bash
uv sync                      # creates .venv with pinned deps (uv.lock)
uv run scripts/download_data.py
```

Raw data can live elsewhere: `export MIPODT_DATA_DIR=/path/to/data`.

## Reproduce

```bash
uv run figures/fig1_system/run.py          # one figure
uv run figures/fig1_system/run.py --plot-only   # re-plot from cached arrays
uv run scripts/run_all.py                  # all figures
```

Each `run.py` splits the work into `compute()` (slow, cached to
`output/cache.npz`) and `plot()` (fast), so styling changes never trigger a
re-reconstruction.

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
