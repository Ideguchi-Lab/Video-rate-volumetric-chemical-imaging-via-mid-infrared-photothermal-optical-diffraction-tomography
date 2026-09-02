# Data

Raw measurements are **not** tracked in git. Place them here (or anywhere and
point `MIPODT_DATA_DIR` at it) with the layout below, or run
`uv run scripts/download_data.py` once the dataset is published.

```
data/
├── fig1/
├── fig2/
├── fig3/
├── figS1/
└── live_cell_20251224/
    └── VLXT-17M.I_700009766081_251224-172142/
        ├── image0000001.tif
        ├── image0000002.tif
        └── ...
```

Each `figures/<fig>/README.md` lists the exact files it needs.

The live-cell pairing CSV is the frozen output of pre-analysis and is tracked at
`manifests/live_cell_20251224_pairings.csv`. In that legacy CSV, `ImagePathA`
means MIR-ON and `ImagePathB` means MIR-OFF.

To keep raw and reconstructed arrays outside this checkout:

```bash
export MIPODT_DATA_DIR=/path/to/unpacked-data
export MIPODT_RECONSTRUCTION_DIR=/path/to/reconstructed-volumes
```
