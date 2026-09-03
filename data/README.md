# Data

Raw measurements are **not** tracked in git. Place them here (or anywhere and
point `MIPODT_DATA_DIR` at it) with the layout below, or run
`uv run scripts/download_data.py` once the dataset is published.

```
data/
├── fig1/
├── fig2_resolution/
│   ├── mipodt/
│   │   └── mipodt_r_000.npy
│   ├── sequential_mipodt/
│   │   └── holograms/   # 22 TIFF holograms (11 MIR-ON/OFF pairs)
│   └── static_odt/
│       ├── odt_cell.npy
│       ├── sample/       # 11 TIFF holograms
│       └── reference/    # 11 TIFF holograms
├── fig3/
├── figS1/
├── anomalous_diffusion_20251224/
│   └── VLXT-17M.I_700009766081_251224-172142/
│       ├── image0000001.tif
│       └── ...
└── hyperspectral_20251214/
    └── try2-cell1/
        ├── image0000001.tif
        └── ...
```

Each `figures/<fig>/README.md` lists the exact files it needs.

`fig2_resolution/static_odt/odt_cell.npy` and
`fig2_resolution/mipodt/mipodt_r_000.npy` are distributed so that the
resolution analysis can be run without repeating reconstruction. The MIP-ODT
array in the Zenodo package is generated from the 22 sequential holograms by
`configs/reconstruction/fig2_resolution_20251202.yaml`. The sample/reference
ordering for static ODT and the MIR-ON/OFF ordering for MIP-ODT are tracked in
the corresponding CSV files under `manifests/`.

The pairing CSVs and hyperspectral volume-to-wavenumber mapping are frozen
outputs of pre-analysis and are tracked under `manifests/`. In the legacy
pairing format, `ImagePathA` means MIR-ON and `ImagePathB` means MIR-OFF.

To keep raw and reconstructed arrays outside this checkout:

```bash
export MIPODT_DATA_DIR=/path/to/unpacked-data
export MIPODT_RECONSTRUCTION_DIR=/path/to/reconstructed-volumes
```
