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
├── fig3_multispectral/
│   ├── hyperspectral_raw/       # raw hyperspectral TIFF holograms
│   ├── mipodt/                  # mipodt_r_000.npy ... mipodt_r_041.npy
│   ├── static_odt/
│   │   ├── odt_cell.npy
│   │   └── raw/{sample,reference}/  # added with the static raw release
│   └── references/              # reference_spectra.csv + metadata YAML
├── fig4_anomalous_diffusion/
│   ├── mipodt/                  # real float32 (x,y,z), IDs 000-069
│   └── tracking/
│       ├── region_1.h5          # frozen probability map (t,z,y,x,class)
│       ├── region_1_lds.h5      # public manual labels
│       └── manifest_subset.yaml # label subset -> volume crop mapping
├── figS1/
├── anomalous_diffusion_20251224/
│   └── VLXT-17M.I_700009766081_251224-172142/
│       ├── image0000001.tif
│       └── ...
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

The Figure 3 archive includes only the 42 hyperspectral volumes with calibrated
wavenumber and power records (IDs 0–41). See
`figures/fig3_multispectral/README.md` for the exact reference-spectrum schema,
static-ODT contract, and generated outputs.

The Figure 4 archive includes reconstructed volumes so users can proceed
directly to tracking. Its frozen probability map is the default reproducible
input; the labels and subset manifest enable the slower probability-regeneration
audit. See `figures/fig4_anomalous_diffusion/README.md` for shapes, axes, and
generated outputs.

To keep raw and reconstructed arrays outside this checkout:

```bash
export MIPODT_DATA_DIR=/path/to/unpacked-data
export MIPODT_RECONSTRUCTION_DIR=/path/to/reconstructed-volumes
```
