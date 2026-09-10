# Data

Raw measurements are **not** tracked in git. Place the unpacked Zenodo data
here, or point `MIPODT_DATA_DIR` at it, with the layout below.

```
data/
├── SHA256SUMS                  # checksums of all distributed files
├── fig2_resolution/
│   ├── mipodt/
│   │   └── mipodt_r_000.npy
│   ├── sequential_mipodt/
│   │   └── holograms/   # 22 TIFF holograms (11 MIR-ON/OFF pairs)
│   ├── metadata/                # static and MIR-ON/OFF pairing CSVs
│   └── static_odt/
│       ├── odt_cell.npy
│       ├── sample/       # 11 TIFF holograms
│       └── reference/    # 11 TIFF holograms
├── fig3_multispectral/
│   ├── hyperspectral_raw/       # raw hyperspectral TIFF holograms
│   ├── mipodt/                  # mipodt_r_000.npy ... mipodt_r_041.npy
│   ├── metadata/                # pairing and MIR calibration CSVs
│   ├── static_odt/
│   │   ├── odt_cell.npy
│   │   └── raw/{sample,reference}/
│   └── references/              # BSA FTIR + external NIST source definitions
└── fig4_anomalous_diffusion/
│   ├── sequential_mipodt/
│   │   └── holograms/           # TIFFs used by volume IDs 000-069
│   ├── mipodt/                  # real float32 (x,y,z), IDs 000-069
│   ├── metadata/                # MIR-ON/OFF pairing CSV
│   └── tracking/
│       ├── region_1.h5          # frozen probability map (t,z,y,x,class)
│       ├── region_1_lds.h5      # public manual labels
│       └── manifest_subset.yaml # label subset -> volume crop mapping
```

Each `figures/<fig>/README.md` lists the exact files it needs.

`fig2_resolution/static_odt/odt_cell.npy` and
`fig2_resolution/mipodt/mipodt_r_000.npy` are distributed so that the
resolution analysis can be run without repeating reconstruction. The MIP-ODT
array in the Zenodo package is generated from the 22 sequential holograms by
`configs/reconstruction/fig2_resolution_20251202.yaml`. The sample/reference
ordering for static ODT and the MIR-ON/OFF ordering for MIP-ODT are tracked in
the corresponding CSV files under `manifests/`.
Copies are included in `fig2_resolution/metadata/`. Static ODT reconstruction
uses `configs/reconstruction/fig2_static_odt.yaml`.

The pairing CSVs and hyperspectral volume-to-wavenumber mapping are frozen
outputs of pre-analysis and are tracked under `manifests/`. In the legacy
pairing format, `ImagePathA` means MIR-ON and `ImagePathB` means MIR-OFF.

The Figure 3 archive includes only the 42 hyperspectral volumes with calibrated
wavenumber and power records (IDs 0–41). See
`figures/fig3_multispectral/README.md` for the exact reference-spectrum schema,
static-ODT contract, and generated outputs.

Figure 3 MIP-ODT slices and ROI spectra use the same volumes reconstructed
from `fig3_multispectral/hyperspectral_raw/`. Static ODT is reconstructed from
`fig3_multispectral/static_odt/raw/`.

The Figure 4 archive includes reconstructed volumes so users can proceed
directly to analysis. Its MIP-ODT arrays are reconstructed with muscopy 1.0.2;
the configuration and software versions are recorded in `mipodt/run_manifest.yaml`.
Tracking uses the frozen probability map from the paper analysis as a separate
input. The labels and subset manifest support optional probability regeneration
from the reconstructed volumes. See `figures/fig4_anomalous_diffusion/README.md`
for shapes, axes, and generated outputs.

To verify the unpacked files, run `sha256sum -c SHA256SUMS` from this directory.

To keep raw and reconstructed arrays outside this checkout:

```bash
export MIPODT_DATA_DIR=/path/to/unpacked-data
export MIPODT_RECONSTRUCTION_DIR=/path/to/reconstructed-volumes
```
