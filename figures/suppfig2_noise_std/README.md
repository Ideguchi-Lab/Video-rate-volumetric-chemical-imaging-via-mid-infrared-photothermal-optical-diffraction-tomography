# Supplementary Fig. 2 — MIP-ODT noise without MIR irradiation

The figure shows the voxel-wise standard deviation of the MIP-ODT RI change
across a sequence of volumes acquired with the MIR laser off. Each pair in the
sequential acquisition is then an OFF/OFF pair, so the reconstructed
`delta_n_MIP` contains only noise, and its standard deviation across the
sequence is the noise floor of the tomograph. The same array yields the mean
noise STD inside a central ROI.

## Required data

| File (relative to `$MIPODT_DATA_DIR`) | Description | Size |
|---|---|---|
| `suppfig2_noise_std/noise_std/std_result.npy` | Voxel-wise noise STD, `(x,y,z)` float32 | 306 MB |

`std_result.npy` is computed from 50 MIR-off MIP-ODT volumes
(`mipodt_r_000.npy` … `mipodt_r_049.npy`, shape `(535, 535, 267)`)
reconstructed from the 2025-12-12 COS-7 acquisition. The Zenodo package also
contains those 50 volumes under `suppfig2_noise_std/mipodt/`, the
reconstruction settings in `mipodt/run_manifest.yaml`, and the 1,100 holograms under
`suppfig2_noise_std/sequential_mipodt/holograms/`, and the pairing CSV under
`suppfig2_noise_std/metadata/` and in the code repository's `manifests/`.

## Reproduce the downstream analysis

```bash
uv run figures/suppfig2_noise_std/run.py --validate-only
uv run figures/suppfig2_noise_std/run.py
```

The script writes the following under `output/`:

- `noise_std_3d.{png,eps}`: four XY planes at z indices `[140, 134, 129, 124]`
  (relative z = 0.0 / −1.3 / −2.4 / −3.5 µm), crop `x, y = 80:470`, `viridis`,
  display range 0 to `1e-4` RI, 10 µm scalebar
- `noise_std_3d_config.yaml`: the z-slice settings
- `region_noise_mean.csv`: mean noise STD inside `x 80:470, y 80:470, z 117:147`
  (`4.27e-05` RI)
- `run_manifest.yaml`: input checksum, ROI result and output list; input paths are
  relative to `path_root` (`data`, `output`, or `repository`). External inputs
  are recorded by filename only (`path_root: external`).

## Reproduce the MIP-ODT reconstruction and the noise STD

```bash
uv run scripts/reconstruct_mipodt.py \
  configs/reconstruction/noise_std_20251212.yaml --validate-only

MIPODT_RECONSTRUCTION_DIR="$MIPODT_DATA_DIR" \
  uv run scripts/reconstruct_mipodt.py \
  configs/reconstruction/noise_std_20251212.yaml

uv run figures/suppfig2_noise_std/run.py --compute-std
```

`manifests/noise_std_20251212_pairings.csv` records the hologram pairs of
volume IDs 0–49 in the standard `MipIndex, ImageIndexA/B, ImagePathA/B`
layout; column A takes the "MIR-ON" role of the reconstruction although the
MIR laser was off. `manifests/noise_std_20251212_volumes.csv` selects those 50
volumes by default. The configuration uses linear Rytov, Poisson phase unwrap,
legacy spectrum placement, gradient and offset corrections off, and `truncate`
Ewald embedding.

`--compute-std` applies Welford's online algorithm to the real part of every
volume in `data/suppfig2_noise_std/mipodt/` with float32 accumulation, saves
`output/std_result.npy`, and produces the same panel and CSV from it. Pass
`--std-output "$MIPODT_DATA_DIR/suppfig2_noise_std/noise_std/std_result.npy"`
to write the array at the path consumed by the default analysis.

## Analysis parameters

- noise ROI `x 80:470, y 80:470, z 117:147` (NumPy half-open indices)
- z slices `[140, 134, 129, 124]`, crop `x, y = 80:470`
- lateral sampling 110 nm/pixel; axial sampling 220 nm/pixel
- 50 MIR-off volumes, unbiased standard deviation (`n − 1`)
