# Fig. 2 — Resolution validation

The figure compares static ODT and video-rate MIP-ODT slices and reproduces
the X/Y/Z Gaussian-fit FWHM values used in the paper analysis.

## Required data

| File (relative to `$MIPODT_DATA_DIR`) | Description | Size |
|---|---|---|
| `fig2_resolution/static_odt/odt_cell.npy` | Static ODT reconstruction, `(x,y,z)` float32 | 306 MB |
| `fig2_resolution/mipodt/mipodt_r_000.npy` | Linear MIP-ODT RI change, `(x,y,z)` float32 | 306 MB |

The Zenodo package also contains the 11 sample and 11 reference holograms
under `fig2_resolution/static_odt/`, plus the 22 sequential holograms under
`fig2_resolution/sequential_mipodt/holograms/`. Thus the reconstruction and
the downstream resolution analysis can be checked independently.

## Reproduce the downstream analysis

```bash
JAX_PLATFORMS=cpu uv run figures/fig2_resolution/run.py
```

The script writes the following panels and numerical results under `output/`:

- `odt_z_slices.{png,eps}` and `mipodt_z_slices.{png,eps}`
- `odt_profiles_with_fit.{png,eps}` and `mipodt_profiles_with_fit.{png,eps}`
- `odt_cross_sections.{png,eps}` and `mipodt_cross_sections.{png,eps}`
- `odt_results.json`, `mipodt_results.json`, and the four corresponding YAML configs
- panel-only BMP files under `output/bmp/` (four z slices, three profiles,
  and two cross sections for each modality)

## Reproduce the MIP-ODT reconstruction

By default the reconstruction is kept separate from the distributed NPY and
written to `reconstructions/fig2_resolution/mipodt/mipodt_r_000.npy`:

```bash
JAX_PLATFORMS=cpu uv run scripts/reconstruct_mipodt.py \
  configs/reconstruction/fig2_resolution_20251202.yaml --volume 0
```

For preparation of the Zenodo data directory, the same command can write the
generated array at the path consumed by the downstream analysis:

```bash
MIPODT_RECONSTRUCTION_DIR="$PWD/data" JAX_PLATFORMS=cpu \
  uv run scripts/reconstruct_mipodt.py \
  configs/reconstruction/fig2_resolution_20251202.yaml --volume 0 --overwrite
```

The configuration uses muscopy 1.0.2's public API with linear Rytov,
Poisson phase unwrap, gradient correction off, offset correction off, and
`truncate` Ewald embedding. The commands above select CPU execution.
Existing arrays are retained unless `--overwrite` is supplied.

## Frozen evaluation parameters

- center `(X, Y, Z) = (284, 202, 124)`
- inclusive half ranges `(5, 5, 7)` pixels
- lateral sampling 110 nm/pixel; axial sampling 220 nm/pixel
- z slices `[140, 134, 129, 124]`
