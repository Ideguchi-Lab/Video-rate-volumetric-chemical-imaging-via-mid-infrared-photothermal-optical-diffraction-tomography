# Figure 4 — video-rate particle tracking and anomalous diffusion

This directory reproduces the individual panels used to assemble Figure 4.
Figure 4 uses reconstructed MIP-ODT volumes for slice images and the supplied
particle-probability map for tracking and diffusion analysis.

## Setup

Clone the public dependencies at their recorded commits and create the locked environment:

```bash
git submodule update --init --recursive
uv sync
```

The two submodule commits are also recorded in `config.yaml`. ParticleTracker is pinned to its public v1.0.0 release.
Particle linking and total 3D MSD analysis use
`submodules/ParticleTracker` directly. `submodules/fastfilters` is fixed to the version used by the reconstruction
environment.

## Data contract

Place the Zenodo files under `MIPODT_DATA_DIR` (default: `data/`) as follows:

```text
fig4_anomalous_diffusion/
├── sequential_mipodt/
│   └── holograms/                  # 1,420 TIFF holograms
├── mipodt/
│   ├── mipodt_r_000.npy
│   ├── ...
│   ├── mipodt_r_069.npy
│   └── run_manifest.yaml        # reconstruction settings and software versions
├── metadata/
│   └── anomalous_diffusion_20251224_pairings.csv
└── tracking/
    ├── region_1.h5
    ├── region_1_lds.h5
    └── manifest_subset.yaml
```

`mipodt_r_*.npy` is a real `float32` array with shape `(535, 535, 267)` and axis order `(x, y, z)`. The normal run
only reads volume 9 for the displayed slices. The complete 0–69 sequence is needed only with
`--recompute-probabilities`.

The 1,420 TIFF files are the holograms referenced by volume IDs 0–69 in
`manifests/anomalous_diffusion_20251224_pairings.csv`. They are needed only to repeat MIP-ODT reconstruction.

`region_1.h5` contains the `exported_data` probability dataset as `float32`, shape
`(70, 20, 340, 340, 2)`, and axis order `(t, z, y, x, class)`. Channel 1 is the target-particle probability.
`region_1_lds.h5` contains the public manual labels in dataset `labels`; `manifest_subset.yaml` maps its subset frames
and z planes onto the reconstructed crop. Labels and the manifest are needed only for the explicit recomputation path.

## Reproduce

Validate the inputs and generate all images:

```bash
uv run figures/fig4_anomalous_diffusion/run.py --validate-only
uv run figures/fig4_anomalous_diffusion/run.py
```

Re-plot from the portable CSV/NPZ cache without rerunning tracking:

```bash
uv run figures/fig4_anomalous_diffusion/run.py --plot-only
```

Regenerate the probability map from reconstructed volumes and labels before running the same downstream pipeline:

```bash
uv run figures/fig4_anomalous_diffusion/run.py --recompute-probabilities
```

Probability regeneration writes the normalized volume crop and probability map under `output/`, using the
random-forest feature set and parameters in `config.yaml`. This step takes longer than analysis of the supplied map.

## Reconstruct the MIP-ODT sequence

The reconstruction remains separate from the distributed NPY files. To write regenerated volumes to a separate
directory, run:

```bash
uv run scripts/reconstruct_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml
```

Set `MIPODT_RECONSTRUCTION_DIR="$PWD/data"` only when intentionally replacing the distributed arrays.

## Outputs

Every image is written independently as PNG and EPS:

- `mipodt_z_slices`: volume 9 at z indices 125 and 131
- `particle_357_trajectory_3d`: selected trajectory, colored by time
- `particle_357_msd_loglog`: total 3D MSD and anomalous-diffusion fit
- `particle_411_trajectory_centroid_shift`: two centroids split at frame 53
- `particle_411_msd_xyz3d_loglog`: X, Y, Z, and total 3D MSD
- `diffusion_scatter`: generalized diffusion coefficient versus case-bootstrap median exponent
- `alpha_spatial`: mean track positions colored by case-bootstrap median exponent

The output directory also contains the selected-particle MSD tables, `analysis_summary.yaml`, and portable analysis
caches. Generated files are ignored by git.

The expected numerical checkpoints are 50 accepted particles, particle 357 `alpha ≈ 0.322` and
`D ≈ 1.04e-2 µm²/s^alpha`, and particle 411 component exponents X/Y/Z/3D of approximately
0.682/0.337/0.137/0.446.
