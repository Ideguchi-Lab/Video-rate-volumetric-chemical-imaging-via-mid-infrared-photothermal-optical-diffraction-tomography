# Fig. 3 — Video-rate volumetric chemical imaging of live cells

**Panels:** a) ..., b) ..., c) ...

## Required data

| File | Description | Size |
|---|---|---|
| `manifests/anomalous_diffusion_20251224_pairings.csv` (tracked in this repository) | Frozen MIR-ON/OFF pairing | small |
| `anomalous_diffusion_20251224/VLXT-17M.I_700009766081_251224-172142/image*.tif` | Sequential holograms | large |

## Reconstruct MIP-ODT volumes

```bash
uv run scripts/reconstruct_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml --validate-only
uv run scripts/reconstruct_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml --volume 0
uv run scripts/reconstruct_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml
uv run scripts/preview_mipodt.py configs/reconstruction/anomalous_diffusion_20251224.yaml
```

The reconstruction output is `delta_n_MIP = n_OFF - n_ON`. Scikit-image phase
unwrapping is enabled; gradient and phase-offset corrections are disabled.

The resulting `mipodt_r_*.npy` sequence is the input to the trajectory-analysis
and figure-panel pipeline. That downstream analysis is kept separate so plots
can be regenerated without repeating the GPU reconstruction.
