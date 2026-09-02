# Fig. 3 — Video-rate volumetric chemical imaging of live cells

**Panels:** a) ..., b) ..., c) ...

## Required data

| File (relative to `$MIPODT_DATA_DIR`) | Description | Size |
|---|---|---|
| `figX/raw_holograms.tif` | ... | ... |

## Reproduce

```bash
uv run figures/fig3_live_cell/run.py            # full pipeline
uv run figures/fig3_live_cell/run.py --plot-only  # re-plot from cached output/*.npz
```

Outputs are written to `output/` (`figX.pdf`, `figX.png`, intermediate `*.npz`).

## Notes

- Parameters live in `config.yaml`; nothing is hard-coded in `run.py`.
- Expected runtime: ~N min on a laptop CPU.
