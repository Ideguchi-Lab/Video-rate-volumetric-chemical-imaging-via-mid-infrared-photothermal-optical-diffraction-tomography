# Pre-analysis manifests

These small files are frozen outputs of acquisition-specific pre-analysis and
are tracked with the code. Reconstruction starts here; it does not repeat MIR
state identification or failed-SLM-update detection.

`live_cell_20251224_pairings.csv` contains 731 valid MIR-ON/OFF pairs grouped
into 72 volumes. It is a line-ending-normalized copy of the pairing used for
the 2025-12-24 COS-7 trajectory analysis. Its legacy columns have these fixed
semantics:

- `ImagePathA`, `ImageIndexA`: MIR-ON hologram
- `ImagePathB`, `ImageIndexB`: MIR-OFF hologram
- `MipIndex`: zero-based reconstructed volume ID
