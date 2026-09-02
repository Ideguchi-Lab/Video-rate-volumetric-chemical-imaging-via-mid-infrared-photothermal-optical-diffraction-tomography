"""Quick-look plots for reconstructed MIP-ODT volumes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mipodt_paper.mipodt import MIPODTRunConfig
from mipodt_paper.plotting import apply_paper_style


def plot_volume_slices(
    volume_path: Path,
    config: MIPODTRunConfig,
    z_indices: tuple[int, ...],
) -> plt.Figure:
    """Plot selected XY planes in physical coordinates without loading a sequence."""
    volume = np.load(volume_path, mmap_mode="r")
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}: {volume_path}")
    invalid = [index for index in z_indices if not 0 <= index < volume.shape[2]]
    if invalid:
        raise ValueError(f"z indices out of range for shape {volume.shape}: {invalid}")

    params = config.make_odt_parameters()
    lateral_um = float(params.imgpx_lateral_m_per_px) * 1e6
    axial_um = float(params.imgpx_axial_m_per_px) * 1e6
    extent = (0.0, volume.shape[0] * lateral_um, 0.0, volume.shape[1] * lateral_um)

    apply_paper_style()
    fig, axes = plt.subplots(1, len(z_indices), figsize=(3.4 * len(z_indices), 3.0), squeeze=False)
    image = None
    for axis, z_index in zip(axes[0], z_indices, strict=True):
        image = axis.imshow(
            np.asarray(volume[:, :, z_index]).T,
            origin="lower",
            extent=extent,
            cmap=config.preview.colormap,
            vmin=config.preview.vmin,
            vmax=config.preview.vmax,
        )
        z_um = (z_index - volume.shape[2] // 2) * axial_um
        axis.set_title(f"z = {z_um:.2f} µm (index {z_index})")
        axis.set_xlabel("x (µm)")
        axis.set_ylabel("y (µm)")
        axis.set_aspect("equal")
    if image is not None:
        colorbar = fig.colorbar(image, ax=list(axes[0]), shrink=0.82)
        colorbar.set_label("$\\Delta n_\\mathrm{MIP} = n_\\mathrm{OFF} - n_\\mathrm{ON}$")
    return fig
