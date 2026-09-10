"""Figure 3 NNLS panels.

Each row shows the measured spectrum, fit and residual; component contributions;
and coefficient fractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter


def resample_to_grid(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Interpolate references without extrapolating beyond their measured range."""
    order = np.argsort(x)
    return np.interp(target, x[order], y[order], left=np.nan, right=np.nan)


def safe_area(x: np.ndarray, y: np.ndarray, window: list[float], clip_negative: bool = True) -> float:
    """Integrate the positive part on the measured fit points."""
    mask = (x >= min(window)) & (x <= max(window))
    return float(np.trapezoid(np.clip(y[mask], 0, None) if clip_negative else y[mask], x[mask]))


REPRESENTATIVE_REFERENCE_LABELS = {
    "oleic_acid": "Unsaturated fatty acid",
    "cholesterol": "Sterol",
    "bsa": "Protein",
    "guanosine": "Nucleoside",
}

REFERENCE_COLORS = {
    "oleic_acid": "#4c78a8",
    "cholesterol": "#f58518",
    "bsa": "#54a24b",
    "guanosine": "#b279a2",
}

REFERENCE_COMPONENT_LABELS = {
    "oleic_acid": "Oleic acid",
    "cholesterol": "Cholesterol",
    "bsa": "BSA",
    "guanosine": "Guanosine",
}

REFERENCE_RATIO_LABELS = {
    "oleic_acid": "Unsaturated\nacyl-chain-like\n(Oleic acid)",
    "cholesterol": "Sterol-like\n(Cholesterol)",
    "bsa": "Protein-like\n(BSA)",
    "guanosine": "Nucleic-acid-like\n(Guanosine)",
}

FIT_COLOR = "#ff334f"
RESIDUAL_COLOR = "#b8b8b8"

CLASS_LABELS = {
    "lipid": "Unsaturated fatty acid",
    "sterol": "Sterol",
    "protein": "Protein",
    "polypeptide": "Protein",
}

CLASS_COLORS = {
    "lipid": "#4c78a8",
    "sterol": "#f58518",
    "protein": "#54a24b",
    "polypeptide": "#54a24b",
}


@dataclass(frozen=True)
class NNLSFigureRow:
    dense_basis: dict[str, pd.DataFrame]
    coeffs: pd.DataFrame
    curves: dict[str, pd.DataFrame]
    title: str
    roi_id: str | None = None
    observed_marker: str = "o"
    observed_marker_size: float = 3.2
    observed_color: str = "#1f77b4"
    reference_colors: dict[str, str] = field(default_factory=dict)
    ratio_reference_order: tuple[str, ...] = ()


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "eps"):
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _format_ir_axis(
    ax: plt.Axes,
    window: tuple[float, float] | list[float],
    y_label: str = "Normalized intensity",
) -> None:
    ax.set_xlim(min(window), max(window))
    ticks = [value for value in (2850, 2900, 2950, 3000) if min(window) <= value <= max(window)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(y_label)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, -2))
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(direction="in", width=0.8, length=3.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _basis_on_corrected_sample_grid(
    ref_df: pd.DataFrame,
    target_x: np.ndarray,
    window: list[float] | tuple[float, float],
    normalize_to_unit_area: bool,
) -> np.ndarray:
    values = resample_to_grid(
        ref_df["wavenumber_cm1"].to_numpy(dtype=float),
        ref_df["absorbance_normalized"].to_numpy(dtype=float),
        target_x,
    )
    if normalize_to_unit_area:
        area = safe_area(target_x, values, window, clip_negative=True)
        if np.isfinite(area) and abs(area) > 1e-15:
            values = values / area
    return values


def _reference_label(metadata: pd.DataFrame, reference_id: str) -> str:
    rows = metadata.loc[metadata["sample_id"] == reference_id, "display_name"]
    return str(rows.iloc[0]) if len(rows) else reference_id


def representative_reference_label(metadata: pd.DataFrame, reference_id: str) -> str:
    if reference_id in REPRESENTATIVE_REFERENCE_LABELS:
        return REPRESENTATIVE_REFERENCE_LABELS[reference_id]
    rows = metadata.loc[metadata["sample_id"] == reference_id]
    if len(rows):
        sample_class = str(rows.iloc[0].get("class", "")).lower()
        if sample_class in CLASS_LABELS:
            return CLASS_LABELS[sample_class]
    return _reference_label(metadata, reference_id)


def reference_color(metadata: pd.DataFrame, reference_id: str) -> str:
    if reference_id in REFERENCE_COLORS:
        return REFERENCE_COLORS[reference_id]
    rows = metadata.loc[metadata["sample_id"] == reference_id]
    if len(rows):
        sample_class = str(rows.iloc[0].get("class", "")).lower()
        if sample_class in CLASS_COLORS:
            return CLASS_COLORS[sample_class]
    return "#7f7f7f"


def _component_label(metadata: pd.DataFrame, reference_id: str) -> str:
    return REFERENCE_COMPONENT_LABELS.get(reference_id, _reference_label(metadata, reference_id))


def _ratio_label(metadata: pd.DataFrame, reference_id: str) -> str:
    return REFERENCE_RATIO_LABELS.get(reference_id, representative_reference_label(metadata, reference_id))


def _plot_nnls_row(
    axes: np.ndarray,
    row: NNLSFigureRow,
    metadata: pd.DataFrame,
    window: list[float],
    normalize_basis_to_unit_area: bool,
) -> None:
    observed_ax, component_ax, ratio_ax = axes
    if row.coeffs.empty or not row.curves:
        observed_ax.text(0.5, 0.5, "No NNLS fit", ha="center", va="center")
        for ax in axes:
            ax.set_axis_off()
        return

    selected_roi = row.roi_id if row.roi_id in row.curves else next(iter(row.curves))
    curve = row.curves[selected_roi]
    subset = row.coeffs[row.coeffs["roi_id"] == selected_roi].copy()
    if subset.empty:
        observed_ax.text(0.5, 0.5, f"No NNLS coefficients for {selected_roi}", ha="center", va="center")
        for ax in axes:
            ax.set_axis_off()
        return

    x = curve["wavenumber_cm1"].to_numpy(dtype=float)
    observed_ax.plot(
        x,
        curve["observed"],
        color=row.observed_color,
        lw=1.1,
        marker=row.observed_marker,
        ms=row.observed_marker_size,
        label="MIP",
    )
    observed_ax.plot(x, curve["fitted"], color=FIT_COLOR, lw=1.1, label="NNLS fit")
    observed_ax.plot(x, curve["residual"], color=RESIDUAL_COLOR, lw=1.0, label="Residual")
    observed_ax.text(-0.18, 1.12, row.title, transform=observed_ax.transAxes, fontsize=10, fontweight="bold")
    observed_ax.legend(
        fontsize=7.5,
        loc="upper left",
        frameon=True,
        framealpha=1.0,
        borderpad=0.35,
        handlelength=1.2,
    )
    _format_ir_axis(observed_ax, window)

    component_ax.plot(x, curve["fitted"], color=FIT_COLOR, lw=1.1, label="NNLS fit")
    for coefficient in subset.itertuples(index=False):
        reference_id = str(coefficient.reference_id)
        if reference_id not in row.dense_basis:
            continue
        basis = _basis_on_corrected_sample_grid(
            row.dense_basis[reference_id],
            x,
            window,
            normalize_basis_to_unit_area,
        )
        color = row.reference_colors.get(reference_id, reference_color(metadata, reference_id))
        component_ax.plot(
            x,
            float(coefficient.coefficient) * basis,
            lw=1.0,
            ls="--",
            color=color,
            label=_component_label(metadata, reference_id),
        )
    component_ax.legend(
        fontsize=7.5,
        loc="upper left",
        frameon=True,
        framealpha=1.0,
        borderpad=0.35,
        handlelength=1.2,
    )
    _format_ir_axis(component_ax, window)

    by_reference = subset.set_index("reference_id", drop=False)
    reference_ids = [reference_id for reference_id in row.ratio_reference_order if reference_id in by_reference.index]
    reference_ids.extend(reference_id for reference_id in subset["reference_id"] if reference_id not in reference_ids)
    fractions = [float(by_reference.loc[reference_id, "coefficient_fraction"]) for reference_id in reference_ids]
    colors = [
        row.reference_colors.get(reference_id, reference_color(metadata, reference_id))
        for reference_id in reference_ids
    ]
    y_pos = np.arange(len(reference_ids))
    ratio_ax.barh(y_pos, fractions, color=colors, height=0.68)
    ratio_ax.set_yticks(y_pos)
    ratio_ax.set_yticklabels([_ratio_label(metadata, reference_id) for reference_id in reference_ids], fontsize=8)
    ratio_ax.set_xlim(0, 1)
    ratio_ax.set_xticks(np.linspace(0, 1, 6))
    ratio_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ratio_ax.set_xlabel("NNLS coefficient fraction")
    ratio_ax.tick_params(direction="in", width=0.8, length=3.5)
    for spine in ratio_ax.spines.values():
        spine.set_linewidth(0.8)


def plot_nnls_paper_figure(
    dense_basis: dict[str, pd.DataFrame],
    coeffs: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    window: list[float],
    output_dir: Path,
    normalize_basis_to_unit_area: bool,
    stem: str,
    title: str,
    roi_id: str | None = None,
    observed_marker: str = "o",
    observed_marker_size: float = 3.2,
    observed_color: str = "#1f77b4",
    reference_colors: dict[str, str] | None = None,
    ratio_reference_order: tuple[str, ...] = (),
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.08]})
    row = NNLSFigureRow(
        dense_basis=dense_basis,
        coeffs=coeffs,
        curves=curves,
        title=title,
        roi_id=roi_id,
        observed_marker=observed_marker,
        observed_marker_size=observed_marker_size,
        observed_color=observed_color,
        reference_colors=reference_colors or {},
        ratio_reference_order=ratio_reference_order,
    )
    _plot_nnls_row(np.asarray(axes), row, metadata, window, normalize_basis_to_unit_area)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.88, wspace=0.42)
    _save(fig, output_dir, stem)


def plot_combined_nnls_paper_figure(
    rows: list[NNLSFigureRow],
    metadata: pd.DataFrame,
    window: list[float],
    output_dir: Path,
    normalize_basis_to_unit_area: bool,
    stem: str,
) -> None:
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(11.5, 3.05 * len(rows)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1, 1, 1.08]},
    )
    for row_axes, row in zip(axes, rows, strict=True):
        _plot_nnls_row(row_axes, row, metadata, window, normalize_basis_to_unit_area)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.94, wspace=0.42, hspace=0.58)
    _save(fig, output_dir, stem)
