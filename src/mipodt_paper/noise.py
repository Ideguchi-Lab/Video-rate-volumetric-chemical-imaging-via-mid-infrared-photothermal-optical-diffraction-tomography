"""Noise statistics of MIP-ODT volumes acquired without MIR irradiation.

The supplementary noise figure uses a sequence of MIP-ODT reconstructions in
which the MIR laser stayed off, so every "MIR-ON/OFF" pair is really an
OFF/OFF pair and the reconstructed RI change contains only noise. The noise
floor is the voxel-wise standard deviation of the real part across that
sequence, computed with Welford's online algorithm so that only one volume is
resident at a time.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Region:
    """NumPy-style half-open box ``volume[x0:x1, y0:y1, z0:z1]``."""

    name: str
    x: tuple[int, int]
    y: tuple[int, int]
    z: tuple[int, int]

    def validate(self, shape: Sequence[int]) -> None:
        if not self.name.strip():
            raise ValueError("Region name must not be empty")
        for axis_name, (start, end), size in zip(("x", "y", "z"), (self.x, self.y, self.z), shape, strict=True):
            if start < 0 or end <= start:
                raise ValueError(f"Region '{self.name}' has an invalid {axis_name} range: {start}, {end}")
            if end > size:
                raise ValueError(f"Region '{self.name}' {axis_name} end {end} exceeds the axis length {size}")

    def selection(self) -> tuple[slice, slice, slice]:
        return (slice(*self.x), slice(*self.y), slice(*self.z))

    @classmethod
    def from_mapping(cls, raw: dict) -> Region:
        def axis(key: str) -> tuple[int, int]:
            value = raw[key]
            return (int(value["start"]), int(value["end"]))

        return cls(name=str(raw["name"]), x=axis("x"), y=axis("y"), z=axis("z"))


def voxel_std(
    volume_paths: Iterable[Path],
    *,
    expected_shape: Sequence[int] | None = None,
    dtype: type[np.floating] = np.float32,
    progress: Callable[[Path], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Sample standard deviation of the real part of each voxel across volumes.

    Welford's update is evaluated in ``dtype``. Returns ``(std, n_volumes)``
    where ``std`` is a ``float32`` array with the volume shape.
    """
    scalar = np.dtype(dtype).type
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    count = 0
    for path in volume_paths:
        if progress is not None:
            progress(path)
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {volume.shape}: {path}")
        if expected_shape is not None and tuple(volume.shape) != tuple(expected_shape):
            raise ValueError(f"Expected {tuple(expected_shape)}, got {volume.shape}: {path}")
        values = np.array(np.real(volume), dtype=dtype, copy=True)
        del volume
        count += 1
        if mean is None:
            mean = values
            m2 = np.zeros_like(values)
            continue
        assert m2 is not None
        delta = values - mean
        mean += delta / scalar(count)
        m2 += delta * (values - mean)
    if mean is None or m2 is None:
        raise ValueError("No volumes were supplied")
    if count < 2:
        raise ValueError("At least two volumes are required to compute a standard deviation")
    return np.sqrt(m2 / scalar(count - 1)).astype(np.float32, copy=False), count


def region_means(volume: np.ndarray, regions: Sequence[Region]) -> list[dict[str, object]]:
    """Mean of ``volume`` inside each region as CSV rows."""
    rows: list[dict[str, object]] = []
    for region in regions:
        region.validate(volume.shape)
        roi = np.asarray(np.real(volume[region.selection()]))
        rows.append(
            {
                "name": region.name,
                "x_start": region.x[0],
                "x_end": region.x[1],
                "y_start": region.y[0],
                "y_end": region.y[1],
                "z_start": region.z[0],
                "z_end": region.z[1],
                "count": int(roi.size),
                "mean": float(np.mean(roi, dtype=np.float64)),
            }
        )
    return rows


REGION_MEAN_COLUMNS = ("name", "x_start", "x_end", "y_start", "y_end", "z_start", "z_end", "count", "mean")


def write_region_means_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(REGION_MEAN_COLUMNS)
        for row in rows:
            writer.writerow([row[column] for column in REGION_MEAN_COLUMNS])
