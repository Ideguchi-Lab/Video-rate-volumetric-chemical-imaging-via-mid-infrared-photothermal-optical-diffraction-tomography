"""Read pre-analyzed MIR-ON/OFF pairing manifests."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MIPPair:
    """One same-illumination MIR-ON/OFF hologram pair."""

    volume_id: int
    on_index: int
    off_index: int
    on_path: Path
    off_path: Path


def _resolve_image_path(image_dir: Path, value: str, *, row_number: int) -> Path:
    relative_path = Path(value.strip())
    if relative_path.is_absolute():
        raise ValueError(f"Pairing row {row_number}: image paths must be relative, got {relative_path}")

    image_dir = image_dir.resolve()
    path = (image_dir / relative_path).resolve()
    try:
        path.relative_to(image_dir)
    except ValueError as exc:
        raise ValueError(f"Pairing row {row_number}: image path escapes the image directory: {relative_path}") from exc
    return path


def load_pairing_csv(csv_path: Path, image_dir: Path, *, require_images: bool = True) -> list[MIPPair]:
    """Load the legacy 2025-12-24 pairing CSV.

    The pre-analysis defined ``ImagePathA`` as MIR-ON and ``ImagePathB`` as
    MIR-OFF. The semantic names are assigned here so this convention does not
    leak into the reconstruction code.
    """
    required_columns = {"MipIndex", "ImageIndexA", "ImageIndexB", "ImagePathA", "ImagePathB"}
    if not csv_path.is_file():
        raise FileNotFoundError(f"Pairing CSV not found: {csv_path}")
    if not image_dir.is_dir() and require_images:
        raise FileNotFoundError(f"Hologram directory not found: {image_dir}")

    pairs: list[MIPPair] = []
    seen_indices: set[tuple[int, int, int]] = set()
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pairing CSV is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            try:
                volume_id = int(row["MipIndex"])
                on_index = int(row["ImageIndexA"])
                off_index = int(row["ImageIndexB"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Pairing row {row_number}: invalid integer field") from exc

            if volume_id < 0 or on_index < 0 or off_index < 0:
                raise ValueError(f"Pairing row {row_number}: indices must be non-negative")
            if off_index <= on_index:
                raise ValueError(
                    f"Pairing row {row_number}: MIR-OFF index must follow MIR-ON index ({on_index}, {off_index})"
                )
            identity = (volume_id, on_index, off_index)
            if identity in seen_indices:
                raise ValueError(f"Pairing row {row_number}: duplicate pair {identity}")
            seen_indices.add(identity)

            on_path = _resolve_image_path(image_dir, row["ImagePathA"], row_number=row_number)
            off_path = _resolve_image_path(image_dir, row["ImagePathB"], row_number=row_number)
            if require_images:
                for path in (on_path, off_path):
                    if not path.is_file():
                        raise FileNotFoundError(f"Pairing row {row_number}: hologram not found: {path}")

            pairs.append(MIPPair(volume_id, on_index, off_index, on_path, off_path))

    if not pairs:
        raise ValueError(f"Pairing CSV contains no pairs: {csv_path}")
    return pairs


def group_pairs_by_volume(pairs: Sequence[MIPPair]) -> dict[int, list[MIPPair]]:
    """Group pairs by volume while preserving acquisition order."""
    grouped: defaultdict[int, list[MIPPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.volume_id].append(pair)

    result = {
        volume_id: sorted(volume_pairs, key=lambda pair: pair.on_index) for volume_id, volume_pairs in grouped.items()
    }
    volume_ids = sorted(result)
    if volume_ids != list(range(volume_ids[-1] + 1)):
        raise ValueError(f"Volume IDs must be contiguous from zero, got {volume_ids}")
    return {volume_id: result[volume_id] for volume_id in volume_ids}
