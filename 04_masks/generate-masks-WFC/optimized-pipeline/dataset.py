"""dataset.py – Load and validate the bark-mask dataset.

Each PNG is a single-channel label image with pixel values:
  0 = background
  1 = slepice
  2 = mehanske poskodbe
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import csv

import numpy as np
from PIL import Image


@dataclass
class MaskSample:
    name: str
    path: Path
    mask: np.ndarray  # shape (H, W), dtype uint8, values in {0, 1, 2}


def load_dataset(dataset_dir: Path, allowed_values: set[int] | None = {0, 1, 2}) -> list[MaskSample]:
    """Load all PNG masks from *dataset_dir*, sorted by filename."""
    paths = sorted(dataset_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {dataset_dir}")

    samples: list[MaskSample] = []
    for p in paths:
        arr = np.array(Image.open(p))
        if arr.ndim == 3:          # drop alpha / colour channels
            arr = arr[..., 0]
        arr = arr.astype(np.uint8)

        if allowed_values is not None:
            bad = set(int(v) for v in np.unique(arr).tolist()) - allowed_values
            if bad:
                raise ValueError(
                    f"{p.name}: unexpected pixel values {bad}. "
                    f"Allowed: {sorted(allowed_values)}"
                )

        samples.append(MaskSample(name=p.stem, path=p, mask=arr))

    return samples


def save_manifest(samples: list[MaskSample], out_dir: Path) -> None:
    """Write dataset_manifest.csv and dataset_summary.json to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for s in samples:
        h, w = s.mask.shape
        vals, counts = np.unique(s.mask, return_counts=True)
        row: dict = {"name": s.name, "height": h, "width": w}
        for v, c in zip(vals.tolist(), counts.tolist()):
            row[f"n_class_{v}"] = int(c)
        rows.append(row)

    keys = sorted({k for r in rows for k in r})
    with (out_dir / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_images": len(samples),
        "height_range": [int(min(s.mask.shape[0] for s in samples)),
                         int(max(s.mask.shape[0] for s in samples))],
        "width_range":  [int(min(s.mask.shape[1] for s in samples)),
                         int(max(s.mask.shape[1] for s in samples))],
        "unique_values": sorted({int(v) for s in samples for v in np.unique(s.mask).tolist()}),
    }
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
