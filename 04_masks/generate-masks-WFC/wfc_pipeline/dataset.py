from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image

from .utils import ensure_dir, save_json, save_csv


@dataclass
class MaskSample:
    name: str
    path: Path
    mask: np.ndarray


@dataclass
class DatasetBundle:
    samples: list[MaskSample]
    unique_values: list[int]

    @property
    def masks(self) -> list[np.ndarray]:
        return [s.mask for s in self.samples]


def _load_single_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def load_dataset(dataset_dir: Path, allowed_values: set[int] | None = None) -> DatasetBundle:
    paths = sorted(dataset_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {dataset_dir}")

    samples: list[MaskSample] = []
    values: set[int] = set()
    for p in paths:
        mask = _load_single_mask(p)
        vals = set(int(v) for v in np.unique(mask).tolist())
        values.update(vals)
        samples.append(MaskSample(name=p.stem, path=p, mask=mask))

    if allowed_values is not None:
        unknown = sorted(v for v in values if v not in allowed_values)
        if unknown:
            raise ValueError(
                f"Unexpected class values in dataset: {unknown}. Allowed values: {sorted(allowed_values)}"
            )

    return DatasetBundle(samples=samples, unique_values=sorted(values))


def save_dataset_manifest(bundle: DatasetBundle, out_dir: Path) -> None:
    ensure_dir(out_dir)

    rows = []
    for s in bundle.samples:
        h, w = s.mask.shape
        vals, counts = np.unique(s.mask, return_counts=True)
        row = {
            "name": s.name,
            "path": str(s.path),
            "height": int(h),
            "width": int(w),
        }
        for v, c in zip(vals.tolist(), counts.tolist()):
            row[f"pixels_class_{int(v)}"] = int(c)
        rows.append(row)

    summary = {
        "num_images": len(bundle.samples),
        "unique_values": bundle.unique_values,
        "height_min": int(min(s.mask.shape[0] for s in bundle.samples)),
        "height_max": int(max(s.mask.shape[0] for s in bundle.samples)),
        "width_min": int(min(s.mask.shape[1] for s in bundle.samples)),
        "width_max": int(max(s.mask.shape[1] for s in bundle.samples)),
    }

    save_json(summary, out_dir / "dataset_summary.json")
    save_csv(rows, out_dir / "dataset_manifest.csv")


def iterate_masks(bundle: DatasetBundle) -> Iterable[np.ndarray]:
    for s in bundle.samples:
        yield s.mask
