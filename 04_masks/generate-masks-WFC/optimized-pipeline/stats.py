"""stats.py – Dataset statistics.

All heavy loops are vectorised with NumPy / SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as cc_label

from dataset import MaskSample

CLASSES = (0, 1, 2)
CLASS_NAMES = {0: "background", 1: "slepice", 2: "mehanske"}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PerImageStats:
    name: str
    height: int
    width: int
    n_pixels: int
    class_counts: dict[int, int]      # class -> pixel count
    class_fractions: dict[int, float] # class -> fraction of total pixels


@dataclass
class DatasetStats:
    per_image: list[PerImageStats]

    # Aggregate counts / fractions across all images
    total_pixels: int
    class_pixel_counts: dict[int, int]
    class_pixel_fractions: dict[int, float]

    # Co-occurrence: which combinations of class 1 / class 2 appear together
    cooccurrence: dict[str, int]  # keys: "only0", "1only", "2only", "both12"

    # Directed 4-neighbourhood transition count matrix, shape (3, 3)
    adjacency: np.ndarray

    # Connected-component size lists per class
    component_sizes: dict[int, list[int]]

    # Per-image class fraction lists, for box-plot-style viz
    class_fraction_per_image: dict[int, list[float]]

    # Image-size distribution
    size_distribution: dict[str, int]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_stats(samples: list[MaskSample]) -> DatasetStats:
    per_image: list[PerImageStats] = []
    total_pixels = 0
    class_pixel_counts = {c: 0 for c in CLASSES}
    adjacency = np.zeros((3, 3), dtype=np.int64)
    component_sizes: dict[int, list[int]] = {1: [], 2: []}
    class_fraction_per_image: dict[int, list[float]] = {c: [] for c in CLASSES}
    size_distribution: dict[str, int] = {}
    cooccurrence_keys = ["only0", "1only", "2only", "both12"]
    cooccurrence: dict[str, int] = {k: 0 for k in cooccurrence_keys}

    for s in samples:
        mask = s.mask
        h, w = mask.shape
        n = h * w
        total_pixels += n

        size_key = f"{h}x{w}"
        size_distribution[size_key] = size_distribution.get(size_key, 0) + 1

        # Per-class counts
        vals, counts = np.unique(mask, return_counts=True)
        local: dict[int, int] = {int(v): int(c) for v, c in zip(vals, counts)}
        cls_counts = {c: local.get(c, 0) for c in CLASSES}
        cls_fracs  = {c: cls_counts[c] / n for c in CLASSES}

        for c in CLASSES:
            class_pixel_counts[c] += cls_counts[c]
            class_fraction_per_image[c].append(cls_fracs[c])

        per_image.append(PerImageStats(
            name=s.name,
            height=h, width=w, n_pixels=n,
            class_counts=cls_counts, class_fractions=cls_fracs,
        ))

        # Co-occurrence
        has1 = cls_counts[1] > 0
        has2 = cls_counts[2] > 0
        if has1 and has2:
            cooccurrence["both12"] += 1
        elif has1:
            cooccurrence["1only"] += 1
        elif has2:
            cooccurrence["2only"] += 1
        else:
            cooccurrence["only0"] += 1

        # Adjacency (vectorised)
        adjacency += _adjacency(mask)

        # Connected components (SciPy)
        for cls in (1, 2):
            sizes = _component_sizes(mask, cls)
            component_sizes[cls].extend(sizes)

    class_pixel_fractions = {
        c: class_pixel_counts[c] / total_pixels if total_pixels > 0 else 0.0
        for c in CLASSES
    }

    return DatasetStats(
        per_image=per_image,
        total_pixels=total_pixels,
        class_pixel_counts=class_pixel_counts,
        class_pixel_fractions=class_pixel_fractions,
        cooccurrence=cooccurrence,
        adjacency=adjacency,
        component_sizes=component_sizes,
        class_fraction_per_image=class_fraction_per_image,
        size_distribution=size_distribution,
    )


def _adjacency(mask: np.ndarray) -> np.ndarray:
    """Vectorised directed 4-neighbourhood transition count matrix (3×3)."""
    adj = np.zeros((3, 3), dtype=np.int64)
    m = mask.astype(np.int32)
    valid = (m >= 0) & (m < 3)

    for a_slice, b_slice in [
        (m[:-1, :], m[1:, :]),   # vertical pairs
        (m[:, :-1], m[:, 1:]),   # horizontal pairs
    ]:
        v_slice_a = valid[:-1, :] if a_slice.shape == m[:-1, :].shape else valid[:, :-1]
        v_slice_b = valid[1:, :]  if b_slice.shape == m[1:, :].shape  else valid[:, 1:]

        vm = (valid[:-1, :] & valid[1:, :]) if a_slice is m[:-1, :] else (valid[:, :-1] & valid[:, 1:])

        a_v, b_v = a_slice[vm], b_slice[vm]
        np.add.at(adj, (a_v, b_v), 1)
        np.add.at(adj, (b_v, a_v), 1)   # symmetric (undirected totals)

    return adj


def _adjacency(mask: np.ndarray) -> np.ndarray:
    """Vectorised directed 4-neighbourhood transition count matrix (3×3)."""
    adj = np.zeros((3, 3), dtype=np.int64)
    m = mask.astype(np.int32)
    valid = (m >= 0) & (m < 3)

    # vertical pairs (row i, row i+1)
    vm = valid[:-1, :] & valid[1:, :]
    av, bv = m[:-1, :][vm], m[1:, :][vm]
    np.add.at(adj, (av, bv), 1)
    np.add.at(adj, (bv, av), 1)

    # horizontal pairs (col j, col j+1)
    hm = valid[:, :-1] & valid[:, 1:]
    ah, bh = m[:, :-1][hm], m[:, 1:][hm]
    np.add.at(adj, (ah, bh), 1)
    np.add.at(adj, (bh, ah), 1)

    return adj


def _component_sizes(mask: np.ndarray, target: int) -> list[int]:
    binary = (mask == target)
    if not binary.any():
        return []
    labeled, num = cc_label(binary)
    if num == 0:
        return []
    # bincount index 0 is background label; skip it
    counts = np.bincount(labeled.ravel())[1:]
    return counts.tolist()


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def distribution_summary(values: list[float | int]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0,
                "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "p10":    float(np.percentile(arr, 10)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
        "max":    float(arr.max()),
    }
