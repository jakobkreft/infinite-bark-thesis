from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Iterable, Sequence
import numpy as np


@dataclass
class StatsResult:
    summary: dict
    per_image_rows: list[dict]
    class_pixel_counts: dict[int, int]
    class_pixel_fractions: dict[int, float]
    size_distribution: dict[str, int]
    cooccurrence_has_1_2: dict[str, int]
    adjacency_matrix: np.ndarray
    component_sizes: dict[int, list[int]]
    class_fraction_per_image: dict[int, list[float]]


def connected_component_sizes(mask: np.ndarray, target: int) -> list[int]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    out: list[int] = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or mask[y, x] != target:
                continue
            size = 0
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                size += 1

                if cy > 0 and not visited[cy - 1, cx] and mask[cy - 1, cx] == target:
                    visited[cy - 1, cx] = True
                    stack.append((cy - 1, cx))
                if cy < h - 1 and not visited[cy + 1, cx] and mask[cy + 1, cx] == target:
                    visited[cy + 1, cx] = True
                    stack.append((cy + 1, cx))
                if cx > 0 and not visited[cy, cx - 1] and mask[cy, cx - 1] == target:
                    visited[cy, cx - 1] = True
                    stack.append((cy, cx - 1))
                if cx < w - 1 and not visited[cy, cx + 1] and mask[cy, cx + 1] == target:
                    visited[cy, cx + 1] = True
                    stack.append((cy, cx + 1))

            out.append(size)
    return out


def _compute_adjacency(mask: np.ndarray, num_classes: int = 3) -> np.ndarray:
    # Directed 4-neighbor transitions P(neighbor=b | center=a)
    adj = np.zeros((num_classes, num_classes), dtype=np.int64)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            a = int(mask[y, x])
            if a < 0 or a >= num_classes:
                continue
            if y > 0:
                b = int(mask[y - 1, x])
                if 0 <= b < num_classes:
                    adj[a, b] += 1
            if y < h - 1:
                b = int(mask[y + 1, x])
                if 0 <= b < num_classes:
                    adj[a, b] += 1
            if x > 0:
                b = int(mask[y, x - 1])
                if 0 <= b < num_classes:
                    adj[a, b] += 1
            if x < w - 1:
                b = int(mask[y, x + 1])
                if 0 <= b < num_classes:
                    adj[a, b] += 1
    return adj


def compute_statistics(
    masks: Sequence[np.ndarray],
    names: Sequence[str] | None = None,
    classes: Sequence[int] = (0, 1, 2),
) -> StatsResult:
    if not masks:
        raise ValueError("Expected at least one mask")
    if names is None:
        names = [f"mask_{i:04d}" for i in range(len(masks))]

    classes = tuple(int(c) for c in classes)
    max_cls = max(classes)

    total_pixels = 0
    class_pixel_counts = {cls: 0 for cls in classes}
    size_distribution: Counter[str] = Counter()
    per_image_rows: list[dict] = []

    cooccurrence = Counter()

    adjacency = np.zeros((max_cls + 1, max_cls + 1), dtype=np.int64)
    component_sizes = {1: [], 2: []}

    class_fraction_per_image = {cls: [] for cls in classes}

    for name, mask in zip(names, masks):
        h, w = mask.shape
        n = int(h * w)
        total_pixels += n
        size_distribution[f"{h}x{w}"] += 1

        vals, counts = np.unique(mask, return_counts=True)
        local = {int(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())}

        row = {
            "name": name,
            "height": int(h),
            "width": int(w),
            "pixels_total": n,
        }

        has1 = local.get(1, 0) > 0
        has2 = local.get(2, 0) > 0
        cooccurrence[(has1, has2)] += 1

        for cls in classes:
            c = int(local.get(cls, 0))
            class_pixel_counts[cls] += c
            frac = c / n
            class_fraction_per_image[cls].append(float(frac))
            row[f"pixels_class_{cls}"] = c
            row[f"fraction_class_{cls}"] = float(frac)
            row[f"has_class_{cls}"] = bool(c > 0)

        per_image_rows.append(row)
        adjacency += _compute_adjacency(mask, num_classes=max_cls + 1)

        if 1 in component_sizes:
            component_sizes[1].extend(connected_component_sizes(mask, 1))
        if 2 in component_sizes:
            component_sizes[2].extend(connected_component_sizes(mask, 2))

    class_pixel_fractions = {
        cls: (class_pixel_counts[cls] / total_pixels if total_pixels > 0 else 0.0)
        for cls in classes
    }

    summary = {
        "num_images": len(masks),
        "total_pixels": total_pixels,
        "class_pixel_counts": {str(k): int(v) for k, v in class_pixel_counts.items()},
        "class_pixel_fractions": {str(k): float(v) for k, v in class_pixel_fractions.items()},
        "cooccurrence_has_class_1_and_2": {
            "has1_false_has2_false": int(cooccurrence[(False, False)]),
            "has1_true_has2_false": int(cooccurrence[(True, False)]),
            "has1_false_has2_true": int(cooccurrence[(False, True)]),
            "has1_true_has2_true": int(cooccurrence[(True, True)]),
        },
        "size_distribution": {k: int(v) for k, v in sorted(size_distribution.items())},
        "component_stats": {
            str(cls): summarize_distribution(component_sizes.get(cls, []))
            for cls in sorted(component_sizes.keys())
        },
    }

    return StatsResult(
        summary=summary,
        per_image_rows=per_image_rows,
        class_pixel_counts=class_pixel_counts,
        class_pixel_fractions=class_pixel_fractions,
        size_distribution=dict(sorted(size_distribution.items())),
        cooccurrence_has_1_2={
            "has1_false_has2_false": int(cooccurrence[(False, False)]),
            "has1_true_has2_false": int(cooccurrence[(True, False)]),
            "has1_false_has2_true": int(cooccurrence[(False, True)]),
            "has1_true_has2_true": int(cooccurrence[(True, True)]),
        },
        adjacency_matrix=adjacency[:3, :3].copy(),
        component_sizes=component_sizes,
        class_fraction_per_image=class_fraction_per_image,
    )


def summarize_distribution(values: Iterable[int | float]) -> dict:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "min": 0.0,
        }

    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }
