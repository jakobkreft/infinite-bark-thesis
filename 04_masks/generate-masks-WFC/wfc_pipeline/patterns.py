from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import json
import numpy as np

from .utils import ensure_dir, save_json

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class PatternLibrary:
    pattern_size: int
    patterns: np.ndarray  # [P, n, n] uint8
    weights: np.ndarray   # [P] int64
    compat: dict[str, np.ndarray]  # direction -> [P, P] bool

    def save(self, out_dir: Path) -> None:
        ensure_dir(out_dir)
        np.savez_compressed(
            out_dir / "pattern_library.npz",
            pattern_size=np.array([self.pattern_size], dtype=np.int32),
            patterns=self.patterns,
            weights=self.weights,
            compat_up=self.compat["up"],
            compat_down=self.compat["down"],
            compat_left=self.compat["left"],
            compat_right=self.compat["right"],
        )

        payload = {
            "pattern_size": int(self.pattern_size),
            "num_patterns": int(self.patterns.shape[0]),
            "weight_sum": int(self.weights.sum()),
            "weight_min": int(self.weights.min()) if self.weights.size else 0,
            "weight_max": int(self.weights.max()) if self.weights.size else 0,
            "compatibility_density": {
                d: float(self.compat[d].mean()) for d in ["up", "down", "left", "right"]
            },
        }
        save_json(payload, out_dir / "pattern_summary.json")

    @staticmethod
    def load(path: Path) -> "PatternLibrary":
        data = np.load(path, allow_pickle=False)
        n = int(data["pattern_size"][0])
        patterns = data["patterns"].astype(np.uint8)
        weights = data["weights"].astype(np.int64)
        compat = {
            "up": data["compat_up"].astype(bool),
            "down": data["compat_down"].astype(bool),
            "left": data["compat_left"].astype(bool),
            "right": data["compat_right"].astype(bool),
        }
        return PatternLibrary(pattern_size=n, patterns=patterns, weights=weights, compat=compat)


def _augmentations(mask: np.ndarray) -> list[np.ndarray]:
    base = mask
    rots = [np.rot90(base, k) for k in range(4)]
    flips = [np.fliplr(r) for r in rots]
    all_aug = rots + flips

    # Deduplicate exact duplicates for symmetric masks.
    uniq: dict[bytes, np.ndarray] = {}
    for a in all_aug:
        uniq[a.tobytes()] = a
    return list(uniq.values())


def extract_patterns(
    masks: list[np.ndarray],
    pattern_size: int,
    augment_symmetry: bool = True,
    min_weight: int = 1,
    max_patterns: int | None = None,
) -> PatternLibrary:
    if pattern_size < 2:
        raise ValueError("pattern_size must be >= 2")

    pattern_to_id: dict[bytes, int] = {}
    patterns: list[np.ndarray] = []
    weights: list[int] = []

    for mask in masks:
        variants = _augmentations(mask) if augment_symmetry else [mask]
        for var in variants:
            h, w = var.shape
            if h < pattern_size or w < pattern_size:
                continue
            for y in range(h - pattern_size + 1):
                for x in range(w - pattern_size + 1):
                    p = var[y : y + pattern_size, x : x + pattern_size]
                    key = p.tobytes()
                    pid = pattern_to_id.get(key)
                    if pid is None:
                        pid = len(patterns)
                        pattern_to_id[key] = pid
                        patterns.append(p.copy())
                        weights.append(1)
                    else:
                        weights[pid] += 1

    if not patterns:
        raise ValueError("No patterns extracted. Check pattern_size and dataset dimensions.")

    pat_arr = np.stack(patterns).astype(np.uint8)
    w_arr = np.array(weights, dtype=np.int64)

    # Keep frequent patterns to reduce noise and improve generation speed.
    keep_mask = w_arr >= int(max(1, min_weight))
    if not np.any(keep_mask):
        raise ValueError(
            f"No patterns satisfy min_weight={min_weight}. Reduce threshold."
        )
    pat_arr = pat_arr[keep_mask]
    w_arr = w_arr[keep_mask]

    if max_patterns is not None and int(max_patterns) > 0 and pat_arr.shape[0] > int(max_patterns):
        idx = np.argsort(w_arr)[::-1][: int(max_patterns)]
        pat_arr = pat_arr[idx]
        w_arr = w_arr[idx]

    compat = compute_compatibility(pat_arr)
    return PatternLibrary(pattern_size=pattern_size, patterns=pat_arr, weights=w_arr, compat=compat)


def compute_compatibility(patterns: np.ndarray) -> dict[str, np.ndarray]:
    p, n, _ = patterns.shape
    comp = {
        "up": np.zeros((p, p), dtype=bool),
        "down": np.zeros((p, p), dtype=bool),
        "left": np.zeros((p, p), dtype=bool),
        "right": np.zeros((p, p), dtype=bool),
    }

    for a in range(p):
        pa = patterns[a]
        for b in range(p):
            pb = patterns[b]
            # b above a (neighbor in "up" direction)
            if np.array_equal(pa[:-1, :], pb[1:, :]):
                comp["up"][a, b] = True
            # b below a
            if np.array_equal(pa[1:, :], pb[:-1, :]):
                comp["down"][a, b] = True
            # b left of a
            if np.array_equal(pa[:, :-1], pb[:, 1:]):
                comp["left"][a, b] = True
            # b right of a
            if np.array_equal(pa[:, 1:], pb[:, :-1]):
                comp["right"][a, b] = True

    return comp


def compatibility_report(lib: PatternLibrary) -> dict:
    p = int(lib.patterns.shape[0])
    degrees = defaultdict(dict)
    for d in ["up", "down", "left", "right"]:
        mat = lib.compat[d]
        out_deg = mat.sum(axis=1)
        degrees[d] = {
            "mean_out_degree": float(np.mean(out_deg)),
            "min_out_degree": int(np.min(out_deg)),
            "max_out_degree": int(np.max(out_deg)),
            "density": float(np.mean(mat)),
        }
    return {
        "num_patterns": p,
        "pattern_size": int(lib.pattern_size),
        "directions": degrees,
    }


def save_compatibility_report(lib: PatternLibrary, out_path: Path) -> None:
    save_json(compatibility_report(lib), out_path)
