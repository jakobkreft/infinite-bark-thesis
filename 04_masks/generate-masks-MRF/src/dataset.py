"""
Dataset loading, augmentation, potential estimation, and neighborhood table
construction for bark mask MRF synthesis.
"""

import os
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.neighbors import KDTree


NUM_LABELS = 3
LABEL_NAMES = {0: "ozadje", 1: "slepice", 2: "mehanske_poskodbe"}


class BarkMaskDataset:
    def __init__(self, folder):
        self.folder = folder
        self.masks = []
        self._load_masks()

    def _load_masks(self):
        files = sorted(f for f in os.listdir(self.folder) if f.endswith(".png"))
        for f in files:
            img = np.array(Image.open(os.path.join(self.folder, f)))
            assert img.ndim == 2, f"Expected 2D mask, got {img.ndim}D for {f}"
            assert set(np.unique(img)).issubset({0, 1, 2}), (
                f"Unexpected labels in {f}: {np.unique(img)}"
            )
            self.masks.append(img.astype(np.int8))
        print(f"Loaded {len(self.masks)} masks from {self.folder}")

    def get_augmented_masks(self):
        """Return all masks with label-preserving augmentations (rotations + flips)."""
        augmented = []
        for m in self.masks:
            augmented.append(m)
            augmented.append(np.rot90(m, 1))
            augmented.append(np.rot90(m, 2))
            augmented.append(np.rot90(m, 3))
            augmented.append(np.fliplr(m))
            augmented.append(np.flipud(m))
            augmented.append(np.fliplr(np.rot90(m, 1)))
            augmented.append(np.flipud(np.rot90(m, 1)))
        return augmented

    def get_patch(self, size, toroidal=True):
        """Extract a random crop of given size from a random mask, with optional toroidal wrapping."""
        mask = self.masks[np.random.randint(len(self.masks))]
        h, w = mask.shape
        ph, pw = size if isinstance(size, tuple) else (size, size)
        si = np.random.randint(h)
        sj = np.random.randint(w)
        if toroidal:
            rows = [(si + di) % h for di in range(ph)]
            cols = [(sj + dj) % w for dj in range(pw)]
            return mask[np.ix_(rows, cols)]
        else:
            si = min(si, h - ph)
            sj = min(sj, w - pw)
            return mask[si : si + ph, sj : sj + pw]

    def estimate_class_frequencies(self):
        """Compute empirical P(c) over all pixels."""
        counts = np.zeros(NUM_LABELS, dtype=np.float64)
        total = 0
        for m in self.masks:
            for c in range(NUM_LABELS):
                counts[c] += np.sum(m == c)
            total += m.size
        freqs = counts / total
        print(f"Class frequencies: {dict(zip(LABEL_NAMES.values(), freqs))}")
        return freqs

    def estimate_pairwise_potentials(self):
        """
        Estimate direction-specific 3x3 compatibility matrices from training masks.
        Returns C_right, C_down, C_diag_dr, C_diag_dl as -log P(x_j=c2 | x_i=c1).
        """
        masks = self.get_augmented_masks()

        directions = {
            "right": (0, 1),
            "down": (1, 0),
            "diag_dr": (1, 1),
            "diag_dl": (1, -1),
        }
        counts = {name: np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.float64)
                  for name in directions}

        for m in masks:
            h, w = m.shape
            for name, (di, dj) in directions.items():
                for i in range(h):
                    for j in range(w):
                        ni = (i + di) % h
                        nj = (j + dj) % w
                        counts[name][m[i, j], m[ni, nj]] += 1

        potentials = {}
        for name, C in counts.items():
            # Normalize each row to get conditional probability
            row_sums = C.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            P = C / row_sums
            # Avoid log(0) with small epsilon
            P = np.clip(P, 1e-8, 1.0)
            potentials[name] = -np.log(P)

        return potentials

    def build_neighborhood_table(self, radius=3, use_augmented=True):
        """
        Build nonparametric conditional lookup table from training data.

        For each pixel, extract an L-shaped causal neighborhood (all pixels in
        the raster-order window that have already been synthesized). Store as:
        {neighborhood_bytes -> list of observed center labels}

        Also builds a KDTree for fallback nearest-neighbor lookup.

        Returns: (table_dict, kdtree, kdtree_labels, neighborhood_offsets)
        """
        masks = self.get_augmented_masks() if use_augmented else self.masks
        offsets = _causal_offsets(radius)

        table = defaultdict(list)
        all_neighborhoods = []
        all_labels = []

        for m in masks:
            h, w = m.shape
            for i in range(h):
                for j in range(w):
                    nbr = _extract_neighborhood(m, i, j, offsets, h, w)
                    key = nbr.tobytes()
                    table[key].append(m[i, j])
                    all_neighborhoods.append(nbr)
                    all_labels.append(m[i, j])

        # Build KDTree for fallback
        all_neighborhoods = np.array(all_neighborhoods, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int8)
        kdtree = KDTree(all_neighborhoods, leaf_size=40)

        print(f"Neighborhood table: {len(table)} unique patterns, "
              f"{len(all_labels)} total entries, radius={radius}")

        return table, kdtree, all_labels, offsets


def _causal_offsets(radius):
    """
    Generate L-shaped causal neighborhood offsets for raster-order synthesis.
    Includes all pixels in rows above (within radius), plus pixels to the left
    in the current row.
    """
    offsets = []
    for di in range(-radius, 0):
        for dj in range(-radius, radius + 1):
            offsets.append((di, dj))
    for dj in range(-radius, 0):
        offsets.append((0, dj))
    return offsets


def _full_neighborhood_offsets(radius):
    """All offsets in a square window of given radius, excluding center."""
    offsets = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            offsets.append((di, dj))
    return offsets


def _extract_neighborhood(grid, i, j, offsets, h, w):
    """Extract neighborhood values at given offsets with toroidal wrapping."""
    nbr = np.empty(len(offsets), dtype=np.int8)
    for k, (di, dj) in enumerate(offsets):
        ni = (i + di) % h
        nj = (j + dj) % w
        nbr[k] = grid[ni, nj]
    return nbr
