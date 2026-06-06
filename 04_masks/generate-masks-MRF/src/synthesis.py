"""
Nonparametric Gibbs sampler for MRF mask synthesis.
Supports raster-order synthesis, multiscale initialization,
Gibbs refinement, and toroidal boundary conditions.

Optimized for performance with pre-computed offset arrays, cached count
distributions, and minimized per-pixel Python overhead.
"""

import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.neighbors import KDTree

from .dataset import (
    _causal_offsets,
    _full_neighborhood_offsets,
    NUM_LABELS,
)


def synthesize_mask(
    dataset,
    height=256,
    width=256,
    radius=3,
    n_refine=3,
    multiscale=True,
    target_ratio=None,
    lambda_ratio=1.0,
    temperature=1.0,
    k_fallback=11,
    seed=None,
):
    """
    Synthesize a new label mask using nonparametric Gibbs sampling.

    Returns:
        (height, width) int8 array with labels in {0, 1, 2}
    """
    if seed is not None:
        np.random.seed(seed)

    class_freqs = dataset.estimate_class_frequencies()

    # Compute proportion bias
    proportion_bias = np.zeros(NUM_LABELS, dtype=np.float64)
    if target_ratio is not None:
        target = np.clip(np.array(target_ratio, dtype=np.float64), 1e-8, 1.0)
        proportion_bias = lambda_ratio * np.log(
            target / np.clip(class_freqs, 1e-8, 1.0)
        )

    # Build causal neighborhood table
    print("Building causal neighborhood lookup table...")
    causal_offset_list = _causal_offsets(radius)
    causal_di = np.array([o[0] for o in causal_offset_list], dtype=np.int32)
    causal_dj = np.array([o[1] for o in causal_offset_list], dtype=np.int32)
    causal_table, causal_kdtree, causal_kd_labels = _build_table(
        dataset, causal_di, causal_dj
    )

    if multiscale:
        canvas = _multiscale_synthesize(
            dataset, causal_table, causal_kdtree, causal_kd_labels,
            causal_di, causal_dj, class_freqs,
            proportion_bias, height, width, radius, temperature, k_fallback,
        )
    else:
        canvas = _init_canvas(height, width, class_freqs)
        _raster_pass(
            canvas, causal_table, causal_kdtree, causal_kd_labels,
            causal_di, causal_dj, proportion_bias, temperature, k_fallback,
            desc="Raster synthesis",
        )

    # Build full neighborhood table for Gibbs refinement
    full_offset_list = _full_neighborhood_offsets(radius)
    full_di = np.array([o[0] for o in full_offset_list], dtype=np.int32)
    full_dj = np.array([o[1] for o in full_offset_list], dtype=np.int32)
    print("Building full-neighborhood table for Gibbs refinement...")
    full_table, full_kdtree, full_kd_labels = _build_table(
        dataset, full_di, full_dj
    )

    for sweep in range(n_refine):
        _gibbs_sweep(
            canvas, full_table, full_kdtree, full_kd_labels,
            full_di, full_dj, proportion_bias, temperature, k_fallback,
            desc=f"Gibbs refinement {sweep + 1}/{n_refine}",
        )

    return canvas


def _init_canvas(height, width, class_freqs):
    """Initialize canvas by sampling from class frequencies."""
    return np.random.choice(
        NUM_LABELS, size=(height, width), p=class_freqs
    ).astype(np.int8)


def _multiscale_synthesize(
    dataset, table, kdtree, kd_labels, offset_di, offset_dj, class_freqs,
    proportion_bias, height, width, radius, temperature, k_fallback,
):
    """Synthesize at 1/4 resolution, then upsample and refine at full resolution."""
    from scipy.ndimage import zoom

    ch, cw = max(height // 4, 8), max(width // 4, 8)
    print(f"Multiscale: synthesizing coarse {ch}x{cw}...")
    coarse = _init_canvas(ch, cw, class_freqs)
    _raster_pass(
        coarse, table, kdtree, kd_labels, offset_di, offset_dj,
        proportion_bias, temperature, k_fallback, desc="Coarse raster",
    )

    # Upsample (nearest neighbor)
    canvas = zoom(coarse.astype(np.float32),
                  (height / ch, width / cw), order=0).astype(np.int8)
    canvas = canvas[:height, :width].copy()
    if canvas.shape != (height, width):
        padded = np.zeros((height, width), dtype=np.int8)
        padded[:canvas.shape[0], :canvas.shape[1]] = canvas
        canvas = padded

    print(f"Multiscale: refining at full {height}x{width}...")
    _raster_pass(
        canvas, table, kdtree, kd_labels, offset_di, offset_dj,
        proportion_bias, temperature, k_fallback, desc="Fine raster",
    )
    return canvas


def _extract_nbr(grid, i, j, offset_di, offset_dj, h, w):
    """Fast neighborhood extraction using numpy vectorized modular indexing."""
    rows = (i + offset_di) % h
    cols = (j + offset_dj) % w
    return grid[rows, cols]


def _raster_pass(
    canvas, table, kdtree, kd_labels, offset_di, offset_dj,
    proportion_bias, temperature, k_fallback, desc="Raster",
):
    """Single raster-order synthesis pass over the canvas."""
    h, w = canvas.shape
    rand_vals = np.random.random(h * w)
    idx = 0
    for i in tqdm(range(h), desc=desc, leave=False):
        for j in range(w):
            nbr = _extract_nbr(canvas, i, j, offset_di, offset_dj, h, w)
            probs = _cond_dist(
                nbr, table, kdtree, kd_labels,
                proportion_bias, temperature, k_fallback,
            )
            # Fast sampling: cumulative sum + searchsorted
            cumprobs = np.cumsum(probs)
            canvas[i, j] = np.searchsorted(cumprobs, rand_vals[idx])
            idx += 1


def _gibbs_sweep(
    canvas, table, kdtree, kd_labels, offset_di, offset_dj,
    proportion_bias, temperature, k_fallback, desc="Gibbs",
):
    """One full Gibbs sweep: visit every pixel in random order, resample."""
    h, w = canvas.shape
    n = h * w
    indices = np.arange(n)
    np.random.shuffle(indices)
    rand_vals = np.random.random(n)
    for idx_pos in tqdm(range(n), desc=desc, leave=False):
        flat_idx = indices[idx_pos]
        i, j = divmod(int(flat_idx), w)
        nbr = _extract_nbr(canvas, i, j, offset_di, offset_dj, h, w)
        probs = _cond_dist(
            nbr, table, kdtree, kd_labels,
            proportion_bias, temperature, k_fallback,
        )
        cumprobs = np.cumsum(probs)
        canvas[i, j] = np.searchsorted(cumprobs, rand_vals[idx_pos])


def _cond_dist(neighborhood, table, kdtree, kd_labels,
               proportion_bias, temperature, k_fallback):
    """
    Compute P(x_i | neighborhood) from the nonparametric table,
    with KDTree fallback and proportion bias.
    """
    key = neighborhood.tobytes()
    counts = table.get(key)

    if counts is not None:
        # Exact match — counts is a pre-computed (3,) float64 array
        log_probs = np.log(counts.clip(min=1e-8))
    else:
        # KDTree fallback: find k nearest neighborhoods
        query = neighborhood.astype(np.float32).reshape(1, -1)
        dists, inds = kdtree.query(query, k=k_fallback)
        dists = dists[0]
        inds = inds[0]
        # Distance-weighted voting
        sigma = max(dists.mean(), 1e-6)
        weights = np.exp(-dists / (2.0 * sigma))
        counts_arr = np.zeros(NUM_LABELS, dtype=np.float64)
        for k_i in range(len(inds)):
            counts_arr[kd_labels[inds[k_i]]] += weights[k_i]
        log_probs = np.log(counts_arr.clip(min=1e-8))

    # Proportion bias + temperature
    log_probs += proportion_bias
    log_probs /= max(temperature, 1e-6)

    # Normalize
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()
    return probs


def _build_table(dataset, offset_di, offset_dj):
    """
    Build nonparametric lookup table from training data.
    Table values are pre-computed count arrays (NUM_LABELS,) instead of lists.
    Also builds KDTree for fallback.
    """
    masks = dataset.get_augmented_masks()
    raw_table = defaultdict(lambda: np.zeros(NUM_LABELS, dtype=np.float64))
    all_neighborhoods = []
    all_labels = []

    for m in masks:
        h, w = m.shape
        for i in range(h):
            for j in range(w):
                nbr = _extract_nbr(m, i, j, offset_di, offset_dj, h, w)
                raw_table[nbr.tobytes()][m[i, j]] += 1.0
                all_neighborhoods.append(nbr.astype(np.float32))
                all_labels.append(m[i, j])

    # Freeze table (convert defaultdict to regular dict)
    table = dict(raw_table)

    all_neighborhoods = np.array(all_neighborhoods, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int8)
    kdtree = KDTree(all_neighborhoods, leaf_size=40)

    print(f"  {len(table)} unique patterns, {len(all_labels)} total entries")
    return table, kdtree, all_labels
