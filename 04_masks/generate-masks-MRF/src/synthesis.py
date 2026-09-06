"""
Nonparametric Gibbs sampler for MRF mask synthesis.
Supports raster-order synthesis, multiscale initialization,
Gibbs refinement, and independently-configurable toroidal
(tileable) boundary conditions per axis.

Optimized for performance with pre-computed offset arrays, cached count
distributions, and minimized per-pixel Python overhead.

The expensive part — the causal/full neighborhood lookup tables built from
the training dataset — is precomputed once and passed in as a `TrainedModel`
(see `model_cache.py`) rather than rebuilt on every call.
"""

import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.neighbors import KDTree

from .dataset import NUM_LABELS


def synthesize_mask(
    trained_model,
    height=256,
    width=256,
    n_refine=3,
    multiscale=True,
    target_ratio=None,
    lambda_ratio=1.0,
    temperature=1.0,
    k_fallback=11,
    seed=None,
    tileable_v=True,
    tileable_h=True,
    verbose=True,
):
    """
    Synthesize a new label mask using nonparametric Gibbs sampling.

    Args:
        trained_model: a `model_cache.TrainedModel` bundling the class
            frequencies and causal/full neighborhood tables precomputed
            from the training dataset (see `model_cache.load_or_build_model`).
        tileable_v: wrap toroidally along the vertical axis (top edge
            connects to bottom edge), so the output tiles seamlessly
            top-to-bottom.
        tileable_h: wrap toroidally along the horizontal axis (left edge
            connects to right edge), so the output tiles seamlessly
            left-to-right.
        verbose: show per-pass tqdm progress bars and status prints.

    Returns:
        (height, width) int8 array with labels in {0, 1, 2}
    """
    if seed is not None:
        np.random.seed(seed)

    class_freqs = trained_model.class_freqs
    ratio_target = _normalize_target_ratio(target_ratio)
    if ratio_target is not None and lambda_ratio <= 0:
        ratio_target = None

    if multiscale:
        canvas = _multiscale_synthesize(
            trained_model.causal_table, trained_model.causal_kdtree,
            trained_model.causal_kd_counts,
            trained_model.causal_di, trained_model.causal_dj, class_freqs,
            ratio_target, lambda_ratio, height, width, temperature, k_fallback,
            tileable_v, tileable_h, verbose,
        )
    else:
        canvas = _init_canvas(height, width, class_freqs, ratio_target)
        ratio_control = _RatioController(canvas, ratio_target, lambda_ratio)
        _raster_pass(
            canvas, trained_model.causal_table, trained_model.causal_kdtree,
            trained_model.causal_kd_counts,
            trained_model.causal_di, trained_model.causal_dj,
            ratio_control, temperature, k_fallback,
            tileable_v, tileable_h, desc="Raster synthesis", verbose=verbose,
        )

    ratio_control = _RatioController(canvas, ratio_target, lambda_ratio)
    for sweep in range(n_refine):
        _gibbs_sweep(
            canvas, trained_model.full_table, trained_model.full_kdtree,
            trained_model.full_kd_counts,
            trained_model.full_di, trained_model.full_dj,
            ratio_control, temperature, k_fallback,
            tileable_v, tileable_h,
            desc=f"Gibbs refinement {sweep + 1}/{n_refine}", verbose=verbose,
        )

    return canvas


def _init_canvas(height, width, class_freqs, target_ratio=None):
    """Initialize canvas from target counts when provided, else frequencies."""
    if target_ratio is not None:
        counts = _integer_target_counts(target_ratio, height * width)
        labels = np.repeat(np.arange(NUM_LABELS, dtype=np.int8), counts)
        np.random.shuffle(labels)
        return labels.reshape(height, width)

    return np.random.choice(
        NUM_LABELS, size=(height, width), p=class_freqs
    ).astype(np.int8)


def _multiscale_synthesize(
    table, kdtree, kd_counts, offset_di, offset_dj, class_freqs,
    target_ratio, lambda_ratio, height, width, temperature, k_fallback,
    tileable_v=True, tileable_h=True, verbose=True,
):
    """Synthesize at 1/4 resolution, then upsample and refine at full resolution."""
    ch, cw = max(height // 4, 8), max(width // 4, 8)
    if verbose:
        print(f"Multiscale: synthesizing coarse {ch}x{cw}...")
    coarse = _init_canvas(ch, cw, class_freqs, target_ratio)
    coarse_ratio = _RatioController(coarse, target_ratio, lambda_ratio)
    _raster_pass(
        coarse, table, kdtree, kd_counts, offset_di, offset_dj,
        coarse_ratio, temperature, k_fallback,
        tileable_v, tileable_h, desc="Coarse raster", verbose=verbose,
    )

    canvas = _nn_upsample_labels(coarse, height, width)

    if verbose:
        print(f"Multiscale: refining at full {height}x{width}...")
    fine_ratio = _RatioController(canvas, target_ratio, lambda_ratio)
    _raster_pass(
        canvas, table, kdtree, kd_counts, offset_di, offset_dj,
        fine_ratio, temperature, k_fallback,
        tileable_v, tileable_h, desc="Fine raster", verbose=verbose,
    )
    return canvas


def _nn_upsample_labels(coarse, height, width):
    """Nearest-neighbor upsample a coarse label grid to (height, width)."""
    from scipy.ndimage import zoom

    ch, cw = coarse.shape
    canvas = zoom(coarse.astype(np.float32),
                  (height / ch, width / cw), order=0).astype(np.int8)
    canvas = canvas[:height, :width].copy()
    if canvas.shape != (height, width):
        padded = np.zeros((height, width), dtype=np.int8)
        padded[:canvas.shape[0], :canvas.shape[1]] = canvas
        canvas = padded
    return canvas


def _nn_downsample_labels(fine, ch, cw):
    """Nearest-neighbor downsample a label grid to (ch, cw)."""
    from scipy.ndimage import zoom

    h, w = fine.shape
    return zoom(fine.astype(np.float32), (ch / h, cw / w), order=0).astype(np.int8)


def _normalize_target_ratio(target_ratio):
    """Return a valid normalized target ratio or None."""
    if target_ratio is None:
        return None
    target = np.asarray(target_ratio, dtype=np.float64)
    if target.shape != (NUM_LABELS,):
        raise ValueError(f"target_ratio must have {NUM_LABELS} entries")
    if np.any(target < 0) or not np.all(np.isfinite(target)):
        raise ValueError("target_ratio must contain finite non-negative values")
    total = target.sum()
    if total <= 0:
        raise ValueError("target_ratio must sum to a positive value")
    return target / total


def _integer_target_counts(target_ratio, n_pixels):
    """Round a ratio to integer label counts that sum exactly to n_pixels."""
    raw = np.asarray(target_ratio, dtype=np.float64) * n_pixels
    counts = np.floor(raw).astype(np.int64)
    remainder = int(n_pixels - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


class _RatioController:
    """Count-aware global ratio penalty for single-site updates."""

    def __init__(self, canvas, target_ratio, lambda_ratio):
        self.enabled = target_ratio is not None and lambda_ratio > 0
        self.lambda_ratio = float(lambda_ratio)
        if not self.enabled:
            self.counts = None
            self.target_counts = None
            self._labels = None
            return

        self.target_counts = _integer_target_counts(target_ratio, canvas.size)
        self.counts = np.bincount(
            canvas.ravel(), minlength=NUM_LABELS
        ).astype(np.int64)
        self._labels = np.arange(NUM_LABELS)

    def label_bias(self, old_label):
        """Log-bias candidate labels by the resulting global count error."""
        if not self.enabled:
            return None

        counts_without = self.counts.copy()
        counts_without[int(old_label)] -= 1
        diff = counts_without - self.target_counts
        bias = (-self.lambda_ratio * (2 * diff + 1)).astype(np.float64)

        # A zero target should mean the class is not sampled at all.
        zero_target = self.target_counts == 0
        if np.any(zero_target):
            bias[zero_target & (self._labels != int(old_label))] = -np.inf
        return bias

    def replace(self, old_label, new_label):
        if not self.enabled:
            return
        old_label = int(old_label)
        new_label = int(new_label)
        if old_label == new_label:
            return
        self.counts[old_label] -= 1
        self.counts[new_label] += 1


def _extract_nbr(grid, i, j, offset_di, offset_dj, h, w,
                  tileable_v=True, tileable_h=True):
    """Fast neighborhood extraction using numpy vectorized indexing.

    Axes marked tileable wrap modularly; non-tileable axes clamp to the
    edge (replicate the border pixel) so the neighborhood vector always has
    a fixed shape regardless of boundary handling.
    """
    rows = i + offset_di
    cols = j + offset_dj
    rows = rows % h if tileable_v else np.clip(rows, 0, h - 1)
    cols = cols % w if tileable_h else np.clip(cols, 0, w - 1)
    return grid[rows, cols]


def _raster_pass(
    canvas, table, kdtree, kd_counts, offset_di, offset_dj,
    ratio_control, temperature, k_fallback,
    tileable_v=True, tileable_h=True, desc="Raster", verbose=True,
    update_mask=None,
):
    """Single raster-order synthesis pass over the canvas.

    If `update_mask` is given, only positions where it's True are visited
    and resampled (in raster order); other positions are left untouched but
    still used as context when they fall inside a neighbor's window. This is
    how inpainting reuses this function: pass the "unknown" mask so known
    pixels act as fixed boundary conditions.
    """
    h, w = canvas.shape
    flat_indices = np.arange(h * w) if update_mask is None else np.flatnonzero(update_mask)
    rand_vals = np.random.random(len(flat_indices))
    counts_cache = {}
    for idx, flat_idx in enumerate(tqdm(flat_indices, desc=desc, leave=False, disable=not verbose)):
        i, j = divmod(int(flat_idx), w)
        nbr = _extract_nbr(canvas, i, j, offset_di, offset_dj, h, w,
                            tileable_v, tileable_h)
        old_label = int(canvas[i, j])
        probs = _cond_dist(
            nbr, table, kdtree, kd_counts,
            ratio_control.label_bias(old_label), temperature, k_fallback,
            counts_cache,
        )
        # Fast sampling: cumulative sum + searchsorted
        cumprobs = np.cumsum(probs)
        new_label = int(np.searchsorted(cumprobs, rand_vals[idx]))
        canvas[i, j] = new_label
        ratio_control.replace(old_label, new_label)


def _gibbs_sweep(
    canvas, table, kdtree, kd_counts, offset_di, offset_dj,
    ratio_control, temperature, k_fallback,
    tileable_v=True, tileable_h=True, desc="Gibbs", verbose=True,
    update_mask=None,
):
    """One full Gibbs sweep: visit every pixel in random order, resample.

    If `update_mask` is given, only positions where it's True are ever
    resampled (see `_raster_pass`).
    """
    h, w = canvas.shape
    flat_indices = np.arange(h * w) if update_mask is None else np.flatnonzero(update_mask)
    np.random.shuffle(flat_indices)
    n = len(flat_indices)
    rand_vals = np.random.random(n)
    counts_cache = {}
    for idx_pos in tqdm(range(n), desc=desc, leave=False, disable=not verbose):
        flat_idx = flat_indices[idx_pos]
        i, j = divmod(int(flat_idx), w)
        nbr = _extract_nbr(canvas, i, j, offset_di, offset_dj, h, w,
                            tileable_v, tileable_h)
        old_label = int(canvas[i, j])
        probs = _cond_dist(
            nbr, table, kdtree, kd_counts,
            ratio_control.label_bias(old_label), temperature, k_fallback,
            counts_cache,
        )
        cumprobs = np.cumsum(probs)
        new_label = int(np.searchsorted(cumprobs, rand_vals[idx_pos]))
        canvas[i, j] = new_label
        ratio_control.replace(old_label, new_label)


def _cond_dist(neighborhood, table, kdtree, kd_counts,
               label_bias, temperature, k_fallback, counts_cache=None):
    """
    Compute P(x_i | neighborhood) from the nonparametric table,
    with KDTree fallback and optional label bias.
    """
    key = neighborhood.tobytes()
    counts = table.get(key)

    if counts is not None:
        counts_arr = np.asarray(counts, dtype=np.float64)
        log_probs = np.log(counts_arr.clip(min=1e-8))
    else:
        counts_arr = None if counts_cache is None else counts_cache.get(key)
        if counts_arr is None:
            # KDTree fallback: find nearest unique observed neighborhoods and
            # weight each one's full class-count vector by distance.
            query = neighborhood.astype(np.float64).reshape(1, -1)
            k = min(k_fallback, len(kd_counts))
            dists, inds = kdtree.query(query, k=k)
            dists = dists[0]
            inds = inds[0]
            sigma = max(dists.mean(), 1e-6)
            weights = np.exp(-dists / (2.0 * sigma))
            counts_arr = np.sum(kd_counts[inds] * weights[:, None], axis=0)
            if counts_cache is not None:
                counts_cache[key] = counts_arr
        log_probs = np.log(counts_arr.clip(min=1e-8))

    if label_bias is not None:
        log_probs += label_bias
        if not np.any(np.isfinite(log_probs)):
            log_probs = np.log(counts_arr.clip(min=1e-8))
    log_probs /= max(temperature, 1e-6)

    # Normalize
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()
    return probs


def _build_table(dataset, offset_di, offset_dj):
    """
    Build nonparametric lookup table from training data.
    Table values are compact count tuples. The KDTree fallback is built over
    unique observed neighborhoods, with a parallel class-count matrix for
    distance-weighted voting.
    """
    masks = dataset.get_augmented_masks()
    raw_table = defaultdict(lambda: np.zeros(NUM_LABELS, dtype=np.float64))
    n_entries = 0

    for m in masks:
        h, w = m.shape
        for i in range(h):
            for j in range(w):
                nbr = _extract_nbr(m, i, j, offset_di, offset_dj, h, w)
                raw_table[nbr.tobytes()][m[i, j]] += 1.0
                n_entries += 1

    keys = list(raw_table.keys())
    table = {key: tuple(counts.tolist()) for key, counts in raw_table.items()}
    kd_neighborhoods = np.vstack([
        np.frombuffer(key, dtype=np.int8) for key in keys
    ]).astype(np.float64)
    kd_counts = np.vstack([raw_table[key] for key in keys]).astype(np.float64)
    kdtree = KDTree(kd_neighborhoods, leaf_size=40)

    print(f"  {len(table)} unique patterns, {n_entries} total entries")
    return table, kdtree, kd_counts
