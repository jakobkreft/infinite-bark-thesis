"""
Nonparametric multiscale Gibbs inpainting.

Unknown pixels are randomly initialized and then synthesized with exactly
the same coarse-to-fine + Gibbs-refinement algorithm `synthesize_mask` uses
to build a mask from nothing (see synthesis.py) -- the only difference is
that every pass is restricted to the unknown region via `update_mask`, so
known pixels are read-only context (fixed boundary conditions) that are
never themselves resampled. This is what makes the fill look like genuine
random bark texture instead of a flat MAP estimate: it's a real sample from
the same nonparametric texture model `synthesize` draws from, just
constrained by whatever's already known around the hole.
"""

import numpy as np

from .dataset import NUM_LABELS
from .synthesis import (
    _RatioController,
    _gibbs_sweep,
    _nn_downsample_labels,
    _nn_upsample_labels,
    _normalize_target_ratio,
    _raster_pass,
)


def inpaint(
    mask,
    trained_model,
    unknown_value=-1,
    target_ratio=None,
    lambda_ratio=1.0,
    tileable_v=True,
    tileable_h=True,
    multiscale=True,
    n_refine=3,
    temperature=1.0,
    k_fallback=11,
    seed=None,
    verbose=True,
):
    """
    Fill unknown regions in a label mask by nonparametric texture synthesis,
    constrained by the known pixels.

    Args:
        mask: (H, W) int array with values in {0,1,2} for known pixels and
            `unknown_value` for pixels to fill in.
        trained_model: a `model_cache.TrainedModel` with class frequencies
            and causal/full neighborhood tables from the training dataset.
        multiscale: fill a coarse (1/4 resolution) version of the hole
            first for large-scale structure, then refine at full
            resolution -- same as `synthesize_mask`. Strongly recommended
            for large holes, which otherwise tend to mix very slowly from a
            flat random initialization.
        n_refine: number of full Gibbs sweeps over the unknown region after
            the initial fill.

    Returns:
        result: (H, W) int8 array, fully filled. Known pixels are returned
        unchanged.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = mask.shape
    mask = mask.copy().astype(np.int8)
    known_mask = mask != unknown_value
    unknown_mask = ~known_mask
    n_unknown = int(unknown_mask.sum())

    print(f"Inpainting {n_unknown} unknown pixels "
          f"({'multiscale' if multiscale else 'single-scale'}, "
          f"{n_refine} refinement sweep(s))...")

    result = mask.copy()
    if n_unknown == 0:
        return result

    class_freqs = trained_model.class_freqs
    ratio_target = _normalize_target_ratio(target_ratio)
    if ratio_target is not None and lambda_ratio <= 0:
        ratio_target = None

    if multiscale:
        ch, cw = max(h // 4, 8), max(w // 4, 8)
        if verbose:
            print(f"Multiscale inpaint: filling coarse {ch}x{cw}...")
        coarse = _nn_downsample_labels(result, ch, cw)
        coarse_unknown = coarse == unknown_value
        _random_fill(coarse, coarse_unknown, class_freqs)
        coarse_ratio = _RatioController(coarse, ratio_target, lambda_ratio)
        _raster_pass(
            coarse, trained_model.causal_table, trained_model.causal_kdtree,
            trained_model.causal_kd_counts,
            trained_model.causal_di, trained_model.causal_dj,
            coarse_ratio, temperature, k_fallback,
            tileable_v, tileable_h, desc="Coarse inpaint raster", verbose=verbose,
            update_mask=coarse_unknown,
        )

        result = _nn_upsample_labels(coarse, h, w)
        result[known_mask] = mask[known_mask]

        if verbose:
            print(f"Multiscale inpaint: filling at full {h}x{w}...")
        fine_ratio = _RatioController(result, ratio_target, lambda_ratio)
        _raster_pass(
            result, trained_model.causal_table, trained_model.causal_kdtree,
            trained_model.causal_kd_counts,
            trained_model.causal_di, trained_model.causal_dj,
            fine_ratio, temperature, k_fallback,
            tileable_v, tileable_h, desc="Fine inpaint raster", verbose=verbose,
            update_mask=unknown_mask,
        )
    else:
        _random_fill(result, unknown_mask, class_freqs)

    refine_ratio = _RatioController(result, ratio_target, lambda_ratio)
    for sweep in range(n_refine):
        _gibbs_sweep(
            result, trained_model.full_table, trained_model.full_kdtree,
            trained_model.full_kd_counts,
            trained_model.full_di, trained_model.full_dj,
            refine_ratio, temperature, k_fallback,
            tileable_v, tileable_h,
            desc=f"Inpaint Gibbs refinement {sweep + 1}/{n_refine}", verbose=verbose,
            update_mask=unknown_mask,
        )

    return result


def _random_fill(canvas, update_mask, class_freqs):
    """Randomly initialize the positions marked True in update_mask."""
    n = int(update_mask.sum())
    if n:
        canvas[update_mask] = np.random.choice(
            NUM_LABELS, size=n, p=class_freqs
        ).astype(np.int8)
