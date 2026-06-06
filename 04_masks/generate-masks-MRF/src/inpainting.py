"""
MAP inpainting for hole filling using graph-cut alpha-expansion (gco-wrapper),
with optional Gibbs refinement on the inpainted region.
"""

import numpy as np
from tqdm import tqdm

from .mrf_model import MRFModel, NUM_LABELS
from .dataset import _full_neighborhood_offsets
from .synthesis import _cond_dist, _build_table, _extract_nbr


def inpaint(
    mask,
    dataset,
    unknown_value=-1,
    pairwise_scale=1.0,
    target_ratio=None,
    lambda_ratio=1.0,
    toroidal=True,
    n_refine=2,
    radius=3,
    temperature=1.0,
    k_fallback=11,
):
    """
    Inpaint unknown regions in a label mask using graph-cut MAP inference,
    optionally followed by Gibbs refinement.

    Args:
        mask: (H, W) int array with values in {0,1,2} for known pixels
              and unknown_value (-1) for pixels to inpaint
        dataset: BarkMaskDataset for estimating potentials and Gibbs refinement

    Returns:
        result: (H, W) int8 array, fully filled
    """
    h, w = mask.shape
    mask = mask.copy().astype(np.int8)

    class_freqs = dataset.estimate_class_frequencies()
    pairwise = dataset.estimate_pairwise_potentials()

    model = MRFModel(
        h, w, class_freqs, pairwise,
        target_ratio=target_ratio,
        lambda_ratio=lambda_ratio,
        toroidal=toroidal,
    )

    fixed_labels = mask.copy()

    print("Running graph-cut MAP inference for inpainting...")
    result = model.map_inference(
        fixed_labels=fixed_labels,
        pairwise_scale=pairwise_scale,
    )
    result = result.astype(np.int8)

    # Restore known pixels (in case graph cut altered them slightly)
    known_mask = mask != unknown_value
    result[known_mask] = mask[known_mask]

    if n_refine > 0:
        print("Gibbs refinement on inpainted region...")
        full_offset_list = _full_neighborhood_offsets(radius)
        full_di = np.array([o[0] for o in full_offset_list], dtype=np.int32)
        full_dj = np.array([o[1] for o in full_offset_list], dtype=np.int32)
        full_table, full_kdtree, full_kd_labels = _build_table(
            dataset, full_di, full_dj
        )

        proportion_bias = np.zeros(NUM_LABELS, dtype=np.float64)
        if target_ratio is not None:
            target = np.clip(np.array(target_ratio, dtype=np.float64), 1e-8, 1.0)
            proportion_bias = lambda_ratio * np.log(
                target / np.clip(class_freqs, 1e-8, 1.0)
            )

        unknown_indices = list(zip(*np.where(~known_mask)))
        rand_vals = np.random.random(len(unknown_indices) * n_refine)
        rv_idx = 0
        for sweep in range(n_refine):
            np.random.shuffle(unknown_indices)
            for i, j in tqdm(
                unknown_indices,
                desc=f"Inpaint Gibbs {sweep + 1}/{n_refine}",
                leave=False,
            ):
                nbr = _extract_nbr(result, i, j, full_di, full_dj, h, w)
                probs = _cond_dist(
                    nbr, full_table, full_kdtree, full_kd_labels,
                    proportion_bias, temperature, k_fallback,
                )
                cumprobs = np.cumsum(probs)
                result[i, j] = np.searchsorted(cumprobs, rand_vals[rv_idx])
                rv_idx += 1

    return result
