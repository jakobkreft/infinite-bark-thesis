"""
Evaluation metrics for synthesized masks:
- Class frequency error
- GLCM texture statistics
- Connected component statistics
- Tileability check (toroidal seam continuity)
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage


NUM_LABELS = 3
LABEL_NAMES = {0: "ozadje", 1: "slepice", 2: "mehanske_poskodbe"}


def class_frequency_error(generated, reference_freqs):
    """
    Compute per-class absolute frequency error between generated mask
    and reference frequencies.

    Args:
        generated: (H, W) int array
        reference_freqs: (NUM_LABELS,) array of target frequencies

    Returns:
        dict with per-class errors and total L1 error
    """
    total = generated.size
    gen_freqs = np.array([np.sum(generated == c) / total for c in range(NUM_LABELS)])
    errors = np.abs(gen_freqs - reference_freqs)
    result = {
        "generated_freqs": gen_freqs,
        "reference_freqs": reference_freqs,
        "per_class_error": errors,
        "total_l1_error": errors.sum(),
    }
    return result


def glcm_statistics(mask, distances=(1, 2, 4), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    """
    Compute GLCM-based texture statistics for a label mask.

    Returns dict with contrast, dissimilarity, homogeneity, energy, correlation
    averaged over distances and angles.
    """
    # graycomatrix expects uint8
    img = mask.astype(np.uint8)
    glcm = graycomatrix(
        img,
        distances=list(distances),
        angles=list(angles),
        levels=NUM_LABELS,
        symmetric=True,
        normed=True,
    )
    props = {}
    for prop_name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        values = graycoprops(glcm, prop_name)
        props[prop_name] = float(values.mean())
    return props


def connected_component_stats(mask):
    """
    Compute connected component statistics per class.

    Returns dict mapping label -> {count, mean_area, median_area, max_area, areas}
    """
    stats = {}
    for c in range(NUM_LABELS):
        binary = (mask == c).astype(np.int32)
        labeled, n_components = ndimage.label(binary)
        if n_components == 0:
            stats[LABEL_NAMES[c]] = {
                "count": 0,
                "mean_area": 0,
                "median_area": 0,
                "max_area": 0,
            }
            continue
        areas = ndimage.sum(binary, labeled, range(1, n_components + 1))
        areas = np.array(areas)
        stats[LABEL_NAMES[c]] = {
            "count": n_components,
            "mean_area": float(areas.mean()),
            "median_area": float(np.median(areas)),
            "max_area": float(areas.max()),
        }
    return stats


def tileability_score(mask):
    """
    Check toroidal tileability by measuring label agreement at seams
    when the mask is tiled 2x2.

    Returns fraction of matching pixels at horizontal and vertical seams.
    """
    h, w = mask.shape
    # Horizontal seam: last row vs first row
    h_match = np.mean(mask[-1, :] == mask[0, :])
    # Vertical seam: last col vs first col
    v_match = np.mean(mask[:, -1] == mask[:, 0])
    return {
        "horizontal_seam_match": float(h_match),
        "vertical_seam_match": float(v_match),
        "mean_seam_match": float((h_match + v_match) / 2),
    }


def evaluate_mask(generated, reference_freqs):
    """Run all evaluations and return a combined report."""
    report = {
        "class_frequency": class_frequency_error(generated, reference_freqs),
        "glcm": glcm_statistics(generated),
        "connected_components": connected_component_stats(generated),
        "tileability": tileability_score(generated),
    }
    return report


def compare_with_training(generated, dataset):
    """
    Compare generated mask statistics with aggregate training mask statistics.
    """
    ref_freqs = dataset.estimate_class_frequencies()

    # Aggregate training GLCM
    train_glcms = [glcm_statistics(m) for m in dataset.masks]
    avg_train_glcm = {}
    for prop in train_glcms[0]:
        avg_train_glcm[prop] = np.mean([g[prop] for g in train_glcms])

    # Aggregate training component stats
    train_cc = [connected_component_stats(m) for m in dataset.masks]

    gen_report = evaluate_mask(generated, ref_freqs)
    gen_report["training_glcm"] = avg_train_glcm

    return gen_report


def print_report(report):
    """Pretty-print an evaluation report."""
    print("\n=== Evaluation Report ===\n")

    cf = report["class_frequency"]
    print("Class Frequencies:")
    for c in range(NUM_LABELS):
        name = LABEL_NAMES[c]
        print(f"  {name}: generated={cf['generated_freqs'][c]:.4f}, "
              f"reference={cf['reference_freqs'][c]:.4f}, "
              f"error={cf['per_class_error'][c]:.4f}")
    print(f"  Total L1 error: {cf['total_l1_error']:.4f}")

    print("\nGLCM Statistics:")
    for prop, val in report["glcm"].items():
        print(f"  {prop}: {val:.4f}")

    if "training_glcm" in report:
        print("\nTraining GLCM (avg):")
        for prop, val in report["training_glcm"].items():
            print(f"  {prop}: {val:.4f}")

    print("\nConnected Components:")
    for label_name, stats in report["connected_components"].items():
        print(f"  {label_name}: count={stats['count']}, "
              f"mean_area={stats['mean_area']:.1f}, "
              f"max_area={stats['max_area']:.0f}")

    print("\nTileability:")
    tile = report["tileability"]
    print(f"  Horizontal seam: {tile['horizontal_seam_match']:.4f}")
    print(f"  Vertical seam: {tile['vertical_seam_match']:.4f}")
    print(f"  Mean seam match: {tile['mean_seam_match']:.4f}")
    print()
