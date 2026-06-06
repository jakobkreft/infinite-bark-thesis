"""viz.py – Research-quality matplotlib visualisations.

Rules:
- NO figure or axis titles  (they are added in LaTeX).
- Axis labels (xlabel, ylabel) are always present where meaningful.
- tight_layout() / constrained_layout to prevent label overlap.
- Consistent colour palette matching the mask classes.
- All figures saved as PNG at 150 dpi.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

# Class 0 = background (dark), 1 = slepice (amber), 2 = mehanske (blue)
PALETTE_RGB = {
    0: (0.086, 0.094, 0.114),
    1: (0.922, 0.655, 0.204),
    2: (0.263, 0.557, 0.969),
}
PALETTE_HEX = {k: "#{:02x}{:02x}{:02x}".format(*[int(v * 255) for v in rgb])
               for k, rgb in PALETTE_RGB.items()}

_CMAP_COLORS = [PALETTE_HEX[0], PALETTE_HEX[1], PALETTE_HEX[2]]
MASK_CMAP   = ListedColormap(_CMAP_COLORS)
MASK_NORM   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], MASK_CMAP.N)

CLASS_LABELS = {0: "background", 1: "slepice", 2: "mehanske"}

DPI = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _mask_legend() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=PALETTE_HEX[c], label=f"{c} – {CLASS_LABELS[c]}")
        for c in (0, 1, 2)
    ]


# ---------------------------------------------------------------------------
# Phase 1 – dataset gallery
# ---------------------------------------------------------------------------

def save_dataset_gallery(
    masks: list[np.ndarray],
    names: list[str],
    path: Path,
    cols: int = 10,
    max_items: int = 80,
    cell_size: float = 1.2,
) -> None:
    """Grid of all (up to max_items) dataset masks."""
    items = list(zip(masks[:max_items], names[:max_items]))
    n = len(items)
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * cell_size, rows * cell_size),
        squeeze=False,
    )
    for i, (mask, name) in enumerate(items):
        ax = axes[i // cols][i % cols]
        ax.imshow(mask, cmap=MASK_CMAP, norm=MASK_NORM,
                  interpolation="nearest", aspect="equal")
        ax.set_xlabel(name, fontsize=5, labelpad=1)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout(pad=0.3)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Phase 2 – statistics visualisations
# ---------------------------------------------------------------------------

def save_class_bar(
    class_counts: dict[int, int | float],
    path: Path,
    ylabel: str = "pixel count",
    log_scale: bool = False,
) -> None:
    """Vertical bar chart of per-class values."""
    classes = sorted(class_counts)
    vals = [class_counts[c] for c in classes]
    colors = [PALETTE_HEX[c] for c in classes]
    labels = [f"{c}\n{CLASS_LABELS[c]}" for c in classes]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="0.3", linewidth=0.7)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("class")
    if log_scale:
        ax.set_yscale("log")
    ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
    ax.margins(y=0.18)
    fig.tight_layout()
    _save(fig, path)


def save_fraction_boxplot(
    class_fraction_per_image: dict[int, list[float]],
    path: Path,
) -> None:
    """Box plot of per-image class fractions."""
    classes = sorted(class_fraction_per_image)
    data = [class_fraction_per_image[c] for c in classes]
    labels = [f"{c}\n{CLASS_LABELS[c]}" for c in classes]
    colors = [PALETTE_HEX[c] for c in classes]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bp = ax.boxplot(
        data, patch_artist=True, labels=labels,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("fraction of pixels")
    ax.set_xlabel("class")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, path)


def save_adjacency_heatmap(
    matrix: np.ndarray,
    path: Path,
    normalize_rows: bool = True,
) -> None:
    """Heatmap of the 3×3 class adjacency / transition matrix."""
    mat = matrix.astype(np.float64)
    if normalize_rows:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    labels = [f"{c}\n{CLASS_LABELS[c]}" for c in (0, 1, 2)]

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(mat, cmap="YlOrBr", vmin=0.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("neighbour class")
    ax.set_ylabel("centre class")

    for r in range(3):
        for c in range(3):
            v = float(mat[r, c])
            txt = f"{v:.2f}" if normalize_rows else f"{int(round(v))}"
            ax.text(c, r, txt, ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 else "black")

    fig.tight_layout()
    _save(fig, path)


def save_adjacency_comparison(
    real_matrix: np.ndarray,
    gen_matrix: np.ndarray,
    path: Path,
) -> None:
    """Side-by-side normalised adjacency heatmaps: real vs generated."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    labels = [f"{c}\n{CLASS_LABELS[c]}" for c in (0, 1, 2)]

    for ax, mat_raw, name in zip(axes, [real_matrix, gen_matrix], ["real", "generated"]):
        mat = mat_raw.astype(np.float64)
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

        im = ax.imshow(mat, cmap="YlOrBr", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("neighbour class")
        ax.set_ylabel("centre class")

        for r in range(3):
            for c in range(3):
                v = float(mat[r, c])
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.6 else "black")

        # small annotation for which panel
        ax.text(0.02, 0.98, name, transform=ax.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    fig.tight_layout(pad=1.0)
    _save(fig, path)


def save_component_histogram(
    sizes: list[int | float],
    path: Path,
    xlabel: str = "component size (pixels)",
    bins: int = 30,
    color: str = "#438ef7",
    log_x: bool = True,
) -> None:
    """Histogram of connected-component sizes with mean/median lines."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if not sizes:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        fig.tight_layout(); _save(fig, path); return

    arr = np.array(sizes, dtype=np.float64)
    plot_arr = arr
    xlabel_plot = xlabel
    if log_x:
        plot_arr = np.log10(np.clip(arr, 1, None))
        xlabel_plot = f"log₁₀({xlabel})"

    ax.hist(plot_arr, bins=bins, color=color, edgecolor="0.3", linewidth=0.5)

    # Add mean and median lines
    mn  = np.mean(plot_arr)
    med = np.median(plot_arr)
    ax.axvline(mn,  color="0.15", linestyle="--", linewidth=1.0,
               label=f"mean {10**mn:.1f}" if log_x else f"mean {mn:.1f}")
    ax.axvline(med, color="0.15", linestyle=":",  linewidth=1.0,
               label=f"median {10**med:.1f}" if log_x else f"median {med:.1f}")
    ax.legend(fontsize=7, loc="upper right")

    ax.set_xlabel(xlabel_plot)
    ax.set_ylabel("count")
    fig.tight_layout()
    _save(fig, path)


def save_cooccurrence_bar(
    cooccurrence: dict[str, int],
    path: Path,
) -> None:
    """Bar chart of the 4 class-1 / class-2 co-occurrence categories."""
    labels = list(cooccurrence.keys())
    vals   = list(cooccurrence.values())

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(labels, vals, color=["0.55", PALETTE_HEX[1], PALETTE_HEX[2], "#8050c8"],
           edgecolor="0.3", linewidth=0.7)
    ax.set_ylabel("number of images")
    ax.set_xlabel("class co-occurrence category")
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    _save(fig, path)


def save_size_distribution_bar(
    size_dist: dict[str, int],
    path: Path,
) -> None:
    """Bar chart of image-size distribution."""
    labels = sorted(size_dist.keys())
    vals   = [size_dist[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.5 + 1), 3.5))
    ax.bar(labels, vals, color="0.5", edgecolor="0.3", linewidth=0.7)
    ax.set_ylabel("number of images")
    ax.set_xlabel("image size (H×W)")
    if len(labels) > 5:
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Phase 3 – pattern library visualisations
# ---------------------------------------------------------------------------

def save_pattern_gallery(
    patterns: np.ndarray,    # [P, n, n]
    weights: np.ndarray,     # [P]
    path: Path,
    cols: int = 16,
    max_items: int = 128,
    cell_size: float = 0.55,
) -> None:
    """Gallery of patterns sorted by descending frequency."""
    order = np.argsort(weights)[::-1][:max_items]
    chosen = patterns[order]
    w_chosen = weights[order]
    n = len(chosen)
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * cell_size, rows * (cell_size + 0.35)),
                              squeeze=False)
    for i in range(n):
        ax = axes[i // cols][i % cols]
        ax.imshow(chosen[i], cmap=MASK_CMAP, norm=MASK_NORM,
                  interpolation="nearest", aspect="equal")
        ax.set_xlabel(f"{w_chosen[i]:.0f}", fontsize=5, labelpad=1)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout(pad=0.3)
    _save(fig, path)


def save_pattern_weight_histogram(
    weights: np.ndarray,
    path: Path,
    bins: int = 30,
) -> None:
    """Log-scale histogram of pattern frequencies."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(np.log10(np.clip(weights, 1e-6, None)), bins=bins,
            color="#6050d0", edgecolor="0.3", linewidth=0.5)
    ax.set_xlabel("log₁₀(weight)")
    ax.set_ylabel("number of patterns")
    fig.tight_layout()
    _save(fig, path)


def save_compat_density_bar(
    compat: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Bar chart of compatibility-matrix density per direction."""
    dirs = ["up", "down", "left", "right"]
    vals = [float(compat[d].mean()) for d in dirs]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(dirs, vals, color="#4090d8", edgecolor="0.3", linewidth=0.7)
    ax.set_ylabel("fraction of compatible pairs")
    ax.set_xlabel("direction")
    ax.set_ylim(0, min(1.0, max(vals) * 1.3))
    fig.tight_layout()
    _save(fig, path)


def save_compat_heatmap(
    compat_matrix: np.ndarray,    # [P, P] bool
    path: Path,
    direction: str = "",
) -> None:
    """Visualise the full P×P compatibility matrix as an image."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(compat_matrix.astype(np.float32), cmap="Blues",
              aspect="auto", interpolation="none", vmin=0, vmax=1)
    ax.set_xlabel(f"pattern index (neighbour){' — ' + direction if direction else ''}")
    ax.set_ylabel("pattern index (anchor)")
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Phase 3.5 – pattern reweighting visualisations
# ---------------------------------------------------------------------------

def save_pattern_reweight_scatter(
    original_weights: np.ndarray,
    reweighted_weights: np.ndarray,
    class_fracs: np.ndarray,   # [P, n_classes]
    path: Path,
) -> None:
    """Log-log scatter of original vs reweighted pattern weights.

    Each point is a pattern; colour encodes its dominant class.
    Points above the diagonal are upweighted (rare class content);
    points below are downweighted (background-dominant).
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.5))

    dominant_class = np.argmax(class_fracs, axis=1)

    for c in (0, 1, 2):
        mask = dominant_class == c
        if not mask.any():
            continue
        ax.scatter(
            original_weights[mask], reweighted_weights[mask],
            c=PALETTE_HEX[c], alpha=0.45, s=10,
            label=f"{c} – {CLASS_LABELS[c]}",
            linewidths=0,
        )

    # Diagonal: no-change reference
    all_vals = np.concatenate([original_weights, reweighted_weights])
    vmin, vmax = max(all_vals.min(), 1e-3), all_vals.max() * 1.1
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.4,
            label="no change")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("original weight")
    ax.set_ylabel("reweighted weight")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    _save(fig, path)


def save_pattern_class_fracs_comparison(
    corpus_fracs: np.ndarray,       # [n_classes] before reweighting
    reweighted_fracs: np.ndarray,   # [n_classes] after reweighting
    target_fracs: np.ndarray,       # [n_classes] desired
    path: Path,
) -> None:
    """Grouped bar: expected class fracs from library (before/after/target).

    Shows how reweighting shifts the library's class composition toward
    the real-data target.
    """
    n_cls = len(target_fracs)
    x = np.arange(n_cls)
    w = 0.22
    labels = [f"{c}\n{CLASS_LABELS[c]}" for c in range(n_cls)]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(x - w,     corpus_fracs,     w, label="corpus (raw)",
           color="0.50", edgecolor="0.2")
    ax.bar(x,         reweighted_fracs, w, label="reweighted",
           color=[PALETTE_HEX[c] for c in range(n_cls)], edgecolor="0.2")
    ax.bar(x + w,     target_fracs,     w, label="target (real data)",
           color=[PALETTE_HEX[c] for c in range(n_cls)], edgecolor="0.5",
           alpha=0.45, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("expected class fraction from patterns")
    ax.set_xlabel("class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Phase 4 – WFC generation snapshots
# ---------------------------------------------------------------------------

def save_wfc_snapshot(
    entropy_map: np.ndarray,    # (ph, pw) float – -1 for contradictions
    preview_mask: np.ndarray,   # (H, W) uint8
    collapsed_fraction: float,
    step: int,
    restart: int,
    path: Path,
) -> None:
    """Two-panel figure: Shannon entropy heatmap (left) + partial reconstruction (right)."""
    fig, (ax_ent, ax_prev) = plt.subplots(1, 2, figsize=(9, 4),
                                           gridspec_kw={"wspace": 0.35})

    # --- Entropy panel ---
    ent = entropy_map.copy().astype(np.float64)
    contradiction_mask = ent < 0
    ent[contradiction_mask] = np.nan

    im = ax_ent.imshow(ent, cmap="plasma", interpolation="nearest",
                       aspect="equal", vmin=0)
    if contradiction_mask.any():
        contra_overlay = np.zeros((*entropy_map.shape, 4), dtype=np.float32)
        contra_overlay[contradiction_mask] = [1, 0, 0, 0.85]
        ax_ent.imshow(contra_overlay, interpolation="nearest", aspect="equal")
    fig.colorbar(im, ax=ax_ent, fraction=0.046, pad=0.04, label="entropy")
    ax_ent.set_xlabel("x (pattern anchor)")
    ax_ent.set_ylabel("y (pattern anchor)")
    ax_ent.text(0.02, 0.98,
                f"step {step}  ·  restart {restart}  ·  collapsed {collapsed_fraction:.1%}",
                transform=ax_ent.transAxes, fontsize=7,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # --- Preview panel ---
    ax_prev.imshow(preview_mask, cmap=MASK_CMAP, norm=MASK_NORM,
                   interpolation="nearest", aspect="equal")
    ax_prev.set_xlabel("x (pixels)")
    ax_prev.set_ylabel("y (pixels)")
    ax_prev.legend(handles=_mask_legend(), loc="lower right",
                   fontsize=6, framealpha=0.7)

    _save(fig, path)


def save_wfc_progress_curve(
    steps: list[int],
    fractions: list[float],
    path: Path,
) -> None:
    """Line plot of collapsed-cell fraction over WFC steps."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(steps, fractions, color="#2060c8", linewidth=1.2)
    ax.set_xlabel("collapse step")
    ax.set_ylabel("fraction collapsed")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, path)


def save_generated_gallery(
    masks: list[np.ndarray],
    names: list[str],
    path: Path,
    cols: int = 4,
    cell_size: float = 2.0,
) -> None:
    """Gallery of successfully generated masks."""
    n = len(masks)
    if n == 0:
        return
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * cell_size, rows * cell_size),
                              squeeze=False)
    for i, (mask, name) in enumerate(zip(masks, names)):
        ax = axes[i // cols][i % cols]
        ax.imshow(mask, cmap=MASK_CMAP, norm=MASK_NORM,
                  interpolation="nearest", aspect="equal")
        ax.set_xlabel(name, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout(pad=0.5)
    _save(fig, path)


def save_single_mask(mask: np.ndarray, path: Path, pixel_size: int = 6) -> None:
    """Save a single mask as a large, clearly coloured PNG."""
    h, w = mask.shape
    fig, ax = plt.subplots(figsize=(w * pixel_size / 72, h * pixel_size / 72))
    ax.imshow(mask, cmap=MASK_CMAP, norm=MASK_NORM,
              interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(handles=_mask_legend(), loc="lower right",
              fontsize=7, framealpha=0.75)
    fig.tight_layout(pad=0.1)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Phase 5 – comparison visualisations
# ---------------------------------------------------------------------------

def save_real_vs_generated_bar(
    real_fracs: dict[int, float],
    gen_fracs: dict[int, float],
    path: Path,
) -> None:
    """Grouped bar chart: real vs generated class fractions."""
    classes = sorted(real_fracs)
    x = np.arange(len(classes))
    w = 0.35
    r_vals = [real_fracs[c] for c in classes]
    g_vals = [gen_fracs[c]  for c in classes]

    fig, ax = plt.subplots(figsize=(5, 3.8))
    b1 = ax.bar(x - w / 2, r_vals, w, label="real",
                color="0.45", edgecolor="0.2")
    b2 = ax.bar(x + w / 2, g_vals, w, label="generated",
                color=[PALETTE_HEX[c] for c in classes], edgecolor="0.2")

    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{CLASS_LABELS[c]}" for c in classes], fontsize=9)
    ax.set_ylabel("fraction of pixels")
    ax.set_xlabel("class")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(r_vals), max(g_vals)) * 1.30)
    fig.tight_layout()
    _save(fig, path)


def save_comparison_component_hist(
    real_sizes: list[int],
    gen_sizes:  list[int],
    path: Path,
    bins: int = 30,
    cls: int = 1,
) -> None:
    """Overlapping histograms of real vs generated component sizes with stats."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    def _log(s): return np.log10(np.array(s, dtype=np.float64).clip(1))

    legend_entries = []
    for sizes, label, color, alpha in [
        (real_sizes, "real", "0.45", 0.60),
        (gen_sizes,  "generated", PALETTE_HEX[cls], 0.55),
    ]:
        if not sizes:
            continue
        data = _log(sizes)
        ax.hist(data, bins=bins, alpha=alpha, color=color, label=label,
                edgecolor="0.3", linewidth=0.4)
        mn, med = float(np.mean(data)), float(np.median(data))
        ax.axvline(mn,  color=color, linestyle="--", linewidth=1.0,
                   alpha=0.9, label=f"{label} mean={10**mn:.1f}px")
        ax.axvline(med, color=color, linestyle=":",  linewidth=1.0,
                   alpha=0.9, label=f"{label} median={10**med:.1f}px")

    ax.set_xlabel("log₁₀(component size in pixels)")
    ax.set_ylabel("count")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Guidance visualisations
# ---------------------------------------------------------------------------

def save_guidance_curve(
    guidance_history: list[tuple[int, np.ndarray]],
    target_fracs: np.ndarray,
    path: Path,
) -> None:
    """Running class fracs vs target during WFC generation (line plot).

    Solid lines = running distribution; dashed lines = target.
    """
    if not guidance_history:
        return

    steps  = np.array([h[0] for h in guidance_history])
    fracs  = np.stack([h[1] for h in guidance_history], axis=0)  # [T, C]
    n_cls  = fracs.shape[1]

    fig, ax = plt.subplots(figsize=(6, 3.8))

    for c in range(n_cls):
        color = PALETTE_HEX.get(c, f"C{c}")
        ax.plot(steps, fracs[:, c], color=color, linewidth=1.3,
                label=f"c{c} – {CLASS_LABELS.get(c, c)}")
        ax.axhline(target_fracs[c], color=color, linewidth=0.9,
                   linestyle="--", alpha=0.65)

    ax.set_xlabel("collapse step")
    ax.set_ylabel("running class fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    _save(fig, path)


def save_guidance_comparison_bar(
    runs: list[dict],
    target_fracs: np.ndarray,
    path: Path,
) -> None:
    """Grouped bar: target vs final class fractions for each run."""
    if not runs:
        return
    n_cls  = len(target_fracs)
    n_runs = len(runs)
    x      = np.arange(n_cls)
    bar_w  = 0.7 / (n_runs + 1)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    ax.bar(x - bar_w * n_runs / 2, target_fracs, bar_w,
           color="0.35", label="target", edgecolor="0.2", zorder=3)

    for i, run in enumerate(runs):
        offset = bar_w * (i - n_runs / 2 + 1)
        colors = [PALETTE_HEX.get(c, f"C{c}") for c in range(n_cls)]
        ax.bar(x + offset, run["final_fracs"], bar_w,
               color=colors, label=run["name"], edgecolor="0.2",
               alpha=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n{CLASS_LABELS.get(c, c)}" for c in range(n_cls)], fontsize=9
    )
    ax.set_ylabel("fraction of pixels")
    ax.set_xlabel("class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
