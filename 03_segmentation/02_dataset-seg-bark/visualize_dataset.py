"""
Visualize dataset_step1: publication-quality figures for segmentation dataset.

Generates:
  1. dataset_overview.png    - Grid of sample images
  2. mask_overview.png       - Grid of color-coded masks
  3. overlay_overview.png    - Grid of images with mask overlay
  4. class_distribution.png  - Class pixel distribution charts
  5. resolution_distribution.png - Image dimension scatter plot

All figures saved at 300 DPI to dataset_step1/visualizations/.
"""

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# --- Configuration ---
DATASET_DIR = Path("dataset_step1")
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"
VIS_DIR = DATASET_DIR / "visualizations"

# Class definitions (matching labeling tool)
CLASSES = {
    0: {"name": "Background", "color": (0, 0, 0)},
    1: {"name": "Slepice / odrezane veje", "color": (0, 255, 255)},       # Cyan
    2: {"name": "Mehanske poškodbe / odluščeno", "color": (255, 128, 0)},  # Orange
}

# Normalized colors for matplotlib
CLASS_COLORS_NORM = {
    0: (0.0, 0.0, 0.0),
    1: (0.0, 1.0, 1.0),
    2: (1.0, 0.5, 0.0),
}

N_SAMPLES = 12  # Number of samples for grid visualizations
GRID_ROWS = 3
GRID_COLS = 4
OVERLAY_ALPHA = 0.4
DPI = 300

# Matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 100,
})


def get_paired_files():
    """Return sorted list of (stem, image_path, mask_path) tuples."""
    pairs = []
    for mask_path in sorted(MASKS_DIR.glob("*.png")):
        stem = mask_path.stem
        img_path = IMAGES_DIR / f"{stem}.jpg"
        if img_path.exists():
            pairs.append((stem, img_path, mask_path))
    return pairs


def load_image_thumbnail(path, max_size=800):
    """Load image and resize for display (keeps aspect ratio)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return np.array(img)


def colorize_mask(mask_arr):
    """Convert grayscale mask array to RGB color-coded array."""
    h, w = mask_arr.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, cls_info in CLASSES.items():
        colored[mask_arr == cls_id] = cls_info["color"]
    return colored


def create_overlay(img_arr, mask_arr, alpha=OVERLAY_ALPHA):
    """Blend color-coded mask over image."""
    colored_mask = colorize_mask(mask_arr).astype(np.float32)
    img_float = img_arr.astype(np.float32)

    # Only blend where mask is non-background
    foreground = mask_arr > 0
    result = img_float.copy()
    for c in range(3):
        result[:, :, c][foreground] = (
            (1 - alpha) * img_float[:, :, c][foreground]
            + alpha * colored_mask[:, :, c][foreground]
        )
    return result.astype(np.uint8)


def make_class_legend():
    """Create legend patches for class colors."""
    patches = []
    for cls_id in sorted(CLASSES.keys()):
        color = CLASS_COLORS_NORM[cls_id]
        label = f"Class {cls_id}: {CLASSES[cls_id]['name']}"
        patches.append(mpatches.Patch(facecolor=color, edgecolor="gray", label=label))
    return patches


def select_samples(pairs, n=N_SAMPLES, seed=42):
    """Select n representative samples (deterministic)."""
    rng = random.Random(seed)
    if len(pairs) <= n:
        return pairs
    return rng.sample(pairs, n)


# === Visualization Functions ===


def plot_image_grid(samples, output_path):
    """1. Dataset overview: grid of sample images."""
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(16, 12))
    fig.suptitle("Dataset Overview — Sample Images", fontweight="bold", y=0.98)

    for idx, ax in enumerate(axes.flat):
        if idx < len(samples):
            stem, img_path, _ = samples[idx]
            img = load_image_thumbnail(img_path)
            ax.imshow(img)
            ax.set_title(stem, fontsize=8)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_mask_grid(samples, output_path):
    """2. Mask overview: grid of color-coded masks."""
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(16, 12))
    fig.suptitle("Mask Overview — Class Labels", fontweight="bold", y=0.98)

    for idx, ax in enumerate(axes.flat):
        if idx < len(samples):
            stem, _, mask_path = samples[idx]
            mask = np.array(Image.open(mask_path).convert("L"))
            colored = colorize_mask(mask)
            ax.imshow(colored)
            ax.set_title(stem, fontsize=8)
        ax.axis("off")

    legend_patches = make_class_legend()
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_overlay_grid(samples, output_path):
    """3. Overlay overview: masks blended over images."""
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(16, 12))
    fig.suptitle("Overlay — Masks on Images (α=0.4)", fontweight="bold", y=0.98)

    for idx, ax in enumerate(axes.flat):
        if idx < len(samples):
            stem, img_path, mask_path = samples[idx]
            img = load_image_thumbnail(img_path)
            mask_full = np.array(Image.open(mask_path).convert("L"))
            # Resize mask to match thumbnail
            mask_resized = np.array(
                Image.fromarray(mask_full).resize(
                    (img.shape[1], img.shape[0]), Image.NEAREST
                )
            )
            overlay = create_overlay(img, mask_resized)
            ax.imshow(overlay)
            ax.set_title(stem, fontsize=8)
        ax.axis("off")

    legend_patches = make_class_legend()
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_class_distribution(pairs, output_path):
    """4. Class distribution: overall and per-image breakdown."""
    # Collect per-image class counts
    stems = []
    counts_per_image = {c: [] for c in CLASSES}

    print("  Computing class distribution across all images...")
    for stem, _, mask_path in pairs:
        mask = np.array(Image.open(mask_path).convert("L"))
        total = mask.size
        for c in CLASSES:
            count = int(np.sum(mask == c))
            counts_per_image[c].append(count / total * 100)  # percentage
        stems.append(stem)

    # Overall totals
    totals = {c: sum(counts_per_image[c]) for c in CLASSES}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 2.5]})
    fig.suptitle("Class Distribution", fontweight="bold", y=1.02)

    # Left: overall pie chart
    labels = [f"Class {c}\n{CLASSES[c]['name']}" for c in sorted(CLASSES.keys())]
    sizes = [totals[c] for c in sorted(CLASSES.keys())]
    colors = [CLASS_COLORS_NORM[c] for c in sorted(CLASSES.keys())]
    # Adjust background color for visibility in pie
    pie_colors = [(0.3, 0.3, 0.3), colors[1], colors[2]]

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 9}
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax1.set_title("Overall Class Proportions")

    # Right: stacked bar chart (per image)
    x = np.arange(len(stems))
    bottom = np.zeros(len(stems))
    bar_colors = [(0.3, 0.3, 0.3), CLASS_COLORS_NORM[1], CLASS_COLORS_NORM[2]]

    for c in sorted(CLASSES.keys()):
        vals = np.array(counts_per_image[c])
        ax2.bar(x, vals, bottom=bottom, color=bar_colors[c],
                label=f"Class {c}: {CLASSES[c]['name']}", width=1.0, edgecolor="none")
        bottom += vals

    ax2.set_xlabel("Image Index")
    ax2.set_ylabel("Percentage of Pixels (%)")
    ax2.set_title("Per-Image Class Distribution")
    ax2.set_xlim(-0.5, len(stems) - 0.5)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_resolution_distribution(pairs, output_path):
    """5. Resolution distribution: scatter plot of image dimensions."""
    widths = []
    heights = []
    stems = []

    for stem, img_path, _ in pairs:
        img = Image.open(img_path)
        w, h = img.size
        widths.append(w)
        heights.append(h)
        stems.append(stem)
        img.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(widths, heights, c="steelblue", alpha=0.6, edgecolors="navy",
                         s=40, linewidths=0.5)

    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.set_title("Image Resolution Distribution", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add summary stats
    stats_text = (
        f"N = {len(widths)}\n"
        f"Width:  {min(widths)}–{max(widths)} px\n"
        f"Height: {min(heights)}–{max(heights)} px\n"
        f"Mean:   {np.mean(widths):.0f} × {np.mean(heights):.0f} px"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5",
            facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = get_paired_files()
    print(f"Found {len(pairs)} image-mask pairs in {DATASET_DIR}")

    if not pairs:
        print("No pairs found. Run prepare_dataset.py first.")
        return

    samples = select_samples(pairs)
    print(f"Selected {len(samples)} samples for grid visualizations")

    print("\nGenerating visualizations:")
    plot_image_grid(samples, VIS_DIR / "dataset_overview.png")
    plot_mask_grid(samples, VIS_DIR / "mask_overview.png")
    plot_overlay_grid(samples, VIS_DIR / "overlay_overview.png")
    plot_class_distribution(pairs, VIS_DIR / "class_distribution.png")
    plot_resolution_distribution(pairs, VIS_DIR / "resolution_distribution.png")

    print(f"\nAll visualizations saved to {VIS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
