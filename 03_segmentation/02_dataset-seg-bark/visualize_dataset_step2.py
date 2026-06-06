"""
Visualize dataset_step2: publication-quality figures for the train/val/test split.

Generates:
  1. split_samples.png           - Sample images with overlays per split
  2. class_distribution_splits.png - Class distribution comparison across splits
  3. split_composition.png       - Summary table figure

All figures saved at 300 DPI to dataset_step2/visualizations/.
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# --- Configuration ---
DATASET_DIR = Path("dataset_step2")
META_DIR = DATASET_DIR / "metadata"
VIS_DIR = DATASET_DIR / "visualizations"

CLASSES = {
    0: {"name": "Background", "color": (0, 0, 0)},
    1: {"name": "Slepice / odrezane veje", "color": (0, 255, 255)},
    2: {"name": "Mehanske poškodbe / odluščeno", "color": (255, 128, 0)},
}

CLASS_COLORS_NORM = {
    0: (0.3, 0.3, 0.3),  # dark gray for visibility
    1: (0.0, 1.0, 1.0),
    2: (1.0, 0.5, 0.0),
}

DPI = 300
OVERLAY_ALPHA = 0.4
SAMPLES_PER_SPLIT = 4

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 100,
})


def load_split_info():
    """Load split.json metadata."""
    with open(META_DIR / "split.json") as f:
        return json.load(f)


def load_thumbnail(path, max_size=800):
    """Load image as thumbnail for display."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return np.array(img)


def colorize_mask(mask_arr):
    """Convert grayscale mask to RGB color-coded."""
    h, w = mask_arr.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, info in CLASSES.items():
        colored[mask_arr == cls_id] = info["color"]
    return colored


def create_overlay(img_arr, mask_arr, alpha=OVERLAY_ALPHA):
    """Blend color-coded mask over image."""
    colored = colorize_mask(mask_arr).astype(np.float32)
    result = img_arr.astype(np.float32).copy()
    fg = mask_arr > 0
    for c in range(3):
        result[:, :, c][fg] = (1 - alpha) * result[:, :, c][fg] + alpha * colored[:, :, c][fg]
    return result.astype(np.uint8)


def make_legend():
    """Create legend patches."""
    return [
        mpatches.Patch(facecolor=CLASS_COLORS_NORM[c], edgecolor="gray",
                       label=f"Class {c}: {CLASSES[c]['name']}")
        for c in sorted(CLASSES.keys())
    ]


def plot_split_samples(split_info, output_path):
    """1. Sample images with mask overlays for each split (3 rows)."""
    splits = ["train", "val", "test"]
    rng = random.Random(42)

    fig, axes = plt.subplots(3, SAMPLES_PER_SPLIT, figsize=(16, 12))
    fig.suptitle("Dataset Split — Sample Images with Mask Overlays", fontweight="bold", y=0.98)

    for row, split_name in enumerate(splits):
        stems = split_info["splits"][split_name]
        selected = rng.sample(stems, min(SAMPLES_PER_SPLIT, len(stems)))

        for col in range(SAMPLES_PER_SPLIT):
            ax = axes[row, col]
            if col < len(selected):
                stem = selected[col]
                img_path = DATASET_DIR / split_name / "images" / f"{stem}.jpg"
                mask_path = DATASET_DIR / split_name / "masks" / f"{stem}.png"

                img = load_thumbnail(img_path)
                mask_full = np.array(Image.open(mask_path).convert("L"))
                mask_resized = np.array(
                    Image.fromarray(mask_full).resize(
                        (img.shape[1], img.shape[0]), Image.NEAREST
                    )
                )
                overlay = create_overlay(img, mask_resized)
                ax.imshow(overlay)
                ax.set_title(stem, fontsize=7)
            ax.axis("off")

            if col == 0:
                n = len(stems)
                ax.set_ylabel(f"{split_name.upper()}\n({n} images)",
                              fontsize=10, fontweight="bold", rotation=0,
                              labelpad=70, va="center")

    legend = make_legend()
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0.08, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_class_distribution_splits(split_info, output_path):
    """2. Class distribution comparison across splits."""
    splits = ["train", "val", "test"]
    dist = split_info["class_distribution_per_split"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("Class Distribution per Split", fontweight="bold", y=1.02)

    bar_colors = [CLASS_COLORS_NORM[0], CLASS_COLORS_NORM[1], CLASS_COLORS_NORM[2]]
    class_names = [f"C{c}: {CLASSES[c]['name'][:20]}" for c in sorted(CLASSES.keys())]

    for idx, split_name in enumerate(splits):
        ax = axes[idx]
        d = dist[split_name]
        values = [d[str(c)] for c in sorted(CLASSES.keys())]

        bars = ax.bar(class_names, values, color=bar_colors, edgecolor="gray", linewidth=0.5)
        ax.set_title(f"{split_name.upper()} ({split_info['image_counts'][split_name]} images)",
                     fontweight="bold")
        ax.set_ylim(0, 100)
        if idx == 0:
            ax.set_ylabel("Percentage of Pixels (%)")
        ax.tick_params(axis="x", rotation=30)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_split_composition(split_info, output_path):
    """3. Summary composition figure with key statistics."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    fig.suptitle("Dataset Step 2 — Split Composition Summary", fontweight="bold", fontsize=14)

    splits = ["train", "val", "test", "unlabeled"]
    headers = ["Split", "Images", "Class 0 (%)", "Class 1 (%)", "Class 2 (%)"]

    cell_text = []
    for split_name in splits:
        n = split_info["image_counts"][split_name]
        if split_name in split_info["class_distribution_per_split"]:
            d = split_info["class_distribution_per_split"][split_name]
            cell_text.append([
                split_name.upper(), str(n),
                f"{d['0']:.1f}", f"{d['1']:.1f}", f"{d['2']:.1f}"
            ])
        else:
            cell_text.append([split_name.upper(), str(n), "—", "—", "—"])

    # Total row
    total = sum(split_info["image_counts"][s] for s in splits)
    cell_text.append(["TOTAL", str(total), "", "", ""])

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style split rows
    split_colors = {
        "TRAIN": "#E2EFDA",
        "VAL": "#FCE4D6",
        "TEST": "#D9E2F3",
        "UNLABELED": "#F2F2F2",
        "TOTAL": "#D6DCE4",
    }
    for i, row_data in enumerate(cell_text):
        color = split_colors.get(row_data[0], "white")
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    # Add metadata text below
    meta_text = (
        f"Seed: {split_info['seed']} | "
        f"Stratification: {split_info['stratification']} | "
        f"Source: {split_info['source']}"
    )
    fig.text(0.5, 0.05, meta_text, ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    split_info = load_split_info()
    print(f"Loaded split info: {split_info['image_counts']}")

    print("\nGenerating visualizations:")
    plot_split_samples(split_info, VIS_DIR / "split_samples.png")
    plot_class_distribution_splits(split_info, VIS_DIR / "class_distribution_splits.png")
    plot_split_composition(split_info, VIS_DIR / "split_composition.png")

    print(f"\nAll visualizations saved to {VIS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
