"""
Dataset Analysis Script

Generates publication-ready figures for dataset exploration and documentation.

Usage:
    conda activate bark-seg
    python analyze_dataset.py --config configs/default.yaml
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm

from src.utils import load_config
from src.visualization import CLASS_COLORS, CLASS_COLORS_FLOAT, CLASS_NAMES, create_mask_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze bark segmentation dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/dataset_analysis")
    return parser.parse_args()


def collect_dataset_info(ds_root: str) -> pd.DataFrame:
    """Scan all splits and collect image/mask statistics."""
    rows = []
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(ds_root, split, "images")
        mask_dir = os.path.join(ds_root, split, "masks")

        image_paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.JPG"))
        )

        for img_path in tqdm(image_paths, desc=f"Scanning {split}"):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, f"{stem}.png")

            img = Image.open(img_path)
            w, h = img.size

            row = {"stem": stem, "split": split, "width": w, "height": h}

            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))
                total = mask.size
                for c in range(3):
                    row[f"class_{c}_pct"] = (mask == c).sum() / total * 100
                row[f"num_classes_present"] = len(np.unique(mask))
            rows.append(row)

    return pd.DataFrame(rows)


def plot_split_summary(df: pd.DataFrame, output_dir: str):
    """Bar chart showing number of images per split."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Image counts
    counts = df.groupby("split").size().reindex(["train", "val", "test"])
    ax = axes[0]
    bars = ax.bar(counts.index, counts.values, color=["#4CAF50", "#2196F3", "#FF9800"], edgecolor="black")
    ax.set_ylabel("Number of Images")
    ax.set_title("Images per Split")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val),
                ha="center", va="bottom", fontweight="bold")

    # Class distribution per split
    ax = axes[1]
    x = np.arange(3)
    width = 0.25
    for i, split in enumerate(["train", "val", "test"]):
        split_df = df[df["split"] == split]
        means = [split_df[f"class_{c}_pct"].mean() for c in range(3)]
        bars = ax.bar(x + i * width, means, width, label=split, edgecolor="black", alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_ylabel("Mean Pixel Percentage (%)")
    ax.set_title("Mean Class Distribution per Split")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "split_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_image_class_distribution(df: pd.DataFrame, output_dir: str):
    """Stacked bar chart showing class distribution per image, sorted by class 2 %."""
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()
        split_df = split_df.sort_values("class_2_pct", ascending=False)
        n = len(split_df)

        fig, ax = plt.subplots(figsize=(max(12, n * 0.3), 6))
        x = np.arange(n)
        bottom = np.zeros(n)

        for c in range(3):
            vals = split_df[f"class_{c}_pct"].values
            color = CLASS_COLORS_FLOAT[c] if c > 0 else [0.7, 0.7, 0.7]
            ax.bar(x, vals, bottom=bottom, label=CLASS_NAMES[c], color=color, width=0.8, edgecolor="none")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(split_df["stem"].values, rotation=90, fontsize=6)
        ax.set_ylabel("Pixel Percentage (%)")
        ax.set_title(f"Per-Image Class Distribution ({split}, sorted by class 2 %)")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"class_distribution_{split}.png"), dpi=200, bbox_inches="tight")
        plt.close()


def plot_resolution_distribution(df: pd.DataFrame, output_dir: str):
    """Scatter plot of image dimensions."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"train": "#4CAF50", "val": "#2196F3", "test": "#FF9800"}
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        ax.scatter(split_df["width"], split_df["height"], label=split,
                   color=colors[split], s=40, alpha=0.7, edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.set_title("Image Resolution Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary stats
    stats_text = (
        f"Width: {df['width'].min()}-{df['width'].max()} (mean {df['width'].mean():.0f})\n"
        f"Height: {df['height'].min()}-{df['height'].max()} (mean {df['height'].mean():.0f})\n"
        f"Unique sizes: {len(df.groupby(['width', 'height']))}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resolution_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_sample_grid(ds_root: str, output_dir: str, n_per_split: int = 3):
    """Grid of sample images with mask overlays from each split."""
    fig, axes = plt.subplots(3, n_per_split * 2, figsize=(6 * n_per_split, 10))

    for row, split in enumerate(["train", "val", "test"]):
        img_dir = os.path.join(ds_root, split, "images")
        mask_dir = os.path.join(ds_root, split, "masks")
        images = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.JPG"))
        )

        # Sample evenly spaced images
        indices = np.linspace(0, len(images) - 1, n_per_split, dtype=int)

        for col, idx in enumerate(indices):
            stem = os.path.splitext(os.path.basename(images[idx]))[0]
            img = np.array(Image.open(images[idx]).convert("RGB"))
            mask_path = os.path.join(mask_dir, f"{stem}.png")

            # Downsample for display
            scale = 800 / max(img.shape[:2])
            h_new, w_new = int(img.shape[0] * scale), int(img.shape[1] * scale)
            img_small = np.array(Image.fromarray(img).resize((w_new, h_new), Image.BILINEAR))

            axes[row, col * 2].imshow(img_small)
            axes[row, col * 2].set_title(f"{split}/{stem}", fontsize=8)
            axes[row, col * 2].axis("off")

            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))
                mask_small = np.array(Image.fromarray(mask).resize((w_new, h_new), Image.NEAREST))
                overlay = create_mask_overlay(img_small, mask_small)
                axes[row, col * 2 + 1].imshow(overlay)
            axes[row, col * 2 + 1].set_title(f"overlay", fontsize=8)
            axes[row, col * 2 + 1].axis("off")

    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(CLASS_COLORS_FLOAT[1:], CLASS_NAMES[1:])]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=11, frameon=True)

    plt.suptitle("Dataset Samples with Mask Overlays", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "sample_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_class1_distribution(df: pd.DataFrame, output_dir: str):
    """Detailed analysis of the rare class 1."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram of class 1 percentage
    ax = axes[0]
    all_c1 = df["class_1_pct"].dropna()
    ax.hist(all_c1[all_c1 > 0], bins=30, color="teal", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Class 1 Pixel Percentage (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Class 1 Distribution (non-zero, n={(all_c1 > 0).sum()}/{len(all_c1)})")
    ax.grid(True, alpha=0.3)

    # Fraction of images with class 1
    ax = axes[1]
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        has_c1 = (split_df["class_1_pct"] > 0).sum()
        total = len(split_df)
        ax.bar(split, has_c1 / total * 100, color="teal", edgecolor="black", alpha=0.7)
        ax.text(split, has_c1 / total * 100 + 1, f"{has_c1}/{total}", ha="center", fontsize=10)
    ax.set_ylabel("% Images with Class 1")
    ax.set_title("Class 1 Presence per Split")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    # Class 1 vs Class 2 scatter
    ax = axes[2]
    ax.scatter(df["class_2_pct"], df["class_1_pct"], c=df["split"].map(
        {"train": "#4CAF50", "val": "#2196F3", "test": "#FF9800"}
    ), s=40, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Class 2 Percentage (%)")
    ax.set_ylabel("Class 1 Percentage (%)")
    ax.set_title("Class 1 vs Class 2 Coverage")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class1_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()


def print_summary_table(df: pd.DataFrame):
    """Print a summary statistics table."""
    print("\n" + "=" * 80)
    print("  DATASET SUMMARY")
    print("=" * 80)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        n = len(split_df)
        print(f"\n  {split.upper()} ({n} images):")
        print(f"    Resolution: {split_df['width'].min()}x{split_df['height'].min()} to "
              f"{split_df['width'].max()}x{split_df['height'].max()}")
        for c in range(3):
            col = f"class_{c}_pct"
            if col in split_df.columns:
                print(f"    Class {c} ({CLASS_NAMES[c]}): "
                      f"mean={split_df[col].mean():.1f}%, "
                      f"min={split_df[col].min():.1f}%, "
                      f"max={split_df[col].max():.1f}%")

    total = len(df)
    print(f"\n  TOTAL: {total} labeled images")

    # Inference images
    print("=" * 80)


def main():
    args = parse_args()
    config = load_config(args.config)
    ds_root = config["dataset"]["root"]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Scanning dataset...")
    df = collect_dataset_info(ds_root)

    # Save raw data
    df.to_csv(os.path.join(output_dir, "dataset_stats.csv"), index=False)

    print_summary_table(df)

    print("\nGenerating visualizations...")
    plot_split_summary(df, output_dir)
    plot_per_image_class_distribution(df, output_dir)
    plot_resolution_distribution(df, output_dir)
    plot_class1_distribution(df, output_dir)
    plot_sample_grid(ds_root, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
