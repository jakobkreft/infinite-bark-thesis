"""
Compare NEAREST vs Scale2x mask upscaling side by side.

Generates NEAREST masks on-the-fly from the low-res originals so we
don't need dataset_step1 on disk. Scale2x masks come from dataset_step1_smooth.

Saved to dataset_step1_smooth/visualizations/
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

SRC_MASKS = Path("dataset_step0") / "masks-unwrap"
SMOOTH_DIR = Path("dataset_step1_smooth")
VIS_DIR = SMOOTH_DIR / "visualizations"

CLASSES = {
    0: {"name": "Background", "color": (0, 0, 0)},
    1: {"name": "Slepice / odrezane veje", "color": (0, 255, 255)},
    2: {"name": "Mehanske poškodbe / odluščeno", "color": (255, 128, 0)},
}

CLASS_COLORS_NORM = {
    0: (0.3, 0.3, 0.3),
    1: (0.0, 1.0, 1.0),
    2: (1.0, 0.5, 0.0),
}

DPI = 300
ALPHA = 0.5
N_SAMPLES = 4


def colorize_mask(mask_arr):
    h, w = mask_arr.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, info in CLASSES.items():
        colored[mask_arr == cls_id] = info["color"]
    return colored


def overlay(img_arr, mask_arr, alpha=ALPHA):
    colored = colorize_mask(mask_arr).astype(np.float32)
    result = img_arr.astype(np.float32).copy()
    fg = mask_arr > 0
    for c in range(3):
        result[:, :, c][fg] = (1 - alpha) * result[:, :, c][fg] + alpha * colored[:, :, c][fg]
    return result.astype(np.uint8)


def find_interesting_crop(mask_arr, crop_size=800):
    """Find a crop region near a class boundary."""
    from PIL import ImageFilter
    h, w = mask_arr.shape
    edges_h = mask_arr[1:, :] != mask_arr[:-1, :]
    edges_v = mask_arr[:, 1:] != mask_arr[:, :-1]
    edge_density = np.zeros((h, w), dtype=np.float32)
    edge_density[1:, :] += edges_h.astype(np.float32)
    edge_density[:, 1:] += edges_v.astype(np.float32)
    density_img = Image.fromarray((edge_density * 255).astype(np.uint8))
    density_img = density_img.filter(ImageFilter.GaussianBlur(radius=50))
    density = np.array(density_img).astype(np.float32)
    half = crop_size // 2
    density[:half, :] = 0
    density[h - half:, :] = 0
    density[:, :half] = 0
    density[:, w - half:] = 0
    if density.max() == 0:
        cy, cx = h // 2, w // 2
    else:
        idx = np.argmax(density)
        cy, cx = np.unravel_index(idx, density.shape)
    y1 = max(0, cy - half)
    x1 = max(0, cx - half)
    return y1, y1 + crop_size, x1, x1 + crop_size


def make_legend():
    return [
        mpatches.Patch(facecolor=CLASS_COLORS_NORM[c], edgecolor="gray",
                       label=f"Class {c}: {CLASSES[c]['name']}")
        for c in sorted(CLASSES.keys())
    ]


def get_samples(n=N_SAMPLES, seed=42):
    """Pick samples with meaningful foreground content."""
    stems = sorted(p.stem for p in (SMOOTH_DIR / "masks").glob("*.png"))
    good = []
    for stem in stems:
        mask = np.array(Image.open(SMOOTH_DIR / "masks" / f"{stem}.png").convert("L"))
        if 100.0 * np.sum(mask > 0) / mask.size > 5:
            good.append(stem)
    return random.Random(seed).sample(good, min(n, len(good)))


def get_nearest_mask(stem, target_w, target_h):
    """Generate NEAREST-upscaled mask on-the-fly from low-res source."""
    src = SRC_MASKS / f"{stem}.png"
    mask = Image.open(src).convert("L")
    return np.array(mask.resize((target_w, target_h), Image.NEAREST))


def plot_full_comparison(samples, output_path):
    """Side-by-side overlay comparison."""
    fig, axes = plt.subplots(len(samples), 3, figsize=(18, 5 * len(samples)))
    fig.suptitle("Mask Upscaling Comparison: NEAREST vs Scale2x (EPX)",
                 fontweight="bold", fontsize=14, y=0.99)

    for row, stem in enumerate(samples):
        img_path = SMOOTH_DIR / "images" / f"{stem}.jpg"
        smooth_path = SMOOTH_DIR / "masks" / f"{stem}.png"

        img = Image.open(img_path).convert("RGB")
        img.thumbnail((1200, 1200), Image.LANCZOS)
        img_arr = np.array(img)
        th, tw = img_arr.shape[:2]

        nearest_mask = get_nearest_mask(stem, tw, th)
        smooth_mask = np.array(
            Image.open(smooth_path).convert("L").resize((tw, th), Image.NEAREST)
        )

        axes[row, 0].imshow(img_arr)
        axes[row, 1].imshow(overlay(img_arr, nearest_mask))
        axes[row, 2].imshow(overlay(img_arr, smooth_mask))

        if row == 0:
            axes[row, 0].set_title("Original Image", fontweight="bold")
            axes[row, 1].set_title("NEAREST Interpolation", fontweight="bold")
            axes[row, 2].set_title("Scale2x (EPX)", fontweight="bold")
        else:
            axes[row, 0].set_title(stem, fontsize=9)

        for col in range(3):
            axes[row, col].axis("off")

    fig.legend(handles=make_legend(), loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_zoom_comparison(samples, output_path):
    """Zoomed-in boundary detail."""
    fig, axes = plt.subplots(len(samples), 3, figsize=(18, 6 * len(samples)))
    fig.suptitle("Zoomed Boundary Detail: NEAREST vs Scale2x (EPX)",
                 fontweight="bold", fontsize=14, y=0.99)
    crop_size = 800

    for row, stem in enumerate(samples):
        img_arr = np.array(Image.open(SMOOTH_DIR / "images" / f"{stem}.jpg").convert("RGB"))
        smooth_arr = np.array(Image.open(SMOOTH_DIR / "masks" / f"{stem}.png").convert("L"))
        h, w = smooth_arr.shape
        nearest_arr = get_nearest_mask(stem, w, h)

        y1, y2, x1, x2 = find_interesting_crop(smooth_arr, crop_size)
        img_crop = img_arr[y1:y2, x1:x2]
        nearest_crop = nearest_arr[y1:y2, x1:x2]
        smooth_crop = smooth_arr[y1:y2, x1:x2]

        axes[row, 0].imshow(img_crop)
        axes[row, 1].imshow(overlay(img_crop, nearest_crop))
        axes[row, 2].imshow(overlay(img_crop, smooth_crop))

        if row == 0:
            axes[row, 0].set_title("Original (zoom)", fontweight="bold")
            axes[row, 1].set_title("NEAREST (zoom)", fontweight="bold")
            axes[row, 2].set_title("Scale2x (zoom)", fontweight="bold")
        else:
            axes[row, 0].set_title(stem, fontsize=9)

        for col in range(3):
            axes[row, col].axis("off")

    fig.legend(handles=make_legend(), loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_mask_only_comparison(samples, output_path):
    """Color-coded masks side by side (no image)."""
    fig, axes = plt.subplots(len(samples), 2, figsize=(14, 5 * len(samples)))
    fig.suptitle("Mask Shape Comparison: NEAREST vs Scale2x (EPX)",
                 fontweight="bold", fontsize=14, y=0.99)

    for row, stem in enumerate(samples):
        smooth_arr = np.array(Image.open(SMOOTH_DIR / "masks" / f"{stem}.png").convert("L"))
        h, w = smooth_arr.shape
        nearest_arr = get_nearest_mask(stem, w, h)

        # Downscale for display
        scale = 1000 / w
        tw, th = 1000, int(h * scale)
        nearest_small = np.array(Image.fromarray(nearest_arr).resize((tw, th), Image.NEAREST))
        smooth_small = np.array(Image.fromarray(smooth_arr).resize((tw, th), Image.NEAREST))

        axes[row, 0].imshow(colorize_mask(nearest_small))
        axes[row, 1].imshow(colorize_mask(smooth_small))

        if row == 0:
            axes[row, 0].set_title("NEAREST", fontweight="bold")
            axes[row, 1].set_title("Scale2x (EPX)", fontweight="bold")
        else:
            axes[row, 0].set_title(stem, fontsize=9)

        for col in range(2):
            axes[row, col].axis("off")

    fig.legend(handles=make_legend(), loc="lower center", ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    samples = get_samples()
    print(f"Selected {len(samples)} samples: {samples}")

    print("\nGenerating comparison visualizations:")
    plot_full_comparison(samples, VIS_DIR / "comparison_overlay.png")
    plot_zoom_comparison(samples, VIS_DIR / "comparison_zoom.png")
    plot_mask_only_comparison(samples, VIS_DIR / "comparison_masks.png")

    print(f"\nAll visualizations saved to {VIS_DIR}/")


if __name__ == "__main__":
    main()
