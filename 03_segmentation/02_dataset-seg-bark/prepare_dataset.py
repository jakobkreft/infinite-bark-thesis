"""
Prepare dataset_step1: full-resolution image-mask pairs for segmentation training.

Reads low-res masks (20px wide) and original images from dataset_step0/,
crops images to match the labeled region, upscales masks to full resolution,
and saves paired outputs to dataset_step1/.

Classes:
    0 - Background (black)
    1 - slepice / odrezane veje (cyan)
    2 - mehanske poskodbe / odlusceno (orange)
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# --- Configuration ---
SRC_DIR = Path("dataset_step0")
SRC_IMAGES = SRC_DIR / "images-unwrap"
SRC_MASKS = SRC_DIR / "masks-unwrap"

DST_DIR = Path("dataset_step1")
DST_IMAGES = DST_DIR / "images"
DST_MASKS = DST_DIR / "masks"

TARGET_WIDTH_1000 = 1000  # Width used by labeling tool
GRID_COLS = 20            # Number of grid columns in labeling tool
CELL_SIZE_1000 = TARGET_WIDTH_1000 / GRID_COLS  # 50.0 px at 1000px scale

VALID_CLASSES = {0, 1, 2}
IMAGE_EXTENSIONS = [".JPG", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


def find_image_for_mask(mask_stem: str) -> Path | None:
    """Find the original image file matching a mask stem."""
    for ext in IMAGE_EXTENSIONS:
        candidate = SRC_IMAGES / (mask_stem + ext)
        if candidate.exists():
            return candidate
    return None


def compute_crop_params(orig_w: int, orig_h: int, mask_h: int):
    """
    Reproduce the labeling tool's grid math and compute crop at original resolution.

    The labeling tool:
    1. Resized image to 1000px wide
    2. Computed grid_h = int(new_h / 50)
    3. Center-cropped to grid_h * 50 pixels tall (at 1000px scale)

    We reverse this to find the crop region at original resolution.

    Returns:
        (top_orig, crop_h_orig, grid_h) or None if grid_h doesn't match mask_h.
    """
    scale = TARGET_WIDTH_1000 / orig_w
    new_h = int(orig_h * scale)
    grid_h = int(new_h / CELL_SIZE_1000)

    if grid_h != mask_h:
        return None

    target_h_1000 = int(grid_h * CELL_SIZE_1000)
    top_1000 = (new_h - target_h_1000) // 2

    # Map back to original resolution
    top_orig = round(top_1000 / scale)
    crop_h_orig = round(target_h_1000 / scale)

    return top_orig, crop_h_orig, grid_h


def process_pair(mask_path: Path, image_path: Path) -> dict:
    """
    Process one image-mask pair.

    Returns a dict with status info and statistics.
    """
    stem = mask_path.stem

    # Load mask
    mask = Image.open(mask_path).convert("L")
    mask_w, mask_h = mask.size
    assert mask_w == GRID_COLS, f"{stem}: mask width {mask_w} != {GRID_COLS}"

    # Load original image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Compute crop parameters
    params = compute_crop_params(orig_w, orig_h, mask_h)
    if params is None:
        scale = TARGET_WIDTH_1000 / orig_w
        new_h = int(orig_h * scale)
        expected_grid_h = int(new_h / CELL_SIZE_1000)
        return {
            "stem": stem,
            "success": False,
            "error": f"grid_h mismatch: expected {expected_grid_h}, mask has {mask_h}",
        }

    top_orig, crop_h_orig, grid_h = params

    # Crop original image
    img_cropped = img.crop((0, top_orig, orig_w, top_orig + crop_h_orig))

    # Upscale mask to match cropped image dimensions (NEAREST preserves class labels)
    mask_upscaled = mask.resize((orig_w, crop_h_orig), Image.NEAREST)

    # Verify dimensions match
    assert img_cropped.size == mask_upscaled.size, (
        f"{stem}: size mismatch after processing: "
        f"img={img_cropped.size}, mask={mask_upscaled.size}"
    )

    # Verify mask values
    mask_values = set(np.unique(np.array(mask_upscaled)))
    if not mask_values.issubset(VALID_CLASSES):
        return {
            "stem": stem,
            "success": False,
            "error": f"unexpected mask values: {mask_values}",
        }

    # Save outputs
    img_cropped.save(DST_IMAGES / f"{stem}.jpg", quality=95)
    mask_upscaled.save(DST_MASKS / f"{stem}.png")

    # Collect statistics
    mask_arr = np.array(mask_upscaled)
    class_counts = {c: int(np.sum(mask_arr == c)) for c in VALID_CLASSES}

    return {
        "stem": stem,
        "success": True,
        "orig_size": (orig_w, orig_h),
        "cropped_size": img_cropped.size,
        "class_counts": class_counts,
    }


def main():
    # Discover mask files
    mask_files = sorted(SRC_MASKS.glob("*.png"))
    if not mask_files:
        print(f"No mask files found in {SRC_MASKS}")
        sys.exit(1)

    print(f"Found {len(mask_files)} masks in {SRC_MASKS}")

    # Create output directories
    DST_IMAGES.mkdir(parents=True, exist_ok=True)
    DST_MASKS.mkdir(parents=True, exist_ok=True)

    # Process pairs
    results = []
    errors = []

    for i, mask_path in enumerate(mask_files):
        stem = mask_path.stem
        image_path = find_image_for_mask(stem)

        if image_path is None:
            msg = f"No matching image found for mask {mask_path.name}"
            print(f"  SKIP [{i+1}/{len(mask_files)}] {stem}: {msg}")
            errors.append(msg)
            continue

        try:
            result = process_pair(mask_path, image_path)
            results.append(result)

            if result["success"]:
                w, h = result["cropped_size"]
                print(f"  OK   [{i+1}/{len(mask_files)}] {stem}: {w}x{h}")
            else:
                print(f"  FAIL [{i+1}/{len(mask_files)}] {stem}: {result['error']}")
                errors.append(f"{stem}: {result['error']}")

        except Exception as e:
            msg = f"{stem}: {e}"
            print(f"  ERR  [{i+1}/{len(mask_files)}] {msg}")
            errors.append(msg)

    # --- Summary ---
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total masks:      {len(mask_files)}")
    print(f"  Processed OK:     {len(successful)}")
    print(f"  Failed:           {len(failed)}")
    print(f"  Skipped (no img): {len(mask_files) - len(results)}")

    if successful:
        widths = [r["cropped_size"][0] for r in successful]
        heights = [r["cropped_size"][1] for r in successful]
        print(f"\n  Resolution range:")
        print(f"    Width:  {min(widths)} - {max(widths)}")
        print(f"    Height: {min(heights)} - {max(heights)}")

        # Aggregate class counts
        total_counts = {c: 0 for c in VALID_CLASSES}
        for r in successful:
            for c, count in r["class_counts"].items():
                total_counts[c] += count
        total_pixels = sum(total_counts.values())

        print(f"\n  Class distribution (total {total_pixels:,} pixels):")
        for c in sorted(VALID_CLASSES):
            pct = 100 * total_counts[c] / total_pixels if total_pixels > 0 else 0
            print(f"    Class {c}: {total_counts[c]:>12,} pixels ({pct:5.1f}%)")

    if errors:
        print(f"\n  ERRORS:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)

    print(f"\nOutput saved to {DST_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
