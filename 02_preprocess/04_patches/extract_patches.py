"""
Extract square patches from cropped images and masks.

Fits the minimum number of non-overlapping (or minimally overlapping) squares
so that the entire image is covered. Saves patches at three resolutions:
original, 1024x1024, and 512x512.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def compute_patch_positions(length: int, patch_size: int) -> list[int]:
    """Compute start positions along one axis to cover `length` with patches of `patch_size`.

    Uses the minimum number of patches with uniform spacing so that:
    - First patch starts at 0
    - Last patch ends at `length`
    - All patches have equal overlap

    Returns:
        List of start positions.
    """
    if length <= patch_size:
        return [0]
    n = math.ceil(length / patch_size)
    if n == 1:
        return [0]
    # Distribute n patches evenly: first at 0, last at length - patch_size
    positions = [round(i * (length - patch_size) / (n - 1)) for i in range(n)]
    return positions


def extract_patches(img: np.ndarray, patch_size: int) -> list[tuple[np.ndarray, int, int]]:
    """Extract square patches covering the entire image.

    Args:
        img: Input image (H, W) or (H, W, C).
        patch_size: Side length of each square patch.

    Returns:
        List of (patch, row_idx, col_idx) tuples.
    """
    H, W = img.shape[:2]
    y_positions = compute_patch_positions(H, patch_size)
    x_positions = compute_patch_positions(W, patch_size)

    patches = []
    for ri, y in enumerate(y_positions):
        for ci, x in enumerate(x_positions):
            patch = img[y:y + patch_size, x:x + patch_size]
            patches.append((patch, ri, ci))
    return patches


def main():
    parser = argparse.ArgumentParser(
        description="Extract square patches from images and masks at multiple resolutions."
    )
    parser.add_argument("--image_dir", type=str,
                        default="D:/jk/dataset_barkseg_cropped/images",
                        help="Input cropped images directory.")
    parser.add_argument("--mask_dir", type=str,
                        default="D:/jk/dataset_barkseg_cropped/masks",
                        help="Input cropped masks directory.")
    parser.add_argument("--output_base", type=str,
                        default="D:/jk",
                        help="Base output directory.")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG save quality (default: 95).")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    # Define output directories for each resolution
    resolutions = {
        "original": None,  # no resize
        "1024": 1024,
        "512": 512,
    }

    out_dirs = {}
    for res_name, res_size in resolutions.items():
        img_out = Path(args.output_base) / f"patches_{res_name}" / "images"
        mask_out = Path(args.output_base) / f"patches_{res_name}" / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        out_dirs[res_name] = (img_out, mask_out, res_size)

    # Collect image files
    extensions = ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(ext))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images")
    print(f"Output resolutions: {list(resolutions.keys())}")
    print()

    total_patches = 0

    for img_path in tqdm(image_paths, desc="Extracting patches"):
        stem = img_path.stem

        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            tqdm.write(f"  ERROR: could not read {img_path.name}")
            continue

        # Read mask
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{stem}.PNG"
        if not mask_path.exists():
            tqdm.write(f"  WARNING: no mask for {img_path.name}, skipping")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            tqdm.write(f"  ERROR: could not read mask {mask_path.name}")
            continue

        H, W = img.shape[:2]
        patch_size = min(H, W)

        # Extract patches
        img_patches = extract_patches(img, patch_size)
        mask_patches = extract_patches(mask, patch_size)

        for (img_patch, ri, ci), (mask_patch, _, _) in zip(img_patches, mask_patches):
            patch_name = f"{stem}_r{ri}_c{ci}"
            total_patches += 1

            for res_name, (img_out, mask_out, res_size) in out_dirs.items():
                if res_size is not None:
                    img_resized = cv2.resize(img_patch, (res_size, res_size),
                                             interpolation=cv2.INTER_AREA)
                    mask_resized = cv2.resize(mask_patch, (res_size, res_size),
                                              interpolation=cv2.INTER_NEAREST)
                else:
                    img_resized = img_patch
                    mask_resized = mask_patch

                cv2.imwrite(str(img_out / f"{patch_name}.jpg"), img_resized,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                cv2.imwrite(str(mask_out / f"{patch_name}.png"), mask_resized)

    print(f"\nDone. Extracted {total_patches} patches from {len(image_paths)} images.")
    for res_name, (img_out, mask_out, res_size) in out_dirs.items():
        n = len(list(img_out.glob("*.jpg")))
        size_label = f"{res_size}x{res_size}" if res_size else "original"
        print(f"  {res_name} ({size_label}): {n} patches in {img_out.parent}")


if __name__ == "__main__":
    main()
