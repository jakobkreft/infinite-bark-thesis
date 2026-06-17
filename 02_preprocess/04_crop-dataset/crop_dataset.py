"""
Crop top 1/6 and bottom 1/6 from images and masks, keeping the middle 2/3.

Removes deformed bark regions at the top and bottom edges of the tree logs.
Works on both the uniform-light images and corresponding masks.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def crop_middle(img: np.ndarray) -> np.ndarray:
    """Crop to the middle 2/3 vertically (remove top 1/6 and bottom 1/6)."""
    H = img.shape[0]
    top = H // 8
    bottom = H - H // 8
    return img[top:bottom]


def main():
    parser = argparse.ArgumentParser(
        description="Crop top/bottom 1/6 from images and masks."
    )
    parser.add_argument("--image_dir", type=str,
                        default="D:/jk/dataset_barkseg/images-uniform-light-noblurry",
                        help="Input images directory.")
    parser.add_argument("--mask_dir", type=str,
                        default="D:/jk/dataset_barkseg/masks-noblurry",
                        help="Input masks directory.")
    parser.add_argument("--output_dir", type=str,
                        default="D:/jk/dataset_barkseg_cropped",
                        help="Output dataset directory.")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG save quality for images (default: 95).")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    out_images = Path(args.output_dir) / "images"
    out_masks = Path(args.output_dir) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

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
    print(f"Output: {args.output_dir}")
    print()

    skipped = 0
    for img_path in tqdm(image_paths, desc="Cropping"):
        # Crop image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            tqdm.write(f"  ERROR: could not read {img_path.name}")
            skipped += 1
            continue

        img_cropped = crop_middle(img)
        cv2.imwrite(str(out_images / img_path.name), img_cropped,
                     [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

        # Crop matching mask (try .png then original extension)
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{stem}.PNG"
        if not mask_path.exists():
            tqdm.write(f"  WARNING: no mask for {img_path.name}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            tqdm.write(f"  ERROR: could not read mask {mask_path.name}")
            skipped += 1
            continue

        mask_cropped = crop_middle(mask)
        cv2.imwrite(str(out_masks / f"{stem}.png"), mask_cropped)

    total = len(image_paths)
    print(f"\nDone. Cropped {total - skipped}/{total} image+mask pairs.")
    if image_paths:
        H_orig = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR).shape[0]
        print(f"  Original height: {H_orig}, Cropped height: {H_orig - 2*(H_orig//6)}")


if __name__ == "__main__":
    main()
