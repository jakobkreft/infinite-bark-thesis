"""
Cut cropped bark images + masks into square patches and write them directly in
DiffInfinite flat-folder format, at three resolutions (original / 1024 / 512).

Pipeline (single pass, no intermediate folders):
  1. Crop CROP_TOP / CROP_BOTTOM pixels off every image + mask (edge distortion).
  2. Tile into square patches:
       - NUM_ROWS rows stacked vertically, overlapping by ROW_OVERLAP_FRACTION
         of the patch side. The square side is chosen so exactly NUM_ROWS rows
         fill the cropped height.
       - Left -> right: minimum number of squares with uniform overlap to cover
         the full width.
  3. For every patch, save it at each resolution into
       diffinfinite-bark_<res>/  as  bark_XXXX.jpg + bark_XXXX_mask.png
     (same bark_XXXX id across all resolutions), plus class_to_int.yml and a
     name_mapping.csv for traceability.

Linux. No CLI flags - edit the CONFIG block below.
"""

import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ===========================================================================
# CONFIG  (edit here)
# ===========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

IMAGE_DIR = SCRIPT_DIR / "dataset" / "images"
MASK_DIR = SCRIPT_DIR / "dataset" / "masks"
OUTPUT_BASE = SCRIPT_DIR                 # diffinfinite-bark_<res> folders go here

# How many vertical rows of squares (1, 2, or 3 ...).
NUM_ROWS = 1
# Vertical overlap between adjacent rows, as a fraction of the patch side.
ROW_OVERLAP_FRACTION = 0.25              # 1/4

# Pixels to crop off the top and bottom before patching (distorted regions).
CROP_TOP = 100
CROP_BOTTOM = 100

# Output resolutions. None = keep original patch size (no resize).
RESOLUTIONS = {
    "original": None,
    "1024": 1024,
    "512": 512,
}

JPEG_QUALITY = 95
CLEAR_OUTPUT = True                      # wipe existing diffinfinite-bark_* first

# DiffInfinite class map (bark=0, knot=1, defect=2)
CLASS_TO_INT_YAML = """\
features:
  target__tfrec:
    class_to_int:
      bark: 0
      knot: 1
      defect: 2
"""
# ===========================================================================


def compute_patch_positions(length: int, patch_size: int) -> list[int]:
    """Start positions to cover `length` with minimum squares of `patch_size`.

    First patch at 0, last patch ends at `length`, uniform overlap in between.
    """
    if length <= patch_size:
        return [0]
    n = math.ceil(length / patch_size)
    if n == 1:
        return [0]
    return [round(i * (length - patch_size) / (n - 1)) for i in range(n)]


def compute_row_positions(length: int, patch_size: int, num_rows: int) -> list[int]:
    """Exactly `num_rows` start positions spanning [0, length - patch_size]."""
    if num_rows <= 1 or length <= patch_size:
        return [0]
    return [round(i * (length - patch_size) / (num_rows - 1)) for i in range(num_rows)]


def compute_geometry(h: int, w: int) -> tuple[int, list[int], list[int]]:
    """Return (patch_size, y_positions, x_positions) for a cropped image.

    Patch side is set so NUM_ROWS rows fill the height with ROW_OVERLAP_FRACTION
    overlap. If that side would exceed the width (e.g. a portrait image), we fall
    back to width-limited squares and auto-fit rows.
    """
    step_fraction = 1.0 - ROW_OVERLAP_FRACTION
    denom = 1.0 + (NUM_ROWS - 1) * step_fraction
    patch_size = int(h / denom)

    if patch_size <= 0:
        return 0, [], []

    if patch_size > w:
        # Geometry assumption (landscape) broken; fit to width and auto-row.
        patch_size = w
        y_positions = compute_patch_positions(h, patch_size)
    else:
        y_positions = compute_row_positions(h, patch_size, NUM_ROWS)

    x_positions = compute_patch_positions(w, patch_size)
    return patch_size, y_positions, x_positions


def find_mask(stem: str) -> Path | None:
    for ext in (".png", ".PNG"):
        p = MASK_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    # ---- prepare output folders ------------------------------------------
    out_dirs: dict[str, Path] = {}
    for res_name in RESOLUTIONS:
        d = OUTPUT_BASE / f"diffinfinite-bark_{res_name}"
        if CLEAR_OUTPUT and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[res_name] = d

    # ---- collect images --------------------------------------------------
    exts = ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    image_paths: list[Path] = []
    for ext in exts:
        image_paths.extend(IMAGE_DIR.glob(ext))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Images:        {len(image_paths)}")
    print(f"Rows:          {NUM_ROWS}  (overlap {ROW_OVERLAP_FRACTION:.2%})")
    print(f"Crop top/bot:  {CROP_TOP}/{CROP_BOTTOM} px")
    print(f"Resolutions:   {list(RESOLUTIONS.keys())}")
    print(f"Output base:   {OUTPUT_BASE}\n")

    idx = 0                       # global patch counter -> shared bark_XXXX id
    mapping: list[tuple] = []     # (new_name, original_stem, row, col)

    for img_path in tqdm(image_paths, desc="Patching"):
        stem = img_path.stem

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            tqdm.write(f"  ERROR: cannot read {img_path.name}")
            continue

        mask_path = find_mask(stem)
        if mask_path is None:
            tqdm.write(f"  WARNING: no mask for {img_path.name}, skipping")
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            tqdm.write(f"  ERROR: cannot read mask {mask_path.name}")
            continue

        # Align mask to image size if they differ (label-safe nearest).
        if mask.shape[:2] != img.shape[:2]:
            tqdm.write(f"  WARNING: mask/image size mismatch for {stem}; resizing mask (nearest)")
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        H, W = img.shape[:2]

        # ---- crop top / bottom -------------------------------------------
        top, bottom = CROP_TOP, H - CROP_BOTTOM
        if bottom - top <= 0:
            tqdm.write(f"  WARNING: {stem} too short to crop {CROP_TOP}+{CROP_BOTTOM}px, skipping")
            continue
        img = img[top:bottom, :]
        mask = mask[top:bottom, :]
        Hc, Wc = img.shape[:2]

        # ---- geometry ----------------------------------------------------
        patch_size, y_positions, x_positions = compute_geometry(Hc, Wc)
        if patch_size <= 0:
            tqdm.write(f"  WARNING: {stem} could not compute geometry, skipping")
            continue

        # ---- emit patches ------------------------------------------------
        for ri, y in enumerate(y_positions):
            for ci, x in enumerate(x_positions):
                img_patch = img[y:y + patch_size, x:x + patch_size]
                mask_patch = mask[y:y + patch_size, x:x + patch_size]

                idx += 1
                new_stem = f"bark_{idx:04d}"
                mapping.append((new_stem, stem, ri, ci))

                for res_name, res_size in RESOLUTIONS.items():
                    if res_size is not None:
                        img_out = cv2.resize(img_patch, (res_size, res_size),
                                             interpolation=cv2.INTER_AREA)
                        mask_out = cv2.resize(mask_patch, (res_size, res_size),
                                              interpolation=cv2.INTER_NEAREST)
                    else:
                        img_out, mask_out = img_patch, mask_patch

                    folder = out_dirs[res_name]
                    cv2.imwrite(str(folder / f"{new_stem}.jpg"), img_out,
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    cv2.imwrite(str(folder / f"{new_stem}_mask.png"), mask_out)

    # ---- per-folder metadata --------------------------------------------
    for res_name, folder in out_dirs.items():
        (folder / "class_to_int.yml").write_text(CLASS_TO_INT_YAML)
        with open(folder / "name_mapping.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["new_name", "original_name", "row", "col"])
            writer.writerows(mapping)

    # ---- summary ---------------------------------------------------------
    print(f"\nDone. {idx} patches per resolution.")
    for res_name, folder in out_dirs.items():
        n_imgs = len(list(folder.glob("*.jpg")))
        n_masks = len(list(folder.glob("*_mask.png")))
        size = RESOLUTIONS[res_name]
        label = f"{size}x{size}" if size else "original"
        print(f"  diffinfinite-bark_{res_name} ({label}): {n_imgs} images, {n_masks} masks")


if __name__ == "__main__":
    main()
