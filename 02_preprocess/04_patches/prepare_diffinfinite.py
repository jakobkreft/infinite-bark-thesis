"""
Convert patch datasets into DiffInfinite-ready flat folder format.

For each resolution (original, 1024, 512):
- Copies images as bark_XXXX.jpg (sequential naming)
- Copies masks as bark_XXXX_mask.png
- Writes class_to_int.yml
"""

import argparse
import csv
import shutil
from pathlib import Path

from tqdm import tqdm


CLASS_TO_INT_YAML = """\
features:
  target__tfrec:
    class_to_int:
      bark: 0
      knot: 1
      defect: 2
"""


def prepare_dataset(src_images: Path, src_masks: Path, output_dir: Path):
    """Convert one patch dataset to DiffInfinite flat format with clean naming."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files sorted for consistent ordering across resolutions
    image_paths = sorted(
        p for p in src_images.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    mapping = []
    for idx, img_path in enumerate(tqdm(image_paths, desc=f"  {output_dir.name}"), start=1):
        old_stem = img_path.stem
        new_stem = f"bark_{idx:04d}"

        # Copy image with new name
        shutil.copy2(img_path, output_dir / f"{new_stem}.jpg")

        # Copy mask with new name + _mask suffix
        mask_src = src_masks / f"{old_stem}.png"
        if not mask_src.exists():
            mask_src = src_masks / f"{old_stem}.PNG"
        if mask_src.exists():
            shutil.copy2(mask_src, output_dir / f"{new_stem}_mask.png")
        else:
            tqdm.write(f"  WARNING: no mask for {old_stem}")

        mapping.append((new_stem, old_stem))

    # Write class_to_int.yml
    (output_dir / "class_to_int.yml").write_text(CLASS_TO_INT_YAML)

    # Write name mapping CSV for traceability
    with open(output_dir / "name_mapping.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["new_name", "original_name"])
        writer.writerows(mapping)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DiffInfinite-ready datasets from patch datasets."
    )
    parser.add_argument("--input_base", type=str, default="D:/jk",
                        help="Base directory containing patches_* folders.")
    parser.add_argument("--output_base", type=str, default="D:/jk",
                        help="Base directory for output diffinfinite_* folders.")
    args = parser.parse_args()

    input_base = Path(args.input_base)
    output_base = Path(args.output_base)

    datasets = {
        "original": (input_base / "patches_original"),
        "1024": (input_base / "patches_1024"),
        "512": (input_base / "patches_512"),
    }

    print("Preparing DiffInfinite datasets...\n")

    for res_name, src_dir in datasets.items():
        src_images = src_dir / "images"
        src_masks = src_dir / "masks"
        output_dir = output_base / f"diffinfinite_{res_name}"

        if not src_images.exists():
            print(f"  SKIP: {src_images} not found")
            continue

        prepare_dataset(src_images, src_masks, output_dir)

        # Report
        n_imgs = len(list(output_dir.glob("*.jpg")))
        n_masks = len(list(output_dir.glob("*_mask.png")))
        print(f"  -> {n_imgs} images, {n_masks} masks in {output_dir}\n")

    print("Done.")


if __name__ == "__main__":
    main()
