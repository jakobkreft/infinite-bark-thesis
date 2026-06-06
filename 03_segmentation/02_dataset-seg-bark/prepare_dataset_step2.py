"""
Prepare dataset_step2: train/val/test split + inference images.

Splits the 140 labeled image-mask pairs from dataset_step1_smooth/ into
train/val/test using stratified sampling, copies files into the split
directories, and copies ALL images (labeled + unlabeled) into inference/.

No patching — that belongs in the training pipeline (step 3).

Classes:
    0 - Background
    1 - slepice / odrezane veje (cyan)
    2 - mehanske poskodbe / odlusceno (orange)
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# --- Configuration ---
STEP1_DIR = Path("dataset_step1_smooth")
STEP1_IMAGES = STEP1_DIR / "images"
STEP1_MASKS = STEP1_DIR / "masks"

STEP0_DIR = Path("dataset_step0")
STEP0_IMAGES = STEP0_DIR / "images-unwrap"

DST_DIR = Path("dataset_step2")

VALID_CLASSES = {0, 1, 2}

# Stratification threshold: images with class 1 > this percentage are "has_class1"
CLASS1_THRESHOLD = 0.5  # percent


def get_labeled_pairs():
    """Return sorted list of (stem, image_path, mask_path) for labeled data."""
    pairs = []
    for mask_path in sorted(STEP1_MASKS.glob("*.png")):
        stem = mask_path.stem
        img_path = STEP1_IMAGES / f"{stem}.jpg"
        if img_path.exists():
            pairs.append((stem, img_path, mask_path))
        else:
            print(f"  WARNING: No image for mask {mask_path.name}")
    return pairs


def get_all_original_images():
    """Return sorted list of (stem, path) for ALL original images."""
    images = []
    for img_path in sorted(STEP0_IMAGES.iterdir()):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            images.append((img_path.stem, img_path))
    return images


def compute_image_stats(mask_path):
    """Compute class distribution and dimensions for one mask."""
    mask = Image.open(mask_path).convert("L")
    w, h = mask.size
    arr = np.array(mask)
    total = arr.size

    counts = {c: int(np.sum(arr == c)) for c in VALID_CLASSES}
    pcts = {c: 100.0 * counts[c] / total for c in VALID_CLASSES}

    return {
        "width": w,
        "height": h,
        "class_counts": counts,
        "class_pcts": pcts,
    }


def do_stratified_split(stems, stats, val_ratio, test_ratio, seed):
    """
    Split stems into train/val/test with stratification on class 1 presence.

    Returns (train_stems, val_stems, test_stems).
    """
    strat_labels = [
        1 if stats[s]["class_pcts"][1] > CLASS1_THRESHOLD else 0 for s in stems
    ]

    # First split: separate test from rest
    rest_stems, test_stems, rest_labels, _ = train_test_split(
        stems,
        strat_labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=strat_labels,
    )

    # Second split: separate val from train
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    train_stems, val_stems = train_test_split(
        rest_stems,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=rest_labels,
    )

    return sorted(train_stems), sorted(val_stems), sorted(test_stems)


def compute_split_class_distribution(stems, stats):
    """Compute aggregate class distribution for a set of stems."""
    total_counts = {c: 0 for c in VALID_CLASSES}
    for s in stems:
        for c in VALID_CLASSES:
            total_counts[c] += stats[s]["class_counts"][c]
    total = sum(total_counts.values())
    if total == 0:
        return {str(c): 0.0 for c in VALID_CLASSES}
    return {str(c): round(100.0 * total_counts[c] / total, 2) for c in VALID_CLASSES}


def verify_no_overlap(train_stems, val_stems, test_stems):
    """Verify there is zero overlap between splits. Raises on failure."""
    train_set = set(train_stems)
    val_set = set(val_stems)
    test_set = set(test_stems)

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    errors = []
    if overlap_tv:
        errors.append(f"Train/Val overlap: {sorted(overlap_tv)}")
    if overlap_tt:
        errors.append(f"Train/Test overlap: {sorted(overlap_tt)}")
    if overlap_vt:
        errors.append(f"Val/Test overlap: {sorted(overlap_vt)}")

    if errors:
        for e in errors:
            print(f"  FATAL: {e}")
        sys.exit(1)

    total = len(train_set) + len(val_set) + len(test_set)
    all_stems = train_set | val_set | test_set
    if total != len(all_stems):
        print(f"  FATAL: Duplicate stems within a split!")
        sys.exit(1)

    print(f"  Verified: 0 overlaps across {total} images in 3 splits")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset_step2: train/val/test split")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Show split without writing files")
    args = parser.parse_args()

    # --- Phase 1: Discovery ---
    print("Phase 1: Discovering data...")
    pairs = get_labeled_pairs()
    if not pairs:
        print(f"No labeled pairs found in {STEP1_DIR}")
        sys.exit(1)
    print(f"  Found {len(pairs)} labeled image-mask pairs")

    all_images = get_all_original_images()
    labeled_stems = {stem for stem, _, _ in pairs}
    unlabeled_count = sum(1 for s, _ in all_images if s not in labeled_stems)
    print(f"  Found {len(all_images)} total images ({len(pairs)} labeled, {unlabeled_count} unlabeled)")

    # Compute per-image statistics
    print("\n  Computing per-image class statistics...")
    stats = {}
    for i, (stem, _, mask_path) in enumerate(pairs):
        stats[stem] = compute_image_stats(mask_path)
        if (i + 1) % 50 == 0 or i == len(pairs) - 1:
            print(f"    {i + 1}/{len(pairs)} masks analyzed")

    # --- Phase 2: Stratified split ---
    print(f"\nPhase 2: Stratified split (seed={args.seed}, "
          f"val={args.val_ratio}, test={args.test_ratio})...")

    stems = sorted(labeled_stems)
    train_stems, val_stems, test_stems = do_stratified_split(
        stems, stats, args.val_ratio, args.test_ratio, args.seed
    )

    verify_no_overlap(train_stems, val_stems, test_stems)

    # Print summary
    print("\n" + "=" * 65)
    print("SPLIT SUMMARY")
    print("=" * 65)

    for split_name, split_stems in [("Train", train_stems), ("Val", val_stems), ("Test", test_stems)]:
        dist = compute_split_class_distribution(split_stems, stats)
        widths = [stats[s]["width"] for s in split_stems]
        heights = [stats[s]["height"] for s in split_stems]
        has_c1 = sum(1 for s in split_stems if stats[s]["class_pcts"][1] > CLASS1_THRESHOLD)

        print(f"\n  {split_name}: {len(split_stems)} images")
        print(f"    Resolution: {min(widths)}–{max(widths)} × {min(heights)}–{max(heights)} px")
        print(f"    Class 0 (background):  {dist['0']:5.1f}%")
        print(f"    Class 1 (slepice):     {dist['1']:5.1f}%  ({has_c1}/{len(split_stems)} images have >0.5%)")
        print(f"    Class 2 (meh. posk.):  {dist['2']:5.1f}%")

    print(f"\n  Inference: {len(all_images)} images (all labeled + unlabeled)")
    print("=" * 65)

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # --- Phase 3: Clean output and copy files ---
    print("\nPhase 3: Writing files...")

    # IMPORTANT: Remove old output to prevent stale files from prior runs
    if DST_DIR.exists():
        print(f"  Removing existing {DST_DIR}/...")
        shutil.rmtree(DST_DIR)

    # Copy labeled pairs into train/val/test
    for split_name, split_stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        img_dst = DST_DIR / split_name / "images"
        mask_dst = DST_DIR / split_name / "masks"
        img_dst.mkdir(parents=True)
        mask_dst.mkdir(parents=True)

        for stem in split_stems:
            shutil.copy2(STEP1_IMAGES / f"{stem}.jpg", img_dst / f"{stem}.jpg")
            shutil.copy2(STEP1_MASKS / f"{stem}.png", mask_dst / f"{stem}.png")

        print(f"  Copied {len(split_stems)} pairs to {split_name}/")

    # Symlink ALL original images to inference/ (saves ~1.9GB vs copying)
    inf_dst = DST_DIR / "inference" / "images"
    inf_dst.mkdir(parents=True)
    for stem, src_path in all_images:
        target = src_path.resolve()
        link = inf_dst / src_path.name
        link.symlink_to(target)
    print(f"  Symlinked {len(all_images)} images to inference/ (all labeled + unlabeled)")

    # --- Phase 4: Metadata ---
    print("\nPhase 4: Writing metadata...")
    meta_dir = DST_DIR / "metadata"
    meta_dir.mkdir(parents=True)

    split_data = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "stratification": f"has_class1 > {CLASS1_THRESHOLD}%",
        "source": str(STEP1_DIR),
        "created": datetime.now().isoformat(),
        "splits": {
            "train": train_stems,
            "val": val_stems,
            "test": test_stems,
        },
        "all_images": [s for s, _ in all_images],
        "class_distribution_per_split": {
            "train": compute_split_class_distribution(train_stems, stats),
            "val": compute_split_class_distribution(val_stems, stats),
            "test": compute_split_class_distribution(test_stems, stats),
        },
        "image_counts": {
            "train": len(train_stems),
            "val": len(val_stems),
            "test": len(test_stems),
            "inference": len(all_images),
        },
    }

    with open(meta_dir / "split.json", "w") as f:
        json.dump(split_data, f, indent=2)

    with open(meta_dir / "image_registry.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "split", "width", "height", "class_0_pct", "class_1_pct", "class_2_pct"])
        for split_name, split_stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
            for s in split_stems:
                st = stats[s]
                writer.writerow([
                    s, split_name, st["width"], st["height"],
                    round(st["class_pcts"][0], 2),
                    round(st["class_pcts"][1], 2),
                    round(st["class_pcts"][2], 2),
                ])

    print("  Saved metadata/split.json")
    print("  Saved metadata/image_registry.csv")

    # --- Phase 5: Final verification ---
    print("\nPhase 5: Verifying output...")

    # Check no file appears in multiple splits
    for split_a, split_b in [("train", "val"), ("train", "test"), ("val", "test")]:
        files_a = set(f.name for f in (DST_DIR / split_a / "images").iterdir())
        files_b = set(f.name for f in (DST_DIR / split_b / "images").iterdir())
        overlap = files_a & files_b
        if overlap:
            print(f"  FATAL: {split_a}/{split_b} share files: {sorted(overlap)[:5]}")
            sys.exit(1)

    # Check image-mask pairing in each split
    for split_name in ["train", "val", "test"]:
        imgs = set(f.stem for f in (DST_DIR / split_name / "images").iterdir())
        masks = set(f.stem for f in (DST_DIR / split_name / "masks").iterdir())
        if imgs != masks:
            missing_masks = imgs - masks
            missing_imgs = masks - imgs
            print(f"  FATAL: {split_name} has mismatched pairs!")
            if missing_masks:
                print(f"    Missing masks: {sorted(missing_masks)[:5]}")
            if missing_imgs:
                print(f"    Missing images: {sorted(missing_imgs)[:5]}")
            sys.exit(1)

    n_inf = len(list((DST_DIR / "inference" / "images").iterdir()))
    # Verify symlinks resolve
    broken = [p.name for p in (DST_DIR / "inference" / "images").iterdir() if not p.resolve().exists()]
    if broken:
        print(f"  WARNING: {len(broken)} broken symlinks in inference/")
    print(f"  Verified: splits are disjoint, all pairs match, {n_inf} inference images (symlinked)")

    print(f"\nOutput saved to {DST_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
