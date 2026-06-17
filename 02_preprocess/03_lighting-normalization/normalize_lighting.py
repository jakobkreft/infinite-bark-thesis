"""
Normalize vertical lighting gradients in tree bark texture images.

Tree bark images photographed on cylindrical logs exhibit a vertical brightness
gradient (dark at top/bottom, bright in middle). This script estimates and removes
that gradient by working in CIELAB color space on the L channel only, preserving
color and texture detail.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


def compute_vertical_profile(L: np.ndarray, smooth_divisor: int):
    """Compute raw and smoothed row-mean luminance profiles.

    Args:
        L: Lightness channel, shape (H, W), float64.
        smooth_divisor: Gaussian sigma = H // smooth_divisor.

    Returns:
        (raw_profile, smoothed_profile) each of shape (H,).
    """
    raw_profile = L.mean(axis=1)
    sigma = max(L.shape[0] // smooth_divisor, 1)
    smoothed_profile = gaussian_filter1d(raw_profile, sigma=sigma)
    return raw_profile, smoothed_profile


def compute_gain(smoothed_profile: np.ndarray, gain_min: float, gain_max: float):
    """Compute per-row multiplicative gain to flatten the luminance profile.

    Args:
        smoothed_profile: Smoothed row-mean luminance, shape (H,).
        gain_min: Minimum allowed gain.
        gain_max: Maximum allowed gain.

    Returns:
        (gain, target) where gain has shape (H,) and target is the scalar target luminance.
    """
    target = float(np.median(smoothed_profile))
    gain = target / (smoothed_profile + 1e-6)
    gain = np.clip(gain, gain_min, gain_max)
    return gain, target


def apply_correction(img_bgr: np.ndarray, gain: np.ndarray):
    """Apply vertical gain correction to an image via its L channel.

    Args:
        img_bgr: Input BGR image, uint8.
        gain: Per-row gain, shape (H,).

    Returns:
        Corrected BGR image (uint8) and fraction of pixels that were clipped.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float64)

    L_corrected = L * gain[:, np.newaxis]
    clipped = np.sum((L_corrected < 0) | (L_corrected > 255))
    pct_clipped = clipped / L.size

    L_corrected = np.clip(L_corrected, 0, 255).astype(np.uint8)
    lab[:, :, 0] = L_corrected

    img_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img_corrected, pct_clipped


def save_diagnostics(filename: str, raw_profile, smoothed_profile, gain, target,
                     diagnostics_dir: str):
    """Save a diagnostic plot showing the luminance profile and gain curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    rows = np.arange(len(raw_profile))
    ax1.plot(rows, raw_profile, alpha=0.4, label="Raw row-mean L")
    ax1.plot(rows, smoothed_profile, linewidth=2, label="Smoothed profile")
    ax1.axhline(target, color="red", linestyle="--", label=f"Target L={target:.1f}")
    ax1.set_ylabel("Luminance (L)")
    ax1.legend()
    ax1.set_title(filename)

    ax2.plot(rows, gain, linewidth=2, color="green", label="Gain")
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Gain")
    ax2.set_xlabel("Row (top → bottom)")
    ax2.legend()

    plt.tight_layout()
    out_path = Path(diagnostics_dir) / f"{Path(filename).stem}_diag.png"
    plt.savefig(str(out_path), dpi=100)
    plt.close(fig)


def process_image(img_path: Path, output_dir: Path, smooth_divisor: int,
                  gain_min: float, gain_max: float, jpeg_quality: int,
                  diagnostics: bool, diagnostics_dir: str):
    """Process a single image: estimate gradient, correct, save.

    Returns:
        Dict with per-image statistics.
    """
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"filename": img_path.name, "error": "failed to read"}

    H, W = img_bgr.shape[:2]

    # Extract L channel for profile estimation
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float64)

    raw_profile, smoothed_profile = compute_vertical_profile(L, smooth_divisor)
    gain, target = compute_gain(smoothed_profile, gain_min, gain_max)
    img_corrected, pct_clipped = apply_correction(img_bgr, gain)

    # Save corrected image
    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), img_corrected,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    if diagnostics:
        save_diagnostics(img_path.name, raw_profile, smoothed_profile, gain,
                         target, diagnostics_dir)

    # Compute stats
    lab_after = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2LAB)
    mean_L_before = float(L.mean())
    mean_L_after = float(lab_after[:, :, 0].astype(np.float64).mean())

    return {
        "filename": img_path.name,
        "height": H,
        "width": W,
        "mean_L_before": round(mean_L_before, 2),
        "mean_L_after": round(mean_L_after, 2),
        "target_L": round(target, 2),
        "gain_min_actual": round(float(gain.min()), 4),
        "gain_max_actual": round(float(gain.max()), 4),
        "pct_clipped": round(pct_clipped * 100, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Normalize vertical lighting gradients in bark texture images."
    )
    parser.add_argument("--input_dir", type=str,
                        default="D:/jk/dataset_barkseg/images",
                        help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str,
                        default="D:/jk/dataset_barkseg/images-uniform-light",
                        help="Directory for corrected output images.")
    parser.add_argument("--smooth_divisor", type=int, default=8,
                        help="Gaussian sigma = image_height // smooth_divisor (default: 8).")
    parser.add_argument("--gain_min", type=float, default=0.5,
                        help="Minimum per-row gain (default: 0.5).")
    parser.add_argument("--gain_max", type=float, default=2.0,
                        help="Maximum per-row gain (default: 2.0).")
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="JPEG save quality (default: 95).")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Save per-image diagnostic plots.")
    parser.add_argument("--diagnostics_dir", type=str,
                        default="./diagnostics",
                        help="Directory for diagnostic plots (default: ./diagnostics).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.diagnostics:
        Path(args.diagnostics_dir).mkdir(parents=True, exist_ok=True)

    # Collect all image files
    extensions = ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_dir.glob(ext))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Settings: smooth_divisor={args.smooth_divisor}, "
          f"gain=[{args.gain_min}, {args.gain_max}], quality={args.jpeg_quality}")
    print()

    all_stats = []
    for img_path in tqdm(image_paths, desc="Normalizing lighting"):
        stats = process_image(
            img_path, output_dir, args.smooth_divisor,
            args.gain_min, args.gain_max, args.jpeg_quality,
            args.diagnostics, args.diagnostics_dir,
        )
        all_stats.append(stats)

        if stats.get("pct_clipped", 0) > 1.0:
            tqdm.write(f"  WARNING: {stats['filename']} has {stats['pct_clipped']:.2f}% clipped pixels")
        if stats.get("error"):
            tqdm.write(f"  ERROR: {stats['filename']}: {stats['error']}")

    # Print summary
    valid = [s for s in all_stats if "error" not in s]
    if valid:
        avg_clip = np.mean([s["pct_clipped"] for s in valid])
        avg_gain_min = np.mean([s["gain_min_actual"] for s in valid])
        avg_gain_max = np.mean([s["gain_max_actual"] for s in valid])
        print(f"\nDone. Processed {len(valid)}/{len(all_stats)} images successfully.")
        print(f"  Avg clipped pixels: {avg_clip:.4f}%")
        print(f"  Avg gain range: [{avg_gain_min:.4f}, {avg_gain_max:.4f}]")

    # Write stats CSV
    csv_path = output_dir / "stats.csv"
    if valid:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=valid[0].keys())
            writer.writeheader()
            writer.writerows(valid)
        print(f"  Stats saved to {csv_path}")


if __name__ == "__main__":
    main()
