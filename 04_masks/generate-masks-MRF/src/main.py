"""
CLI entry point for MRF mask synthesis and inpainting.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# Add parent directory so we can import src as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import BarkMaskDataset
from src.synthesis import synthesize_mask
from src.inpainting import inpaint
from src.evaluate import compare_with_training, print_report
from src.scale2x import upscale_epx_to_factor, compute_aligned_dims, center_crop


def parse_ratio(s):
    """Parse a comma-separated ratio string like '0.8,0.1,0.1'."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Ratio must have 3 comma-separated values")
    total = sum(parts)
    return [p / total for p in parts]  # normalize


def main():
    parser = argparse.ArgumentParser(
        description="MRF-based semantic mask synthesis for log bark textures"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- synthesize ---
    syn = subparsers.add_parser("synthesize", help="Generate a new mask from scratch")
    syn.add_argument("--data", default="masks-dataset", help="Path to mask dataset folder")
    syn.add_argument("--height", type=int, default=256, help="Output height")
    syn.add_argument("--width", type=int, default=256, help="Output width")
    syn.add_argument("--radius", type=int, default=3, help="Neighborhood radius")
    syn.add_argument("--refine", type=int, default=3, help="Number of Gibbs refinement passes")
    syn.add_argument("--no-multiscale", action="store_true", help="Disable multiscale init")
    syn.add_argument("--ratio", type=parse_ratio, default=None,
                     help="Target class proportions, e.g. '0.7,0.05,0.25'")
    syn.add_argument("--lambda-ratio", type=float, default=1.0,
                     help="Weight for proportion bias")
    syn.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    syn.add_argument("--k-fallback", type=int, default=11, help="KDTree fallback k")
    syn.add_argument("--seed", type=int, default=None, help="Random seed")
    syn.add_argument("--pixel-scale", type=int, default=1,
                     help="Each mask pixel becomes a pixel_scale x pixel_scale "
                          "block in the saved output, using EPX (Scale2x) rounding. "
                          "Default 1 = no upscale.")
    syn.add_argument("--target-multiple", type=int, default=512,
                     help="Final output size is snapped to a multiple of this value "
                          "(per side). Set to 1 to disable.")
    syn.add_argument("--size-mode", choices=["ceil", "floor", "off"], default="ceil",
                     help="How to snap to target-multiple: ceil (synth slightly larger "
                          "then center-crop), floor (crop down), or off.")
    syn.add_argument("--output", "-o", default="outputs/synthesized.png",
                     help="Output PNG path")
    syn.add_argument("--evaluate", action="store_true", help="Run evaluation after synthesis")

    # --- inpaint ---
    inp = subparsers.add_parser("inpaint", help="Fill holes in an existing mask")
    inp.add_argument("input", help="Input PNG mask with holes (unknown pixels = 255)")
    inp.add_argument("--data", default="masks-dataset", help="Path to mask dataset folder")
    inp.add_argument("--unknown-value", type=int, default=255,
                     help="Pixel value marking unknown regions in input")
    inp.add_argument("--pairwise-scale", type=float, default=1.0,
                     help="Pairwise cost scaling")
    inp.add_argument("--ratio", type=parse_ratio, default=None,
                     help="Target class proportions")
    inp.add_argument("--lambda-ratio", type=float, default=1.0)
    inp.add_argument("--refine", type=int, default=2, help="Gibbs refinement passes")
    inp.add_argument("--radius", type=int, default=3, help="Neighborhood radius")
    inp.add_argument("--temperature", type=float, default=1.0)
    inp.add_argument("--k-fallback", type=int, default=11)
    inp.add_argument("--pixel-scale", type=int, default=1,
                     help="Each mask pixel becomes a pixel_scale x pixel_scale "
                          "block in the saved output via EPX upscaling.")
    inp.add_argument("--output", "-o", default="outputs/inpainted.png")
    inp.add_argument("--evaluate", action="store_true")

    # --- evaluate ---
    evl = subparsers.add_parser("evaluate", help="Evaluate a generated mask")
    evl.add_argument("input", help="PNG mask to evaluate")
    evl.add_argument("--data", default="masks-dataset", help="Path to mask dataset folder")

    # --- stats ---
    sta = subparsers.add_parser("stats", help="Print dataset statistics")
    sta.add_argument("--data", default="masks-dataset", help="Path to mask dataset folder")

    args = parser.parse_args()

    if args.command == "synthesize":
        dataset = BarkMaskDataset(args.data)

        h_synth, w_synth, h_final, w_final = compute_aligned_dims(
            args.height, args.width, args.pixel_scale,
            args.target_multiple, args.size_mode,
        )
        if (h_synth, w_synth) != (args.height, args.width):
            print(
                f"Size alignment ({args.size_mode}, multiple={args.target_multiple}): "
                f"synth {h_synth}x{w_synth} -> upscale x{args.pixel_scale} -> "
                f"crop to {h_final}x{w_final}"
            )

        result = synthesize_mask(
            dataset,
            height=h_synth,
            width=w_synth,
            radius=args.radius,
            n_refine=args.refine,
            multiscale=not args.no_multiscale,
            target_ratio=args.ratio,
            lambda_ratio=args.lambda_ratio,
            temperature=args.temperature,
            k_fallback=args.k_fallback,
            seed=args.seed,
        )
        if args.evaluate:
            report = compare_with_training(result, dataset)
            print_report(report)

        if args.pixel_scale > 1:
            result = upscale_epx_to_factor(result, args.pixel_scale)
        if result.shape != (h_final, w_final):
            result = center_crop(result, h_final, w_final)
        _save_mask(result, args.output)
        print(f"Saved synthesized mask to {args.output}")

    elif args.command == "inpaint":
        dataset = BarkMaskDataset(args.data)
        input_mask = np.array(Image.open(args.input)).astype(np.int8)
        # Map unknown value to -1
        input_mask[input_mask == args.unknown_value] = -1
        n_unknown = np.sum(input_mask == -1)
        print(f"Inpainting {n_unknown} unknown pixels...")

        result = inpaint(
            input_mask,
            dataset,
            unknown_value=-1,
            pairwise_scale=args.pairwise_scale,
            target_ratio=args.ratio,
            lambda_ratio=args.lambda_ratio,
            toroidal=True,
            n_refine=args.refine,
            radius=args.radius,
            temperature=args.temperature,
            k_fallback=args.k_fallback,
        )
        if args.evaluate:
            report = compare_with_training(result, dataset)
            print_report(report)

        if args.pixel_scale > 1:
            result = upscale_epx_to_factor(result, args.pixel_scale)
        _save_mask(result, args.output)
        print(f"Saved inpainted mask to {args.output}")

    elif args.command == "evaluate":
        dataset = BarkMaskDataset(args.data)
        mask = np.array(Image.open(args.input)).astype(np.int8)
        report = compare_with_training(mask, dataset)
        print_report(report)

    elif args.command == "stats":
        dataset = BarkMaskDataset(args.data)
        freqs = dataset.estimate_class_frequencies()
        pairwise = dataset.estimate_pairwise_potentials()
        print("\nPairwise potential matrices (-log P):")
        for name, C in pairwise.items():
            print(f"\n  {name}:")
            for row in C:
                print(f"    [{', '.join(f'{v:.3f}' for v in row)}]")

        # Show some training mask component stats
        from src.evaluate import connected_component_stats
        all_stats = [connected_component_stats(m) for m in dataset.masks]
        print("\nTraining mask connected components (averaged):")
        for label_name in ["ozadje", "slepice", "mehanske_poskodbe"]:
            counts = [s[label_name]["count"] for s in all_stats]
            areas = [s[label_name]["mean_area"] for s in all_stats if s[label_name]["count"] > 0]
            print(f"  {label_name}: avg_count={np.mean(counts):.1f}, "
                  f"avg_mean_area={np.mean(areas) if areas else 0:.1f}")


def _save_mask(mask, path):
    """Save label mask as uint8 PNG and a colored 1024x1024 visualization."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    img.save(path)
    _save_viz(mask, path)


# Class colors: 0=black (ozadje), 1=yellow (slepice), 2=blue (mehanske poskodbe)
VIZ_COLORS = np.array([
    [0, 0, 0],        # 0: black
    [230, 190, 40],    # 1: yellowish
    [50, 100, 210],    # 2: blueish
], dtype=np.uint8)
VIZ_MIN_SIZE = 1024


def _save_viz(mask, mask_path):
    """Save a colored visualization next to the mask.

    If the mask is already large (>= VIZ_MIN_SIZE per side) the viz is
    rendered at the mask's native resolution so EPX-rounded edges remain
    visible. Smaller masks are NN-upscaled to VIZ_MIN_SIZE for legibility.
    """
    h, w = mask.shape
    rgb = VIZ_COLORS[mask.astype(np.uint8)]  # (H, W, 3)
    viz = Image.fromarray(rgb, mode="RGB")

    if max(h, w) < VIZ_MIN_SIZE:
        scale = VIZ_MIN_SIZE / max(h, w)
        viz = viz.resize(
            (int(round(w * scale)), int(round(h * scale))),
            resample=Image.NEAREST,
        )

    base, ext = os.path.splitext(mask_path)
    viz_path = f"{base}_viz.png"
    viz.save(viz_path)
    print(f"Saved visualization to {viz_path} ({viz.size[0]}x{viz.size[1]})")


if __name__ == "__main__":
    main()
