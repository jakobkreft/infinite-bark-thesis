"""
CLI entry point for MRF mask synthesis and inpainting.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory so we can import src as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import BarkMaskDataset
from src.synthesis import synthesize_mask
from src.inpainting import inpaint
from src.evaluate import compare_with_training, print_report
from src.scale2x import upscale_epx_to_factor, compute_aligned_dims, center_crop
from src.model_cache import load_or_build_model, DEFAULT_CACHE_DIR


# (tileable_v, tileable_h) per --tileable choice. "horizontal" wraps
# left/right (tiles side by side); "vertical" wraps top/bottom.
TILEABLE_MODES = {
    "none": (False, False),
    "horizontal": (False, True),
    "vertical": (True, False),
    "both": (True, True),
}


def parse_ratio(s):
    """Parse a comma-separated ratio string like '0.8,0.1,0.1'."""
    try:
        parts = [float(x.strip()) for x in s.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Ratio must contain numeric comma-separated values"
        ) from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Ratio must have 3 comma-separated values")
    if any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError("Ratio values must be non-negative")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("Ratio values must sum to a positive number")
    return [p / total for p in parts]  # normalize


def _add_data_and_cache_args(p):
    p.add_argument("--data", default="masks-dataset", help="Path to mask dataset folder")
    p.add_argument("--radius", type=int, default=3, help="Neighborhood radius")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                   help="Directory for the cached MRF model (class frequencies, "
                        "pairwise potentials, and neighborhood tables). A valid "
                        "cache for the same --data contents and --radius is "
                        "reused instead of rebuilt.")
    p.add_argument("--no-cache", action="store_true",
                   help="Don't read or write the model cache; always rebuild in memory")
    p.add_argument("--refresh-cache", action="store_true",
                   help="Rebuild the model cache even if a valid one already exists")


def _add_tileable_arg(p, default="both"):
    p.add_argument("--tileable", choices=list(TILEABLE_MODES), default=default,
                   help="Which axes wrap toroidally so the output tiles seamlessly: "
                        "'horizontal' (left/right wrap only), 'vertical' (top/bottom "
                        "wrap only), 'both' (default, seamless in every direction), "
                        "or 'none' (bounded, no wrap).")


def main():
    parser = argparse.ArgumentParser(
        description="MRF-based semantic mask synthesis for log bark textures"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- synthesize ---
    syn = subparsers.add_parser("synthesize", help="Generate a new mask from scratch")
    _add_data_and_cache_args(syn)
    syn.add_argument("--height", type=int, default=256, help="Output height")
    syn.add_argument("--width", type=int, default=256, help="Output width")
    syn.add_argument("--refine", type=int, default=3, help="Number of Gibbs refinement passes")
    syn.add_argument("--no-multiscale", action="store_true", help="Disable multiscale init")
    syn.add_argument("--ratio", type=parse_ratio, default=None,
                     help="Target class proportions, e.g. '0.7,0.05,0.25'")
    syn.add_argument("--lambda-ratio", type=float, default=1.0,
                     help="Weight for the global count penalty used by --ratio")
    syn.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    syn.add_argument("--k-fallback", type=int, default=11, help="KDTree fallback k")
    syn.add_argument("--seed", type=int, default=None,
                     help="Random seed. With --count > 1, image i uses seed + i - 1.")
    _add_tileable_arg(syn)
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
    syn.add_argument("--count", "-n", type=int, default=1,
                     help="Number of masks to synthesize in this one process, reusing "
                          "the same loaded/cached model instead of paying its startup "
                          "cost per image (e.g. instead of a shell loop of separate "
                          "invocations). If --output contains '{i}' it is used as a "
                          "format placeholder (1-based); otherwise an index is appended "
                          "before the extension when --count > 1.")
    syn.add_argument("--output", "-o", default="outputs/synthesized.png",
                     help="Output PNG path")
    syn.add_argument("--evaluate", action="store_true", help="Run evaluation after synthesis")

    # --- inpaint ---
    inp = subparsers.add_parser("inpaint", help="Fill holes in an existing mask")
    inp.add_argument("input", help="Input PNG mask with holes (unknown pixels = 255)")
    _add_data_and_cache_args(inp)
    inp.add_argument("--unknown-value", type=int, default=255,
                     help="Pixel value marking unknown regions in input")
    inp.add_argument("--ratio", type=parse_ratio, default=None,
                     help="Target class proportions")
    inp.add_argument("--lambda-ratio", type=float, default=1.0,
                     help="Weight for the global count penalty used by --ratio")
    inp.add_argument("--no-multiscale", action="store_true",
                     help="Disable coarse-to-fine fill; fill the hole at full "
                          "resolution directly. Coarse-to-fine (the default) "
                          "gives large holes plausible large-scale structure "
                          "instead of noise.")
    inp.add_argument("--refine", type=int, default=3, help="Gibbs refinement passes")
    inp.add_argument("--temperature", type=float, default=1.0)
    inp.add_argument("--k-fallback", type=int, default=11)
    inp.add_argument("--seed", type=int, default=None, help="Random seed")
    _add_tileable_arg(inp)
    inp.add_argument("--pixel-scale", type=int, default=1,
                     help="Each mask pixel becomes a pixel_scale x pixel_scale "
                          "block in the saved output via EPX upscaling.")
    inp.add_argument("--output", "-o", default="outputs/inpainted.png")
    inp.add_argument("--evaluate", action="store_true")

    # --- precompute ---
    pre = subparsers.add_parser(
        "precompute",
        help="Build and cache the MRF model ahead of time (class frequencies, "
             "pairwise potentials, neighborhood tables), so a later synthesize/"
             "inpaint call just loads it instead of rebuilding it.",
    )
    pre.add_argument("--data", default="masks-dataset", help="Path to mask dataset folder")
    pre.add_argument("--radius", type=int, default=3, help="Neighborhood radius")
    pre.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    pre.add_argument("--refresh-cache", action="store_true",
                     help="Rebuild even if a valid cache already exists")

    # --- evaluate ---
    evl = subparsers.add_parser("evaluate", help="Evaluate a generated mask")
    evl.add_argument("input", help="PNG mask to evaluate")
    evl.add_argument("--data", default="masks-dataset", help="Path to mask dataset folder")

    # --- stats ---
    sta = subparsers.add_parser("stats", help="Print dataset statistics")
    sta.add_argument("--data", default="masks-dataset", help="Path to mask dataset folder")

    args = parser.parse_args()

    if args.command == "synthesize":
        tileable_v, tileable_h = TILEABLE_MODES[args.tileable]

        trained_model, dataset = load_or_build_model(
            args.data, args.radius,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            refresh=args.refresh_cache,
        )

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

        verbose = args.count == 1
        indices = range(1, args.count + 1)
        if args.count > 1:
            indices = tqdm(indices, desc="Synthesizing masks")

        for i in indices:
            seed_i = None if args.seed is None else args.seed + i - 1
            result = synthesize_mask(
                trained_model,
                height=h_synth,
                width=w_synth,
                n_refine=args.refine,
                multiscale=not args.no_multiscale,
                target_ratio=args.ratio,
                lambda_ratio=args.lambda_ratio,
                temperature=args.temperature,
                k_fallback=args.k_fallback,
                seed=seed_i,
                tileable_v=tileable_v,
                tileable_h=tileable_h,
                verbose=verbose,
            )
            if args.evaluate:
                if dataset is None:
                    dataset = BarkMaskDataset(args.data)
                report = compare_with_training(result, dataset)
                print_report(report)

            if args.pixel_scale > 1:
                result = upscale_epx_to_factor(result, args.pixel_scale, verbose=verbose)
            if result.shape != (h_final, w_final):
                result = center_crop(result, h_final, w_final)

            out_path = _output_path_for_index(args.output, i, args.count)
            _save_mask(result, out_path, verbose=verbose)
            if verbose:
                print(f"Saved synthesized mask to {out_path}")

        if args.count > 1:
            print(f"Saved {args.count} synthesized masks "
                  f"(pattern: {_output_path_for_index(args.output, 1, args.count)} ...)")

    elif args.command == "inpaint":
        tileable_v, tileable_h = TILEABLE_MODES[args.tileable]

        trained_model, dataset = load_or_build_model(
            args.data, args.radius,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            refresh=args.refresh_cache,
        )

        # .convert("L") drops any alpha/color channels (e.g. LA or RGBA
        # exports from image editors) down to a single label channel.
        # Compare against --unknown-value (typically 255) before narrowing to
        # int8, since int8 can't hold values > 127 and would silently wrap
        # e.g. 255 -> -1 or 248 -> -8, breaking the equality check below.
        raw_mask = np.array(Image.open(args.input).convert("L")).astype(np.int16)
        input_mask = np.where(raw_mask == args.unknown_value, -1, raw_mask).astype(np.int8)

        result = inpaint(
            input_mask,
            trained_model,
            unknown_value=-1,
            target_ratio=args.ratio,
            lambda_ratio=args.lambda_ratio,
            tileable_v=tileable_v,
            tileable_h=tileable_h,
            multiscale=not args.no_multiscale,
            n_refine=args.refine,
            temperature=args.temperature,
            k_fallback=args.k_fallback,
            seed=args.seed,
        )
        if args.evaluate:
            if dataset is None:
                dataset = BarkMaskDataset(args.data)
            report = compare_with_training(result, dataset)
            print_report(report)

        if args.pixel_scale > 1:
            result = upscale_epx_to_factor(result, args.pixel_scale)
        _save_mask(result, args.output)
        print(f"Saved inpainted mask to {args.output}")

    elif args.command == "precompute":
        load_or_build_model(
            args.data, args.radius,
            cache_dir=args.cache_dir,
            use_cache=True,
            refresh=args.refresh_cache,
        )

    elif args.command == "evaluate":
        dataset = BarkMaskDataset(args.data)
        mask = np.array(Image.open(args.input).convert("L")).astype(np.int8)
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


def _output_path_for_index(template, i, count):
    """Expand an --output template for the i-th (1-based) of `count` images.

    If the template contains '{' it's used as a str.format placeholder
    (e.g. "outputs/mask_{i:04d}.png"); otherwise an index is appended before
    the extension (e.g. "outputs/mask.png" -> "outputs/mask_0001.png").
    """
    if count <= 1:
        return template
    if "{" in template:
        return template.format(i=i)
    base, ext = os.path.splitext(template)
    width = len(str(count))
    return f"{base}_{i:0{width}d}{ext}"


def _save_mask(mask, path, verbose=True):
    """Save label mask as uint8 PNG and a colored 1024x1024 visualization."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    img.save(path)
    _save_viz(mask, path, verbose=verbose)


# Class colors: 0=black (ozadje), 1=yellow (slepice), 2=blue (mehanske poskodbe)
VIZ_COLORS = np.array([
    [0, 0, 0],        # 0: black
    [230, 190, 40],    # 1: yellowish
    [50, 100, 210],    # 2: blueish
], dtype=np.uint8)
VIZ_MIN_SIZE = 1024


def _save_viz(mask, mask_path, verbose=True):
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
    if verbose:
        print(f"Saved visualization to {viz_path} ({viz.size[0]}x{viz.size[1]})")


if __name__ == "__main__":
    main()
