#!/usr/bin/env python3
"""run.py – Entry point for the WFC bark-mask research pipeline.

Usage examples
--------------
  # Full pipeline with recommended defaults (reweighting + guidance)
  python run.py

  # Only run phases 1–3 (fast: no generation)
  python run.py --phases 1,2,3

  # Generate 5 masks, torus mode, explicit output dir
  python run.py --tile-mode torus --n-generations 5 --output-dir outputs_torus

  # Larger pattern size for bigger blob structure (class 2)
  python run.py --pattern-size 5

  # Plain WFC without any guidance or reweighting (baseline comparison)
  python run.py --no-reweight-patterns --guidance-strength 0

  # Strong guidance only (no reweighting)
  python run.py --no-reweight-patterns --guidance-strength 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import Config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WFC mask generation research pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O ---
    p.add_argument("--dataset-dir", type=Path, default=Path("masks-unwrap"),
                   help="Directory containing PNG mask files")
    p.add_argument("--output-dir",  type=Path, default=Path("outputs"),
                   help="Directory for all output artefacts")

    # --- Phases ---
    p.add_argument("--phases", type=str, default="all",
                   help="Comma-separated list of phases to run: "
                        "1/dataset, 2/stats, 3/patterns, 4/generate, 5/compare, or 'all'")

    # --- Pattern extraction ---
    p.add_argument("--pattern-size",  type=int, default=3,
                   help="Side length of WFC tile patterns (n×n). "
                        "Use 5 for larger blobs (class 2). "
                        "max-patterns auto-scales: 3→512, 5→1024, 7→2048.")
    p.add_argument("--min-weight",    type=int, default=2,
                   help="Discard patterns seen fewer times than this")
    p.add_argument("--max-patterns",  type=int, default=0,
                   help="Keep only the top-K most frequent patterns "
                        "(0 = auto: 512 for n=3, 1024 for n=5, 2048 for n=7)")
    p.add_argument("--no-symmetry",   action="store_true",
                   help="Disable 8-fold dihedral augmentation during extraction")

    # --- Generation ---
    p.add_argument("--target-size",    type=int, default=100,
                   help="Output mask size (height = width = target-size)")
    p.add_argument("--tile-mode",      type=str, default="none",
                   choices=["none", "torus", "cylinder_x", "cylinder_y"],
                   help="Boundary mode for tiling")
    p.add_argument("--n-generations", type=int, default=3,
                   help="Number of masks to generate")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--max-restarts",   type=int, default=20,
                   help="WFC restarts per generation before giving up")
    p.add_argument("--snapshot-interval", type=int, default=200,
                   help="Save a WFC entropy/preview snapshot every N collapse steps")
    p.add_argument("--log-interval",   type=int, default=500,
                   help="Print WFC progress every N steps (0 = silent)")

    # --- Distribution matching ---
    p.add_argument("--guidance-strength", type=float, default=3.0,
                   help="Log-ratio guidance strength β (0=off, 1=Dirichlet IW, 3–5=strong). "
                        "Steers per-step collapse probabilities toward real-data fracs.")
    p.add_argument("--no-guide-to-real-fracs", action="store_true",
                   help="Disable auto-computation of target class fracs from dataset")
    p.add_argument("--reweight-patterns", dest="reweight_patterns",
                   action="store_true", default=True,
                   help="Reweight pattern library so expected class fracs match real data "
                        "(RECOMMENDED — the primary fix for background over-generation)")
    p.add_argument("--no-reweight-patterns", dest="reweight_patterns",
                   action="store_false",
                   help="Disable pattern reweighting (baseline / ablation)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        pattern_size=args.pattern_size,
        target_size=args.target_size,
        tile_mode=args.tile_mode,
        n_generations=args.n_generations,
        seed=args.seed,
        max_restarts=args.max_restarts,
        snapshot_interval=args.snapshot_interval,
        log_interval=args.log_interval,
        augment_symmetry=not args.no_symmetry,
        min_pattern_weight=args.min_weight,
        max_patterns=args.max_patterns,
        guidance_strength=args.guidance_strength,
        guide_to_real_fracs=not args.no_guide_to_real_fracs,
        reweight_patterns=args.reweight_patterns,
    )

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    if not phases:
        phases = ["all"]

    # Print config summary
    sep = "=" * 62
    print(sep)
    print("WFC Mask Pipeline")
    print(sep)
    print(f"  Dataset:         {cfg.dataset_dir.resolve()}")
    print(f"  Output:          {cfg.output_dir.resolve()}")
    print(f"  Phases:          {phases}")
    print(f"  Pattern size:    {cfg.pattern_size}×{cfg.pattern_size}  "
          f"(max_patterns={cfg.effective_max_patterns()}, "
          f"min_weight={cfg.min_pattern_weight}, "
          f"symmetry_aug={cfg.augment_symmetry})")
    print(f"  Target size:     {cfg.target_size}×{cfg.target_size}  "
          f"tile_mode={cfg.tile_mode}")
    print(f"  Generations:     {cfg.n_generations}  "
          f"(seed={cfg.seed}, max_restarts={cfg.max_restarts})")
    print(f"  Distribution:")
    print(f"    reweight_patterns   = {cfg.reweight_patterns}  "
          f"(recommended: True)")
    print(f"    guidance_strength β = {cfg.guidance_strength}  "
          f"(0=off, 1=Dirichlet-IW, 3–5=strong)")
    print(f"    guide_to_real_fracs = {cfg.guide_to_real_fracs}")
    print(sep)

    run_pipeline(cfg, phases)


if __name__ == "__main__":
    main()
