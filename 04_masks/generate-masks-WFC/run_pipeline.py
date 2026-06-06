#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json

from wfc_pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research-grade WFC mask pipeline")
    p.add_argument("--dataset-dir", type=Path, default=Path("masks-unwrap"), help="Directory with PNG masks")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output artifact directory")
    p.add_argument("--pattern-size", type=int, default=3, help="WFC overlapping pattern size")
    p.add_argument("--target-size", type=int, default=100, help="Generated mask size (H=W=target-size)")
    p.add_argument(
        "--tile-mode",
        type=str,
        default="none",
        choices=["none", "torus", "cylinder_x", "cylinder_y"],
        help="Boundary mode for tileable outputs: none|torus|cylinder_x|cylinder_y",
    )
    p.add_argument("--num-generations", type=int, default=3, help="Number of generated masks")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max-restarts", type=int, default=20, help="WFC restarts per generated sample")
    p.add_argument("--snapshot-interval", type=int, default=150, help="Save WFC snapshot every N collapse steps")
    p.add_argument("--min-pattern-weight", type=int, default=2, help="Keep patterns with frequency >= this value")
    p.add_argument("--max-patterns", type=int, default=512, help="Keep top-K patterns by frequency")
    p.add_argument(
        "--augment-symmetry",
        action="store_true",
        default=True,
        help="Use rotation/flip augmentation during pattern extraction (default: enabled)",
    )
    p.add_argument(
        "--no-augment-symmetry",
        dest="augment_symmetry",
        action="store_false",
        help="Disable rotation/flip augmentation",
    )
    p.add_argument(
        "--phases",
        type=str,
        default="all",
        help="Comma-separated phases: all|phase1|phase2|phase3|phase4|phase5",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        pattern_size=args.pattern_size,
        target_size=args.target_size,
        tile_mode=args.tile_mode,
        num_generations=args.num_generations,
        seed=args.seed,
        max_restarts=args.max_restarts,
        snapshot_interval=args.snapshot_interval,
        augment_symmetry=args.augment_symmetry,
        min_pattern_weight=args.min_pattern_weight,
        max_patterns=args.max_patterns,
    )

    phases = [x.strip() for x in args.phases.split(",") if x.strip()]
    report = run_pipeline(config=config, phases=phases)

    print("Pipeline finished.")
    print(json.dumps(report, indent=2))
    print(f"Artifacts saved under: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
