"""pipeline.py – Phase-based WFC research pipeline.

Phases
------
1  Dataset loading     → phase1_dataset/
2  Statistics          → phase2_stats/
3  Pattern extraction  → phase3_patterns/
4  WFC generation      → phase4_generation/
5  Comparison          → phase5_comparison/

Each phase saves:
- PNG visualisations (no titles — add them in LaTeX)
- JSON / CSV data files for reproducibility
- Intermediate .npz artefacts so individual phases can be re-run

Distribution matching
---------------------
To produce generated masks whose class distribution matches real data, two
mechanisms work together:

  1. Pattern reweighting (cfg.reweight_patterns=True):
     The pattern library weights are rescaled using multiplicative importance
     ratios so that the expected class distribution from the library matches
     the real-data target.  This is the primary fix for background dominance.

  2. Per-step log-ratio guidance (cfg.guidance_strength > 0):
     During WFC collapse, pattern probabilities are multiplied by
     exp(β · log(target/current)) per class.  Complements reweighting.

Recommended settings for matching real-data statistics:
  --reweight-patterns --guidance-strength 3

For larger blob structures (class 2 median=15 px in real data), use:
  --pattern-size 5 --max-patterns 1024 --reweight-patterns
"""
from __future__ import annotations

import json
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from dataset import MaskSample, load_dataset, save_manifest
from stats   import compute_stats, DatasetStats, distribution_summary, CLASS_NAMES
from patterns import PatternLibrary, extract_patterns
from wfc     import WFCGenerator, WFCResult
import viz


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    dataset_dir:   Path = Path("masks-unwrap")
    output_dir:    Path = Path("outputs")
    pattern_size:  int  = 3
    target_size:   int  = 100
    tile_mode:     str  = "none"       # none | torus | cylinder_x | cylinder_y
    n_generations: int  = 3
    seed:          int  = 42
    max_restarts:  int  = 20
    snapshot_interval: int = 200
    augment_symmetry:  bool = True
    min_pattern_weight: int = 2
    max_patterns:  int  = 0            # 0 = auto-scale by pattern_size
    # Reward-guided generation
    guidance_strength:    float = 3.0   # 0 = plain WFC; 1 = Dirichlet IW; 3–5 useful
    guide_to_real_fracs:  bool  = True  # use real-data class fracs as target
    reweight_patterns:    bool  = True  # reweight library before generation
    log_interval:         int   = 500   # WFC progress print interval (0 = silent)

    def effective_max_patterns(self) -> int:
        """Auto-scale max_patterns with pattern_size if max_patterns=0."""
        if self.max_patterns > 0:
            return self.max_patterns
        # Larger patterns → more unique patterns → need bigger library
        return {3: 512, 5: 1024, 7: 2048}.get(self.pattern_size,
                512 * max(1, self.pattern_size - 1))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["dataset_dir"] = str(self.dataset_dir)
        d["output_dir"]  = str(self.output_dir)
        with path.open("w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def load(path: Path) -> "Config":
        with path.open() as f:
            d = json.load(f)
        d["dataset_dir"] = Path(d["dataset_dir"])
        d["output_dir"]  = Path(d["output_dir"])
        # handle fields added after old configs were saved
        d.setdefault("reweight_patterns", True)
        d.setdefault("log_interval", 500)
        d.setdefault("guidance_strength", 3.0)
        d.setdefault("guide_to_real_fracs", True)
        return Config(**d)


# ---------------------------------------------------------------------------
# Phase 1 – Dataset
# ---------------------------------------------------------------------------

def phase1_dataset(cfg: Config) -> list[MaskSample]:
    """Load dataset, save manifest, save gallery."""
    print("[Phase 1] Loading dataset …")
    out = cfg.output_dir / "phase1_dataset"
    out.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(cfg.dataset_dir)
    save_manifest(samples, out)

    print(f"  {len(samples)} masks loaded")

    viz.save_dataset_gallery(
        masks=[s.mask for s in samples],
        names=[s.name for s in samples],
        path=out / "gallery_all.png",
        cols=10, max_items=len(samples),
    )
    print(f"  Gallery → {out / 'gallery_all.png'}")

    return samples


# ---------------------------------------------------------------------------
# Phase 2 – Statistics
# ---------------------------------------------------------------------------

def phase2_stats(cfg: Config, samples: list[MaskSample]) -> DatasetStats:
    """Compute and visualise dataset statistics."""
    print("[Phase 2] Computing statistics …")
    out = cfg.output_dir / "phase2_stats"
    out.mkdir(parents=True, exist_ok=True)

    stats = compute_stats(samples)

    # --- CSV export ---
    _save_per_image_csv(stats, out / "per_image_stats.csv")

    # --- JSON summary ---
    summary = {
        "n_images": len(stats.per_image),
        "total_pixels": stats.total_pixels,
        "class_pixel_counts": {str(k): v for k, v in stats.class_pixel_counts.items()},
        "class_pixel_fractions": {str(k): v for k, v in stats.class_pixel_fractions.items()},
        "cooccurrence": stats.cooccurrence,
        "component_stats": {
            str(cls): distribution_summary(stats.component_sizes.get(cls, []))
            for cls in (1, 2)
        },
    }
    with (out / "stats_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Visualisations ---
    viz.save_class_bar(
        stats.class_pixel_counts,
        out / "class_pixel_counts.png",
        ylabel="pixel count",
    )
    viz.save_class_bar(
        stats.class_pixel_fractions,
        out / "class_pixel_fractions.png",
        ylabel="fraction of total pixels",
    )
    viz.save_fraction_boxplot(
        stats.class_fraction_per_image,
        out / "class_fraction_per_image_boxplot.png",
    )
    viz.save_cooccurrence_bar(
        stats.cooccurrence,
        out / "class_cooccurrence.png",
    )
    viz.save_adjacency_heatmap(
        stats.adjacency,
        out / "adjacency_normalised.png",
        normalize_rows=True,
    )
    viz.save_adjacency_heatmap(
        stats.adjacency,
        out / "adjacency_counts.png",
        normalize_rows=False,
    )
    for cls in (1, 2):
        viz.save_component_histogram(
            stats.component_sizes.get(cls, []),
            out / f"component_sizes_class{cls}.png",
            xlabel=f"component size – class {cls}",
            color=viz.PALETTE_HEX[cls],
        )
    viz.save_size_distribution_bar(
        stats.size_distribution,
        out / "image_size_distribution.png",
    )

    print(f"  Statistics saved to {out}")
    return stats


# ---------------------------------------------------------------------------
# Phase 3 – Patterns
# ---------------------------------------------------------------------------

def phase3_patterns(cfg: Config, samples: list[MaskSample]) -> PatternLibrary:
    """Extract WFC patterns, build compatibility, visualise."""
    print("[Phase 3] Extracting patterns …")
    out = cfg.output_dir / "phase3_patterns"
    out.mkdir(parents=True, exist_ok=True)

    max_pats = cfg.effective_max_patterns()
    if max_pats != cfg.max_patterns and cfg.max_patterns == 0:
        print(f"  Auto max_patterns={max_pats} for pattern_size={cfg.pattern_size}")

    lib = extract_patterns(
        masks=[s.mask for s in samples],
        pattern_size=cfg.pattern_size,
        augment_symmetry=cfg.augment_symmetry,
        min_weight=cfg.min_pattern_weight,
        max_patterns=max_pats,
    )
    lib.save(out)

    P = lib.patterns.shape[0]
    print(f"  {P} patterns (size {cfg.pattern_size}×{cfg.pattern_size})")

    viz.save_pattern_gallery(
        lib.patterns, lib.weights,
        out / "top_patterns_gallery.png",
        cols=16, max_items=128,
    )
    viz.save_pattern_weight_histogram(
        lib.weights,
        out / "pattern_weight_histogram.png",
    )
    viz.save_compat_density_bar(
        lib.compat,
        out / "compat_density_per_direction.png",
    )
    for d in ("up", "right"):
        M = lib.compat[d]
        n_show = min(128, M.shape[0])
        viz.save_compat_heatmap(
            M[:n_show, :n_show],
            out / f"compat_heatmap_{d}.png",
            direction=d,
        )

    print(f"  Pattern library → {out}")
    return lib


# ---------------------------------------------------------------------------
# Phase 4 – Generation
# ---------------------------------------------------------------------------

def phase4_generate(
    cfg: Config,
    lib: PatternLibrary,
    target_fracs: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Run WFC generation with optional reweighting and guidance."""
    print("[Phase 4] WFC generation …")
    out = cfg.output_dir / "phase4_generation"
    out.mkdir(parents=True, exist_ok=True)

    # --- Build target fraction vector ---
    tf: np.ndarray | None = None
    if target_fracs is not None:
        tf = np.array([target_fracs.get(c, 0.0) for c in range(3)], dtype=np.float64)
        tf /= tf.sum()
        print(f"  Target fracs: " + "  ".join(f"c{c}={tf[c]:.3f}" for c in range(3)))
    elif cfg.guidance_strength > 0.0:
        print("  WARNING: guidance_strength > 0 but no target fracs provided; "
              "using plain WFC.")

    # --- Pattern reweighting (before guidance, applied to library weights) ---
    lib_for_gen = lib
    if cfg.reweight_patterns and tf is not None:
        print("  Reweighting pattern library for target distribution …")
        lib_for_gen = lib.reweight_for_target(tf)

        # Visualise the reweighting effect
        pcf = lib.compute_class_fracs(n_classes=3)
        corpus_before = lib.corpus_class_fracs(n_classes=3)
        corpus_after  = lib_for_gen.corpus_class_fracs(n_classes=3)
        viz.save_pattern_reweight_scatter(
            lib.weights, lib_for_gen.weights, pcf,
            out / "pattern_reweight_scatter.png",
        )
        viz.save_pattern_class_fracs_comparison(
            corpus_before, corpus_after, tf,
            out / "pattern_class_fracs_comparison.png",
        )
    elif cfg.guidance_strength > 0.0:
        print("  Skipping reweighting (guide_to_real_fracs disabled or no target fracs).")

    # --- Build generator ---
    gen = WFCGenerator(
        library=lib_for_gen,
        height=cfg.target_size,
        width=cfg.target_size,
        tile_mode=cfg.tile_mode,
        seed=cfg.seed,
        target_fracs=tf,
        guidance_strength=cfg.guidance_strength,
        log_interval=cfg.log_interval,
    )

    if cfg.guidance_strength > 0.0 and tf is not None:
        print(f"  Log-ratio guidance: β={cfg.guidance_strength:.1f}")
    else:
        print("  Guidance: off (plain WFC)")

    generated_masks: list[np.ndarray] = []
    generated_names: list[str] = []
    run_summaries:   list[dict] = []

    for i in range(cfg.n_generations):
        run_name = f"run_{i:03d}"
        run_dir  = out / run_name
        snap_dir = run_dir / "snapshots"
        run_dir.mkdir(parents=True, exist_ok=True)

        gen.seed = cfg.seed + i * 10_000
        print(f"\n  [{run_name}]  seed={gen.seed}  "
              f"size={cfg.target_size}×{cfg.target_size}  "
              f"patterns={lib_for_gen.patterns.shape[0]}")

        # --- Snapshot callback ---
        snap_steps: list[int] = []
        snap_fracs: list[float] = []

        def _snap(step: int, wave: np.ndarray, tag: str, _gen=gen) -> None:
            frac = _gen.collapsed_fraction(wave)
            snap_steps.append(step)
            snap_fracs.append(frac)
            if step == 0 or step % cfg.snapshot_interval == 0 or "done" in tag or "start" in tag:
                snap_dir.mkdir(parents=True, exist_ok=True)
                viz.save_wfc_snapshot(
                    entropy_map=_gen.entropy_map(wave),
                    preview_mask=_gen.preview_from_wave(wave),
                    collapsed_fraction=frac,
                    step=step,
                    restart=int(tag.split("r")[1].split("_")[0]) if tag.startswith("r") else 0,
                    path=snap_dir / f"snap_{step:06d}_{tag}.png",
                )

        result = gen.generate(
            max_restarts=cfg.max_restarts,
            snapshot_interval=cfg.snapshot_interval,
            snapshot_cb=_snap,
        )

        meta = {
            "run": run_name,
            "success": result.success,
            "steps": result.steps,
            "restarts": result.restarts,
            "message": result.message,
            "target_size": [cfg.target_size, cfg.target_size],
            "tile_mode": cfg.tile_mode,
            "pattern_size": lib.pattern_size,
            "n_patterns": int(lib_for_gen.patterns.shape[0]),
            "reweighted": cfg.reweight_patterns and tf is not None,
            "guidance_strength": cfg.guidance_strength,
        }
        with (run_dir / "run_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

        if result.success and result.mask is not None:
            np.save(run_dir / "mask.npy", result.mask)
            Image.fromarray(result.mask, mode="L").save(run_dir / "mask_raw.png")
            viz.save_single_mask(result.mask, run_dir / "mask_coloured.png")

            if snap_steps:
                viz.save_wfc_progress_curve(snap_steps, snap_fracs,
                                             run_dir / "progress_curve.png")

            if tf is not None and result.guidance_history:
                viz.save_guidance_curve(
                    result.guidance_history, tf,
                    run_dir / "guidance_distribution_curve.png",
                )

            generated_masks.append(result.mask)
            generated_names.append(run_name)
        else:
            print(f"  {run_name} FAILED: {result.message}")

        run_summaries.append(meta)

    # Gallery of all successes
    if generated_masks:
        viz.save_generated_gallery(
            generated_masks, generated_names,
            out / "generated_gallery.png",
            cols=min(4, len(generated_masks)),
        )

    with (out / "generation_summary.json").open("w") as f:
        json.dump({
            "n_requested": cfg.n_generations,
            "n_success": len(generated_masks),
            "runs": run_summaries,
        }, f, indent=2)

    print(f"\n  {len(generated_masks)}/{cfg.n_generations} succeeded → {out}")
    return generated_masks, generated_names


# ---------------------------------------------------------------------------
# Phase 5 – Comparison
# ---------------------------------------------------------------------------

def phase5_compare(
    cfg: Config,
    real_stats: DatasetStats,
    gen_masks: list[np.ndarray],
    gen_names: list[str],
) -> None:
    """Compare real and generated mask distributions."""
    if not gen_masks:
        print("[Phase 5] Skipped – no generated masks.")
        return

    print("[Phase 5] Comparing real vs generated …")
    out = cfg.output_dir / "phase5_comparison"
    out.mkdir(parents=True, exist_ok=True)

    from dataset import MaskSample as _MS
    fake_samples = [_MS(name=n, path=Path(), mask=m)
                    for n, m in zip(gen_names, gen_masks)]
    gen_stats = compute_stats(fake_samples)

    # Class fraction comparison
    viz.save_real_vs_generated_bar(
        real_stats.class_pixel_fractions,
        gen_stats.class_pixel_fractions,
        out / "class_fractions_comparison.png",
    )

    # Adjacency heatmaps
    viz.save_adjacency_heatmap(gen_stats.adjacency,
                                out / "generated_adjacency_normalised.png",
                                normalize_rows=True)
    viz.save_adjacency_heatmap(gen_stats.adjacency,
                                out / "generated_adjacency_counts.png",
                                normalize_rows=False)

    # Side-by-side adjacency comparison
    viz.save_adjacency_comparison(
        real_stats.adjacency, gen_stats.adjacency,
        out / "adjacency_comparison.png",
    )

    # Component size overlapping histograms
    for cls in (1, 2):
        viz.save_comparison_component_hist(
            real_stats.component_sizes.get(cls, []),
            gen_stats.component_sizes.get(cls, []),
            out / f"component_sizes_class{cls}_comparison.png",
            cls=cls,
        )

    # Numeric drift summary
    drift = {}
    for c in (0, 1, 2):
        r = real_stats.class_pixel_fractions[c]
        g = gen_stats.class_pixel_fractions[c]
        drift[f"class_{c}_delta"] = round(g - r, 6)

    summary = {
        "n_generated": len(gen_masks),
        "class_fraction_drift": drift,
        "generated_class_fractions": {str(k): v for k, v in gen_stats.class_pixel_fractions.items()},
        "real_class_fractions":      {str(k): v for k, v in real_stats.class_pixel_fractions.items()},
        "generated_component_stats": {
            str(cls): distribution_summary(gen_stats.component_sizes.get(cls, []))
            for cls in (1, 2)
        },
        "real_component_stats": {
            str(cls): distribution_summary(real_stats.component_sizes.get(cls, []))
            for cls in (1, 2)
        },
    }
    with (out / "comparison_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Comparison saved to {out}")
    _print_comparison_summary(summary)


def _print_comparison_summary(summary: dict) -> None:
    print("\n  === Comparison summary ===")
    print("  Class fractions  (real → generated  [delta]):")
    rf = summary["real_class_fractions"]
    gf = summary["generated_class_fractions"]
    names = {0: "background", 1: "slepice", 2: "mehanske"}
    for c in (0, 1, 2):
        r, g = rf[str(c)], gf[str(c)]
        print(f"    c{c} {names[c]:12s}: {r:.3f} → {g:.3f}  [{g-r:+.3f}]")
    print("  Component size stats:")
    for cls in (1, 2):
        rs = summary["real_component_stats"][str(cls)]
        gs = summary["generated_component_stats"][str(cls)]
        print(f"    class {cls}  real mean={rs['mean']:.1f} median={rs['median']:.1f} "
              f"| gen mean={gs['mean']:.1f} median={gs['median']:.1f}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Config, phases: Sequence[str] = ("all",)) -> None:
    """Run requested phases in order."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.output_dir / "config.json")

    enabled = _parse_phases(phases)

    samples:    list[MaskSample]  | None = None
    stats:      DatasetStats      | None = None
    lib:        PatternLibrary    | None = None
    gen_masks:  list[np.ndarray]        = []
    gen_names:  list[str]               = []

    if "phase1" in enabled:
        samples = phase1_dataset(cfg)

    if "phase2" in enabled:
        if samples is None:
            samples = load_dataset(cfg.dataset_dir)
        stats = phase2_stats(cfg, samples)

    if "phase3" in enabled:
        if samples is None:
            samples = load_dataset(cfg.dataset_dir)
        lib = phase3_patterns(cfg, samples)

    if "phase4" in enabled:
        if lib is None:
            npz = cfg.output_dir / "phase3_patterns" / "pattern_library.npz"
            if npz.exists():
                lib = PatternLibrary.load(npz)
            else:
                if samples is None:
                    samples = load_dataset(cfg.dataset_dir)
                lib = phase3_patterns(cfg, samples)

        real_fracs_for_guidance = None
        if cfg.guide_to_real_fracs or cfg.guidance_strength > 0.0 or cfg.reweight_patterns:
            if stats is None:
                if samples is None:
                    samples = load_dataset(cfg.dataset_dir)
                stats = compute_stats(samples)
            real_fracs_for_guidance = stats.class_pixel_fractions

        gen_masks, gen_names = phase4_generate(cfg, lib, target_fracs=real_fracs_for_guidance)

    if "phase5" in enabled:
        if stats is None:
            if samples is None:
                samples = load_dataset(cfg.dataset_dir)
            stats = phase2_stats(cfg, samples)
        if not gen_masks:
            gen_masks, gen_names = _load_saved_masks(cfg.output_dir / "phase4_generation")
        phase5_compare(cfg, stats, gen_masks, gen_names)

    print("\nPipeline complete. Outputs in:", cfg.output_dir.resolve())


def _parse_phases(phases: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for p in phases:
        p = p.strip().lower()
        if p in ("all", "*", ""):
            return {"phase1", "phase2", "phase3", "phase4", "phase5"}
        aliases = {
            "1": "phase1", "2": "phase2", "3": "phase3",
            "4": "phase4", "5": "phase5",
            "dataset": "phase1", "stats": "phase2", "patterns": "phase3",
            "generate": "phase4", "wfc": "phase4", "compare": "phase5",
        }
        result.add(aliases.get(p, p))
    return result


def _load_saved_masks(phase4_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    masks, names = [], []
    if not phase4_dir.exists():
        return masks, names
    for run_dir in sorted(phase4_dir.glob("run_*")):
        npy = run_dir / "mask.npy"
        if npy.exists():
            masks.append(np.load(npy).astype(np.uint8))
            names.append(run_dir.name)
    return masks, names


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def _save_per_image_csv(stats: DatasetStats, path: Path) -> None:
    rows = []
    for pi in stats.per_image:
        row: dict = {
            "name": pi.name,
            "height": pi.height, "width": pi.width,
            "n_pixels": pi.n_pixels,
        }
        for c in (0, 1, 2):
            row[f"count_class{c}"]    = pi.class_counts[c]
            row[f"fraction_class{c}"] = round(pi.class_fractions[c], 6)
        rows.append(row)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
