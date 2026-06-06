from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable
import json
import math
import numpy as np
from PIL import Image, ImageDraw

from .config import PipelineConfig
from .dataset import DatasetBundle, load_dataset, save_dataset_manifest
from .patterns import PatternLibrary, extract_patterns, save_compatibility_report
from .stats import StatsResult, compute_statistics, summarize_distribution
from .utils import ensure_dir, save_csv, save_json, class_name
from .visualize import (
    mask_to_rgb,
    save_bar_chart,
    save_fraction_boxplot_like,
    save_heatmap,
    save_histogram,
    save_mask_gallery,
    save_mask_png,
    save_pattern_gallery,
    save_text_report,
)
from .wfc import WFCGenerationResult, WFCGenerator


def run_pipeline(config: PipelineConfig, phases: Iterable[str] | None = None) -> dict:
    phases = normalize_phases(phases)

    ensure_dir(config.output_dir)
    config.to_json(config.output_dir / "pipeline_config.json")

    phase_outputs: dict[str, dict] = {}

    bundle: DatasetBundle | None = None
    real_stats: StatsResult | None = None
    pattern_lib: PatternLibrary | None = None
    generated_masks: list[np.ndarray] = []
    generated_names: list[str] = []

    if phase_enabled(phases, "phase1") or phase_enabled(phases, "phase2") or phase_enabled(phases, "phase3"):
        bundle = run_phase1_dataset(config)
        phase_outputs["phase1"] = {
            "dataset_dir": str(config.dataset_dir),
            "num_images": len(bundle.samples),
            "values": bundle.unique_values,
        }

    if phase_enabled(phases, "phase2"):
        if bundle is None:
            bundle = load_dataset(config.dataset_dir, allowed_values={0, 1, 2})
        real_stats = run_phase2_statistics(config, bundle)
        phase_outputs["phase2"] = real_stats.summary

    if phase_enabled(phases, "phase3"):
        if bundle is None:
            bundle = load_dataset(config.dataset_dir, allowed_values={0, 1, 2})
        pattern_lib = run_phase3_patterns(config, bundle)
        phase_outputs["phase3"] = {
            "num_patterns": int(pattern_lib.patterns.shape[0]),
            "pattern_size": int(pattern_lib.pattern_size),
        }

    if phase_enabled(phases, "phase4"):
        if pattern_lib is None:
            pattern_npz = config.output_dir / "phase3_patterns" / "pattern_library.npz"
            if not pattern_npz.exists():
                if bundle is None:
                    bundle = load_dataset(config.dataset_dir, allowed_values={0, 1, 2})
                pattern_lib = run_phase3_patterns(config, bundle)
            else:
                pattern_lib = PatternLibrary.load(pattern_npz)

        generated_masks, generated_names = run_phase4_generation(config, pattern_lib)
        phase_outputs["phase4"] = {
            "num_generated": len(generated_masks),
            "target_size": [config.target_size, config.target_size],
            "tile_mode": config.tile_mode,
        }

    if phase_enabled(phases, "phase5"):
        if real_stats is None:
            if bundle is None:
                bundle = load_dataset(config.dataset_dir, allowed_values={0, 1, 2})
            real_stats = compute_statistics(
                masks=[s.mask for s in bundle.samples],
                names=[s.name for s in bundle.samples],
                classes=(0, 1, 2),
            )

        if not generated_masks:
            generated_masks, generated_names = load_generated_masks(config.output_dir / "phase4_wfc")
        if not generated_masks:
            raise RuntimeError(
                "No generated masks found. Run phase4 first or keep outputs in outputs/phase4_wfc."
            )

        comparison = run_phase5_comparison(config, real_stats, generated_masks, generated_names)
        phase_outputs["phase5"] = comparison

    save_json({"phases": phase_outputs}, config.output_dir / "pipeline_report.json")
    return phase_outputs


def normalize_phases(phases: Iterable[str] | None) -> set[str]:
    if phases is None:
        return {"phase1", "phase2", "phase3", "phase4", "phase5"}

    norm = set()
    for p in phases:
        p = p.strip().lower()
        if p in {"all", "*"}:
            return {"phase1", "phase2", "phase3", "phase4", "phase5"}
        if p in {"1", "phase1", "dataset"}:
            norm.add("phase1")
        elif p in {"2", "phase2", "stats"}:
            norm.add("phase2")
        elif p in {"3", "phase3", "patterns"}:
            norm.add("phase3")
        elif p in {"4", "phase4", "wfc", "generate"}:
            norm.add("phase4")
        elif p in {"5", "phase5", "compare", "comparison"}:
            norm.add("phase5")
        else:
            raise ValueError(f"Unknown phase label: {p}")
    return norm


def phase_enabled(phases: set[str], phase: str) -> bool:
    return phase in phases


def run_phase1_dataset(config: PipelineConfig) -> DatasetBundle:
    out_dir = config.output_dir / "phase1_dataset"
    ensure_dir(out_dir)

    bundle = load_dataset(config.dataset_dir, allowed_values={0, 1, 2})
    save_dataset_manifest(bundle, out_dir)

    names = [s.name for s in bundle.samples]
    masks = [s.mask for s in bundle.samples]
    save_mask_gallery(
        masks=masks,
        titles=names,
        path=out_dir / "dataset_sample_gallery.png",
        cols=10,
        scale=8,
        max_items=80,
        title="Dataset Samples (first 80)",
    )

    lines = [
        f"num_images: {len(bundle.samples)}",
        f"unique_values: {bundle.unique_values}",
        f"height_range: {min(s.mask.shape[0] for s in bundle.samples)}..{max(s.mask.shape[0] for s in bundle.samples)}",
        f"width_range: {min(s.mask.shape[1] for s in bundle.samples)}..{max(s.mask.shape[1] for s in bundle.samples)}",
    ]
    save_text_report(lines, out_dir / "dataset_overview.png", title="Phase 1 - Dataset Overview")

    return bundle


def run_phase2_statistics(config: PipelineConfig, bundle: DatasetBundle) -> StatsResult:
    out_dir = config.output_dir / "phase2_statistics"
    ensure_dir(out_dir)

    masks = [s.mask for s in bundle.samples]
    names = [s.name for s in bundle.samples]
    stats = compute_statistics(masks=masks, names=names, classes=(0, 1, 2))

    save_json(stats.summary, out_dir / "stats_summary.json")
    save_csv(stats.per_image_rows, out_dir / "per_image_stats.csv")

    cls_labels = [f"{cls}:{class_name(cls)}" for cls in [0, 1, 2]]
    cls_counts = [stats.class_pixel_counts[c] for c in [0, 1, 2]]
    cls_frac = [stats.class_pixel_fractions[c] * 100.0 for c in [0, 1, 2]]

    save_bar_chart(
        labels=cls_labels,
        values=cls_counts,
        path=out_dir / "class_pixel_counts.png",
        title="Class Pixel Counts (Dataset)",
        y_label="pixels",
    )
    save_bar_chart(
        labels=cls_labels,
        values=cls_frac,
        path=out_dir / "class_pixel_percent.png",
        title="Class Pixel Percentage (Dataset)",
        y_label="percent",
        bar_color=(244, 160, 71),
    )

    size_labels = list(stats.size_distribution.keys())
    size_vals = list(stats.size_distribution.values())
    save_bar_chart(
        labels=size_labels,
        values=size_vals,
        path=out_dir / "size_distribution.png",
        title="Image Size Distribution",
        y_label="count",
        bar_color=(130, 120, 220),
    )

    save_fraction_boxplot_like(
        stats.class_fraction_per_image,
        path=out_dir / "per_image_class_fraction_percentiles.png",
        title="Per-image Class Fraction Percentiles",
    )

    co_labels = list(stats.cooccurrence_has_1_2.keys())
    co_vals = [stats.cooccurrence_has_1_2[k] for k in co_labels]
    save_bar_chart(
        labels=co_labels,
        values=co_vals,
        path=out_dir / "class1_class2_cooccurrence.png",
        title="Class 1 / Class 2 Co-occurrence per Image",
        y_label="images",
        bar_color=(90, 180, 160),
    )

    save_heatmap(
        matrix=stats.adjacency_matrix,
        path=out_dir / "adjacency_counts.png",
        row_labels=cls_labels,
        col_labels=cls_labels,
        title="Adjacency Count Matrix (Directed 4-neighborhood)",
        normalize_rows=False,
    )
    save_heatmap(
        matrix=stats.adjacency_matrix,
        path=out_dir / "adjacency_probabilities.png",
        row_labels=cls_labels,
        col_labels=cls_labels,
        title="Adjacency Transition Probabilities P(neighbor|center)",
        normalize_rows=True,
    )

    for cls in [1, 2]:
        save_histogram(
            values=stats.component_sizes.get(cls, []),
            path=out_dir / f"component_size_hist_class_{cls}.png",
            title=f"Connected Component Size Histogram - class {cls}",
            x_label="component_size",
            y_label="count",
            bins=30,
            color=(200, 110, 60) if cls == 1 else (90, 130, 240),
            log_x=True,
        )

    lines = [
        f"num_images={stats.summary['num_images']}",
        f"total_pixels={stats.summary['total_pixels']}",
    ]
    for cls in [0, 1, 2]:
        lines.append(
            f"class_{cls}_fraction={stats.class_pixel_fractions[cls]:.5f} "
            f"(count={stats.class_pixel_counts[cls]})"
        )
    for cls in [1, 2]:
        comp = stats.summary["component_stats"][str(cls)]
        lines.append(
            f"class_{cls}_components: count={comp['count']} mean={comp['mean']:.2f} "
            f"median={comp['median']:.2f} p90={comp['p90']:.2f}"
        )

    save_text_report(lines, out_dir / "statistics_overview.png", title="Phase 2 - Statistics Overview")

    return stats


def run_phase3_patterns(config: PipelineConfig, bundle: DatasetBundle) -> PatternLibrary:
    out_dir = config.output_dir / "phase3_patterns"
    ensure_dir(out_dir)

    masks = [s.mask for s in bundle.samples]
    lib = extract_patterns(
        masks=masks,
        pattern_size=config.pattern_size,
        augment_symmetry=config.augment_symmetry,
        min_weight=config.min_pattern_weight,
        max_patterns=config.max_patterns,
    )
    lib.save(out_dir)
    save_compatibility_report(lib, out_dir / "compatibility_report.json")

    save_pattern_gallery(
        patterns=[p for p in lib.patterns],
        weights=lib.weights.tolist(),
        path=out_dir / "top_patterns_gallery.png",
        title=f"Top Patterns (size={config.pattern_size})",
        max_items=config.max_patterns_for_gallery,
        scale=20,
    )

    save_histogram(
        values=lib.weights.tolist(),
        path=out_dir / "pattern_weight_histogram.png",
        title="Pattern Frequency Histogram",
        x_label="pattern_frequency",
        y_label="count",
        bins=25,
        color=(120, 80, 210),
        log_x=True,
    )

    density_labels = ["up", "down", "left", "right"]
    density_values = [float(lib.compat[d].mean()) for d in density_labels]
    save_bar_chart(
        labels=density_labels,
        values=density_values,
        path=out_dir / "compatibility_density.png",
        title="Compatibility Matrix Density per Direction",
        y_label="density",
        bar_color=(70, 140, 220),
    )

    for d in density_labels:
        out_deg = lib.compat[d].sum(axis=1).astype(np.int64)
        save_histogram(
            values=out_deg.tolist(),
            path=out_dir / f"out_degree_hist_{d}.png",
            title=f"Pattern Out-degree Histogram ({d})",
            x_label="out_degree",
            y_label="patterns",
            bins=25,
            color=(110, 160, 120),
            log_x=False,
        )

    lines = [
        f"pattern_size={lib.pattern_size}",
        f"min_pattern_weight={config.min_pattern_weight}",
        f"max_patterns={config.max_patterns}",
        f"num_patterns={lib.patterns.shape[0]}",
        f"weight_sum={int(lib.weights.sum())}",
        f"weight_min={int(lib.weights.min())}",
        f"weight_max={int(lib.weights.max())}",
    ]
    for d in ["up", "down", "left", "right"]:
        lines.append(f"compat_density_{d}={float(lib.compat[d].mean()):.6f}")

    save_text_report(lines, out_dir / "pattern_overview.png", title="Phase 3 - Pattern Library Overview")

    return lib


def run_phase4_generation(config: PipelineConfig, lib: PatternLibrary) -> tuple[list[np.ndarray], list[str]]:
    out_dir = config.output_dir / "phase4_wfc"
    ensure_dir(out_dir)

    generated_masks: list[np.ndarray] = []
    generated_names: list[str] = []

    generator = WFCGenerator(
        library=lib,
        target_height=config.target_size,
        target_width=config.target_size,
        seed=config.seed,
        tile_mode=config.tile_mode,
    )

    for i in range(config.num_generations):
        run_name = f"run_{i:03d}"
        run_dir = out_dir / run_name
        steps_dir = run_dir / "steps"
        ensure_dir(steps_dir)

        # Use shifted seed between runs for diversity.
        generator.seed = config.seed + i * 10_000

        def snapshot_cb(step: int, wave: np.ndarray, tag: str) -> None:
            snap_name = f"step_{step:05d}_{tag}.png"
            _save_wfc_snapshot(generator, wave, steps_dir / snap_name, title=f"{run_name} | step={step} | {tag}")

        result = generator.generate(
            max_restarts=config.max_restarts,
            snapshot_interval=config.snapshot_interval,
            snapshot_cb=snapshot_cb,
        )

        run_meta = {
            "run_name": run_name,
            "success": bool(result.success),
            "steps": int(result.steps),
            "restart_index": int(result.restart_index),
            "message": result.message,
            "target_size": [config.target_size, config.target_size],
            "pattern_size": int(lib.pattern_size),
            "tile_mode": config.tile_mode,
            "periodic_x": bool(generator.periodic_x),
            "periodic_y": bool(generator.periodic_y),
        }

        if not result.success or result.mask is None:
            save_json(run_meta, run_dir / "run_meta.json")
            continue

        raw_path = run_dir / "generated_mask_raw.png"
        Image.fromarray(result.mask.astype(np.uint8), mode="L").save(raw_path)
        np.save(run_dir / "generated_mask.npy", result.mask.astype(np.uint8))
        save_mask_png(result.mask, run_dir / "generated_mask_color.png", scale=6)

        if result.collapsed_pattern_ids is not None:
            np.save(run_dir / "collapsed_pattern_ids.npy", result.collapsed_pattern_ids.astype(np.int32))

        save_json(run_meta, run_dir / "run_meta.json")

        generated_masks.append(result.mask.astype(np.uint8))
        generated_names.append(run_name)

    if generated_masks:
        save_mask_gallery(
            masks=generated_masks,
            titles=generated_names,
            path=out_dir / "generated_gallery.png",
            cols=min(4, len(generated_masks)),
            scale=5,
            max_items=len(generated_masks),
            title=f"Generated Masks ({config.target_size}x{config.target_size})",
        )

    summary = {
        "requested_generations": int(config.num_generations),
        "successful_generations": len(generated_masks),
        "target_size": int(config.target_size),
        "pattern_size": int(config.pattern_size),
        "tile_mode": config.tile_mode,
    }
    save_json(summary, out_dir / "generation_summary.json")

    return generated_masks, generated_names


def _save_wfc_snapshot(
    generator: WFCGenerator,
    wave: np.ndarray,
    path: Path,
    title: str,
) -> None:
    ensure_dir(path.parent)

    counts = wave.sum(axis=2)
    collapsed = (counts == 1).astype(np.uint8)
    entropy = generator.entropy_map(wave)
    preview = generator.reconstruct_preview_from_wave(wave)

    collapsed_img = Image.fromarray((collapsed * 255).astype(np.uint8), mode="L").resize(
        (collapsed.shape[1] * 4, collapsed.shape[0] * 4), resample=Image.NEAREST
    ).convert("RGB")

    entropy_img = _entropy_to_rgb(entropy).resize((entropy.shape[1] * 4, entropy.shape[0] * 4), resample=Image.NEAREST)

    preview_img = Image.fromarray(mask_to_rgb(preview), mode="RGB").resize(
        (preview.shape[1] * 4, preview.shape[0] * 4), resample=Image.NEAREST
    )

    panel_w = max(collapsed_img.width, entropy_img.width, preview_img.width)
    title_h = 42
    gap = 16
    h = title_h + collapsed_img.height + entropy_img.height + preview_img.height + gap * 4
    canvas = Image.new("RGB", (panel_w + 24, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((12, 10), title, fill=(25, 25, 25))
    y = title_h
    canvas.paste(collapsed_img, (12, y))
    draw.text((14, y + 2), "collapsed cells", fill=(255, 0, 0))
    y += collapsed_img.height + gap

    canvas.paste(entropy_img, (12, y))
    draw.text((14, y + 2), "entropy map", fill=(0, 0, 0))
    y += entropy_img.height + gap

    canvas.paste(preview_img, (12, y))
    draw.text((14, y + 2), "preview reconstruction", fill=(0, 0, 0))

    canvas.save(path)


def _entropy_to_rgb(entropy: np.ndarray) -> Image.Image:
    arr = np.array(entropy, dtype=np.float64)
    valid = arr >= 0
    if np.any(valid):
        lo = float(arr[valid].min())
        hi = float(arr[valid].max())
        den = hi - lo if hi > lo else 1.0
        norm = np.zeros_like(arr)
        norm[valid] = (arr[valid] - lo) / den
    else:
        norm = np.zeros_like(arr)

    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if arr[y, x] < 0:
                rgb[y, x] = (220, 0, 120)
            else:
                t = float(norm[y, x])
                rgb[y, x] = _viridis_like(t)
    return Image.fromarray(rgb, mode="RGB")


def _viridis_like(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        a = t / 0.5
        r = int(35 + a * (53 - 35))
        g = int(40 + a * (183 - 40))
        b = int(130 + a * (121 - 130))
    else:
        a = (t - 0.5) / 0.5
        r = int(53 + a * (252 - 53))
        g = int(183 + a * (231 - 183))
        b = int(121 + a * (37 - 121))
    return (r, g, b)


def load_generated_masks(phase4_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    if not phase4_dir.exists():
        return [], []

    masks: list[np.ndarray] = []
    names: list[str] = []
    for run_dir in sorted(phase4_dir.glob("run_*")):
        npy = run_dir / "generated_mask.npy"
        png = run_dir / "generated_mask_raw.png"
        if npy.exists():
            arr = np.load(npy)
        elif png.exists():
            arr = np.array(Image.open(png))
            if arr.ndim == 3:
                arr = arr[..., 0]
        else:
            continue

        masks.append(arr.astype(np.uint8))
        names.append(run_dir.name)

    return masks, names


def run_phase5_comparison(
    config: PipelineConfig,
    real_stats: StatsResult,
    generated_masks: list[np.ndarray],
    generated_names: list[str],
) -> dict:
    out_dir = config.output_dir / "phase5_comparison"
    ensure_dir(out_dir)

    gen_stats = compute_statistics(generated_masks, names=generated_names, classes=(0, 1, 2))
    save_json(gen_stats.summary, out_dir / "generated_stats_summary.json")
    save_csv(gen_stats.per_image_rows, out_dir / "generated_per_image_stats.csv")

    labels = []
    values = []
    for cls in [0, 1, 2]:
        labels.append(f"real_c{cls}")
        values.append(float(real_stats.class_pixel_fractions[cls] * 100.0))
        labels.append(f"gen_c{cls}")
        values.append(float(gen_stats.class_pixel_fractions[cls] * 100.0))

    save_bar_chart(
        labels=labels,
        values=values,
        path=out_dir / "real_vs_generated_class_percent.png",
        title="Real vs Generated Class Pixel Percentage",
        y_label="percent",
        bar_color=(235, 120, 70),
    )

    save_heatmap(
        matrix=gen_stats.adjacency_matrix,
        path=out_dir / "generated_adjacency_probabilities.png",
        row_labels=[f"{cls}:{class_name(cls)}" for cls in [0, 1, 2]],
        col_labels=[f"{cls}:{class_name(cls)}" for cls in [0, 1, 2]],
        title="Generated Adjacency Probabilities",
        normalize_rows=True,
    )

    for cls in [1, 2]:
        save_histogram(
            values=gen_stats.component_sizes.get(cls, []),
            path=out_dir / f"generated_component_hist_class_{cls}.png",
            title=f"Generated Component Size Histogram class {cls}",
            x_label="component_size",
            y_label="count",
            bins=30,
            color=(200, 110, 60) if cls == 1 else (90, 130, 240),
            log_x=True,
        )

    save_mask_gallery(
        masks=generated_masks,
        titles=generated_names,
        path=out_dir / "generated_masks_gallery.png",
        cols=min(4, len(generated_masks)),
        scale=5,
        max_items=len(generated_masks),
        title="Generated masks used for comparison",
    )

    # Simple numeric drift summary.
    drift = {}
    lines = ["Real vs Generated Drift Summary"]
    for cls in [0, 1, 2]:
        r = float(real_stats.class_pixel_fractions[cls])
        g = float(gen_stats.class_pixel_fractions[cls])
        d = g - r
        drift[f"class_{cls}_fraction_delta"] = d
        lines.append(f"class {cls}: real={r:.6f} generated={g:.6f} delta={d:+.6f}")

    for cls in [1, 2]:
        real_comp = summarize_distribution(real_stats.component_sizes.get(cls, []))
        gen_comp = summarize_distribution(gen_stats.component_sizes.get(cls, []))
        lines.append(
            f"class {cls} component mean: real={real_comp['mean']:.3f}, generated={gen_comp['mean']:.3f}, "
            f"delta={gen_comp['mean'] - real_comp['mean']:+.3f}"
        )

    save_text_report(lines, out_dir / "comparison_summary.png", title="Phase 5 - Comparison Summary")

    comparison_payload = {
        "num_generated": len(generated_masks),
        "class_fraction_drift": drift,
    }
    save_json(comparison_payload, out_dir / "comparison_metrics.json")

    return comparison_payload
