#!/usr/bin/env python3
"""
Generate 3-class PNG masks for diffusion-conditioning tests.

Output masks contain only integer class IDs: 0, 1, 2 (stored as uint8 PNG).
The default suite is intentionally small but high-coverage:
- Tileability checks
- Class mapping sanity checks
- Sharp boundaries and thin structures
- Circular structures (especially class 2)
- Multi-scale complexity and overlap stress
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


MaskGenerator = Callable[[int, int, np.random.Generator], np.ndarray]
CLASS_COLORS = np.array(
    [
        [0, 0, 0],       # class 0 -> black
        [0, 0, 255],     # class 1 -> blue
        [255, 190, 0],   # class 2 -> orange-yellow
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class Pattern:
    name: str
    tileable: bool
    description: str
    generator: MaskGenerator


def _validate_mask(mask: np.ndarray) -> None:
    if mask.dtype != np.uint8:
        raise ValueError(f"Mask dtype must be uint8, got: {mask.dtype}")
    unique = np.unique(mask)
    if np.any((unique < 0) | (unique > 2)):
        raise ValueError(f"Mask contains invalid class values: {unique.tolist()}")


def _save_mask(mask: np.ndarray, out_path: Path) -> None:
    _validate_mask(mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_path, format="PNG")


def _save_visualized_mask(mask: np.ndarray, out_path: Path) -> None:
    _validate_mask(mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = CLASS_COLORS[mask]
    Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG")


def _paint_disk(mask: np.ndarray, cx: int, cy: int, radius: int, cls: int) -> None:
    h, w = mask.shape
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    patch = mask[y0:y1, x0:x1]
    patch[inside] = cls


def _uniform_class(cls: int) -> MaskGenerator:
    def gen(height: int, width: int, _rng: np.random.Generator) -> np.ndarray:
        return np.full((height, width), cls, dtype=np.uint8)

    return gen


def tile_checkerboard_0_1(height: int, width: int, _rng: np.random.Generator) -> np.ndarray:
    cell = max(4, min(height, width) // 16)
    yy = np.arange(height, dtype=np.int32)[:, None]
    xx = np.arange(width, dtype=np.int32)[None, :]
    board = ((yy // cell) + (xx // cell)) % 2
    return board.astype(np.uint8)  # classes 0 and 1


def tile_stripes_0_2(height: int, width: int, _rng: np.random.Generator) -> np.ndarray:
    stripe = max(2, width // 24)
    xx = np.arange(width, dtype=np.int32)[None, :]
    return ((xx // stripe) % 2 * 2).repeat(height, axis=0).astype(np.uint8)  # 0/2


def tile_ring_lattice_0_1_2(height: int, width: int, _rng: np.random.Generator) -> np.ndarray:
    period = max(12, min(height, width) // 8)
    outer = max(3, int(period * 0.34))
    inner = max(1, int(period * 0.17))

    yy = np.arange(height, dtype=np.int32)[:, None]
    xx = np.arange(width, dtype=np.int32)[None, :]

    # Distances on a periodic grid so opposite edges match (tileable).
    gx = ((xx + period // 2) % period) - period // 2
    gy = ((yy + period // 2) % period) - period // 2
    d2 = gx * gx + gy * gy

    mask = np.zeros((height, width), dtype=np.uint8)
    ring = d2 <= outer**2
    core = d2 <= inner**2
    mask[ring] = 1
    mask[core] = 2
    return mask


def center_square_with_core_circle(height: int, width: int, _rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    side = max(8, int(min(height, width) * 0.5))
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    mask[y0 : y0 + side, x0 : x0 + side] = 1

    cy = y0 + side // 2
    cx = x0 + side // 2
    r = max(3, int(side * 0.18))
    yy, xx = np.ogrid[:height, :width]
    core = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    mask[core] = 2
    return mask


def concentric_circles_0_1_2(height: int, width: int, _rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    yy, xx = np.ogrid[:height, :width]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2

    r_inner = max(3.0, min(height, width) * 0.17)
    r_outer = max(r_inner + 2.0, min(height, width) * 0.34)

    mask[d2 <= r_outer**2] = 1
    mask[d2 <= r_inner**2] = 2
    return mask


def multiscale_circle_stress(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    min_dim = min(height, width)

    # Scale count and radius with image size while keeping runtime bounded.
    n_circles = int(np.clip((height * width) / (512 * 512) * 18, 18, 120))
    r_min = max(2, min_dim // 140)
    r_max = max(r_min + 2, min_dim // 10)

    for _ in range(n_circles):
        cls = int(rng.integers(0, 3))
        radius = int(rng.integers(r_min, r_max + 1))
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        _paint_disk(mask, cx, cy, radius, cls)

    # Add very small class-2 targets to test tiny circular details.
    n_tiny = int(np.clip((height * width) / (512 * 512) * 24, 12, 180))
    tiny_r = max(1, min_dim // 280)
    for _ in range(n_tiny):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        _paint_disk(mask, cx, cy, tiny_r, 2)

    return mask


def wavy_three_band_boundary(height: int, width: int, _rng: np.random.Generator) -> np.ndarray:
    # Thin, curved interfaces to stress boundary-following behavior.
    mask = np.zeros((height, width), dtype=np.uint8)
    x = np.arange(width, dtype=np.float64)
    denom = max(1, width - 1)
    amp = max(2, height // 10)

    y1 = (0.33 * height + amp * np.sin(2.0 * np.pi * 3.0 * x / denom)).astype(np.int32)
    y2 = (0.66 * height + amp * np.cos(2.0 * np.pi * 5.0 * x / denom)).astype(np.int32)

    for col in range(width):
        a = int(np.clip(y1[col], 1, max(1, height - 2)))
        b = int(np.clip(y2[col], a + 1, max(a + 1, height - 1)))
        mask[a:b, col] = 1
        mask[b:, col] = 2

    return mask


PATTERNS: list[Pattern] = [
    Pattern(
        name="tile_uniform_0",
        tileable=True,
        description="All pixels are class 0 (tiling sanity baseline).",
        generator=_uniform_class(0),
    ),
    Pattern(
        name="tile_uniform_1",
        tileable=True,
        description="All pixels are class 1 (class mapping sanity check).",
        generator=_uniform_class(1),
    ),
    Pattern(
        name="tile_uniform_2",
        tileable=True,
        description="All pixels are class 2 (class mapping sanity check).",
        generator=_uniform_class(2),
    ),
    Pattern(
        name="tile_checkerboard_0_1",
        tileable=True,
        description="Periodic high-frequency checkerboard with classes 0/1.",
        generator=tile_checkerboard_0_1,
    ),
    Pattern(
        name="tile_stripes_0_2",
        tileable=True,
        description="Periodic stripe pattern with classes 0/2.",
        generator=tile_stripes_0_2,
    ),
    Pattern(
        name="tile_ring_lattice_0_1_2",
        tileable=True,
        description="Periodic circular lattice with ring/core (0/1/2).",
        generator=tile_ring_lattice_0_1_2,
    ),
    Pattern(
        name="center_square_with_core_circle",
        tileable=False,
        description="Centered square (class 1) and core circle (class 2) on class 0.",
        generator=center_square_with_core_circle,
    ),
    Pattern(
        name="concentric_circles_0_1_2",
        tileable=False,
        description="Large ring + central disk to test radial class transitions.",
        generator=concentric_circles_0_1_2,
    ),
    Pattern(
        name="multiscale_circle_stress",
        tileable=False,
        description="Deterministic random overlaps of multi-scale circles (all classes).",
        generator=multiscale_circle_stress,
    ),
    Pattern(
        name="wavy_three_band_boundary",
        tileable=False,
        description="Three classes separated by thin sinusoidal boundaries.",
        generator=wavy_three_band_boundary,
    ),
]


def _pattern_map() -> dict[str, Pattern]:
    return {pattern.name: pattern for pattern in PATTERNS}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3-class PNG masks (values: 0,1,2) for diffusion tests."
    )
    parser.add_argument("--width", type=int, default=2048, help="Mask width in pixels.")
    parser.add_argument("--height", type=int, default=2048, help="Mask height in pixels.")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Optional shorthand to set both width and height.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("generated_masks"),
        help="Directory to save PNG masks.",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="Directory for colorized mask previews. Defaults to <outdir>/visualized_masks.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix (e.g., runA_).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used for stochastic-looking deterministic patterns.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="Pattern names to generate. Omit to generate the full test suite.",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List available pattern names and exit.",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Disable colorized preview output.",
    )
    parser.set_defaults(visualize=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pmap = _pattern_map()

    if args.list_patterns:
        for pattern in PATTERNS:
            tile = "tileable" if pattern.tileable else "non_tileable"
            print(f"{pattern.name:30s} [{tile}] - {pattern.description}")
        return

    width = args.width
    height = args.height
    if args.size is not None:
        width, height = args.size

    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive. Got width={width}, height={height}")

    selected_names = args.patterns if args.patterns is not None else [p.name for p in PATTERNS]
    unknown = [name for name in selected_names if name not in pmap]
    if unknown:
        raise ValueError(f"Unknown pattern(s): {unknown}. Use --list-patterns to inspect available names.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.viz_dir if args.viz_dir is not None else args.outdir / "visualized_masks"
    if args.visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Generating {len(selected_names)} masks at {width}x{height} into: {args.outdir.resolve()}"
    )
    if args.visualize:
        print(f"Colorized previews -> {viz_dir.resolve()}")

    for idx, name in enumerate(selected_names):
        pattern = pmap[name]
        rng = np.random.default_rng(args.seed + idx)
        mask = pattern.generator(height, width, rng)
        out_path = args.outdir / f"{args.prefix}{name}.png"
        _save_mask(mask, out_path)
        if args.visualize:
            viz_out_path = viz_dir / f"{args.prefix}{name}.png"
            _save_visualized_mask(mask, viz_out_path)
        classes = np.unique(mask).tolist()
        tile = "tileable" if pattern.tileable else "non_tileable"
        print(f"- {out_path.name} | {tile:12s} | classes={classes}")


if __name__ == "__main__":
    main()
