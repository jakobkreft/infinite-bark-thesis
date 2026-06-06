# MRF Semantic Mask Generator for Log Bark Textures

Markov Random Field (MRF) based system for synthesizing 2D semantic segmentation masks of log bark surfaces. Learns a nonparametric conditional distribution over label neighborhoods from annotated training masks and synthesizes new masks via Gibbs sampling.

## Labels

| Value | Name | Description |
|-------|------|-------------|
| 0 | ozadje | Background / smooth bark |
| 1 | slepice | Knots |
| 2 | mehanske poškodbe | Mechanical damage |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Dataset statistics

```bash
python -m src.main stats --data masks-dataset
```

### Synthesize a new mask

```bash
# Default 256x256
python -m src.main synthesize --output outputs/synth.png --evaluate

# Custom size with target class proportions
python -m src.main synthesize --height 128 --width 128 \
    --ratio 0.70,0.05,0.25 --lambda-ratio 2.0 \
    --output outputs/synth_128.png --evaluate

# Faster with no multiscale and fewer refinement passes
python -m src.main synthesize --height 64 --width 64 \
    --no-multiscale --refine 1 --output outputs/fast.png

# Reproducible with seed
python -m src.main synthesize --seed 42 --output outputs/seed42.png

# Upscale each mask pixel to 50x50 via Scale2x (EPX) — 100x100 mask -> 5000x5000 output
python -m src.main synthesize --height 100 --width 100 --pixel-scale 50 \
    --output outputs/synth_5000.png
```

### Inpaint a mask

Mark unknown pixels with value 255 in the input PNG, then:

```bash
python -m src.main inpaint input_with_holes.png \
    --output outputs/inpainted.png --evaluate
```

### Evaluate a mask

```bash
python -m src.main evaluate outputs/synth.png --data masks-dataset
```

## CLI Arguments

### `synthesize`

| Argument | Default | Description |
|----------|---------|-------------|
| `--height` | 256 | Output height in pixels |
| `--width` | 256 | Output width in pixels |
| `--radius` | 3 | Neighborhood window radius |
| `--refine` | 3 | Number of Gibbs refinement passes |
| `--no-multiscale` | false | Disable coarse-to-fine initialization |
| `--ratio` | empirical | Target class proportions, e.g. `0.7,0.05,0.25` |
| `--lambda-ratio` | 1.0 | Weight for proportion bias |
| `--temperature` | 1.0 | Sampling temperature (lower = more deterministic) |
| `--k-fallback` | 11 | KDTree nearest neighbor k for fallback |
| `--seed` | None | Random seed for reproducibility |
| `--pixel-scale` | 1 | Upscale each mask pixel to a `pixel_scale × pixel_scale` block using iterated Scale2x (EPX) + nearest-neighbor resample to the exact factor. Rounds diagonal edges. |
| `--target-multiple` | 512 | Snap final output side lengths to a multiple of this value. Set to 1 to disable. |
| `--size-mode` | `ceil` | How to snap: `ceil` (synth slightly larger, center-crop), `floor` (crop down), or `off`. |
| `--output` | `outputs/synthesized.png` | Output path |
| `--evaluate` | false | Print evaluation report (runs on the pre-upscale mask) |

### `inpaint`

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | required | Input PNG with holes (unknown = 255) |
| `--unknown-value` | 255 | Pixel value marking unknown regions |
| `--pairwise-scale` | 1.0 | Scaling for pairwise costs |
| `--ratio` | None | Target class proportions |
| `--refine` | 2 | Gibbs refinement passes on inpainted region |
| `--pixel-scale` | 1 | Upscale each mask pixel to a `pixel_scale × pixel_scale` block via Scale2x (EPX). |
| `--output` | `outputs/inpainted.png` | Output path |

## Architecture

```
src/
├── dataset.py      # Dataset loading, augmentation, potential estimation
├── mrf_model.py    # Unary/pairwise potentials, graph-cut MAP inference
├── synthesis.py    # Nonparametric Gibbs sampler with multiscale init
├── inpainting.py   # MAP inpainting via graph cuts + Gibbs refinement
├── scale2x.py      # EPX (Scale2x) upscaler for mask-pixel → N×N blocks with rounded edges
├── evaluate.py     # Class freq error, GLCM, component stats, tileability
└── main.py         # CLI entry point
```

## Method

1. **Potential estimation**: Direction-specific 3×3 pairwise compatibility matrices (`-log P(c2|c1)`) estimated from all 4-connected neighbor pairs in training masks. Unary potentials from empirical class frequencies with optional Dirichlet-sampled proportion bias.

2. **Nonparametric Gibbs synthesis**: For each pixel, extract its neighborhood window (toroidal wrapping), look up the empirical conditional distribution from training data (with KDTree fallback for unseen patterns), and sample a label.

3. **Multiscale initialization**: Synthesize at 1/4 resolution first, upsample via nearest neighbor, then refine at full resolution to prevent large-scale structural failures.

4. **Gibbs refinement**: Multiple full sweeps in random order using the complete (non-causal) neighborhood to improve global consistency.

5. **Toroidal boundaries**: All neighborhood lookups use modular indexing — output masks tile seamlessly.

6. **Inpainting**: Known pixels pinned via strong unary potentials; unknown pixels solved by alpha-expansion graph cuts (gco-wrapper), followed by optional Gibbs refinement.
