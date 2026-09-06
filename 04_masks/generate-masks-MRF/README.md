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

## Model caching

Learning the nonparametric neighborhood tables from the training dataset (looping over every pixel of every augmented mask and fitting KDTree fallbacks over the unique observed neighborhoods) takes minutes and depends only on `--data` and `--radius` — not on canvas size, seed, or anything else. `synthesize` and `inpaint` cache this to disk automatically and reuse it on the next matching call, so **only the first run against a given dataset/radius pays the slow cost**; every run after that just loads the compact cache file.

The cache lives under `.mrf_cache/` (gitignored) and is keyed on the dataset folder's contents (filenames/sizes/mtimes) plus `--radius`, so editing the training masks or changing `--radius` invalidates it automatically.

```bash
# Optional: build the cache ahead of time instead of paying for it on the first synthesize/inpaint call
python -m src.main precompute --radius 3

# Generate many masks in one process — the model is built/loaded once and reused
# for all of them, instead of a shell loop paying the setup cost on every iteration
python -m src.main synthesize --height 64 --width 64 --pixel-scale 8 \
    --target-multiple 1 --count 1000 --output "outputs/512/mask.png"
# -> outputs/512/mask_0001.png ... outputs/512/mask_1000.png
```

Other cache-related flags (see full tables below): `--cache-dir` (default `.mrf_cache`), `--no-cache` (compute in memory, don't read/write the cache), `--refresh-cache` (force a rebuild).

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

# Only tileable left-right (e.g. wraps around a cylinder but has a distinct top/bottom)
python -m src.main synthesize --tileable horizontal --output outputs/tile_h.png

# Generate 1000 masks in one process (see "Model caching" above)
python -m src.main synthesize --count 1000 --output "outputs/512/mask.png"
```

### Inpaint a mask

Mark unknown pixels with value 255 in the input PNG, then:

```bash
python -m src.main inpaint input_with_holes.png \
    --output outputs/inpainted.png --evaluate
```

Unknown pixels are randomly initialized and then filled with the same coarse-to-fine + Gibbs-refinement texture synthesis `synthesize` uses to build a mask from nothing — known pixels are never modified, so they act as fixed boundary context the fill is constrained by. This is what makes the result look like plausible bark texture instead of a flat blob: it's a genuine random sample from the texture model, not a smoothness-maximizing MAP estimate. For large holes, keep the default coarse-to-fine fill (`--no-multiscale` skips it and fills at full resolution directly, which mixes much more slowly for big holes and can leave visible noise/blockiness).

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
| `--lambda-ratio` | 1.0 | Weight for the global count penalty used to keep generated proportions near `--ratio`. Set to `0` to disable ratio control. |
| `--temperature` | 1.0 | Sampling temperature (lower = more deterministic) |
| `--k-fallback` | 11 | KDTree nearest neighbor k for fallback |
| `--seed` | None | Random seed. With `--count > 1`, image *i* uses `seed + i - 1` |
| `--tileable` | `both` | Which axes wrap toroidally: `none`, `horizontal` (left/right only), `vertical` (top/bottom only), or `both` |
| `--pixel-scale` | 1 | Upscale each mask pixel to a `pixel_scale × pixel_scale` block using iterated Scale2x (EPX) + nearest-neighbor resample to the exact factor. Rounds diagonal edges. |
| `--target-multiple` | 512 | Snap final output side lengths to a multiple of this value. Set to 1 to disable. |
| `--size-mode` | `ceil` | How to snap: `ceil` (synth slightly larger, center-crop), `floor` (crop down), or `off`. |
| `--count`, `-n` | 1 | Number of masks to generate in this process, reusing the same loaded model. `--output` may contain a `{i}` placeholder; otherwise an index is appended before the extension |
| `--output`, `-o` | `outputs/synthesized.png` | Output PNG path |
| `--evaluate` | false | Print evaluation report (runs on the pre-upscale mask) |
| `--cache-dir` | `.mrf_cache` | Directory for the cached model |
| `--no-cache` | false | Don't read or write the model cache |
| `--refresh-cache` | false | Rebuild the model cache even if a valid one exists |

### `inpaint`

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | required | Input PNG with holes (unknown = 255) |
| `--unknown-value` | 255 | Pixel value marking unknown regions |
| `--ratio` | None | Target class proportions |
| `--lambda-ratio` | 1.0 | Weight for the global count penalty used by `--ratio` |
| `--no-multiscale` | false | Fill at full resolution directly instead of coarse-to-fine. Coarse-to-fine (default) matters most for large holes. |
| `--refine` | 3 | Gibbs refinement passes over the unknown region |
| `--temperature` | 1.0 | Sampling temperature (lower = more deterministic) |
| `--k-fallback` | 11 | KDTree nearest neighbor k for fallback |
| `--seed` | None | Random seed for reproducibility |
| `--tileable` | `both` | Which axes the fill treats as toroidal: `none`, `horizontal`, `vertical`, or `both` |
| `--pixel-scale` | 1 | Upscale each mask pixel to a `pixel_scale × pixel_scale` block via Scale2x (EPX). |
| `--output`, `-o` | `outputs/inpainted.png` | Output path |
| `--cache-dir` | `.mrf_cache` | Directory for the cached model |
| `--no-cache` | false | Don't read or write the model cache |
| `--refresh-cache` | false | Rebuild the model cache even if a valid one exists |

### `precompute`

Builds and caches the model without synthesizing anything — useful to warm the cache ahead of a batch job.

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `masks-dataset` | Path to mask dataset folder |
| `--radius` | 3 | Neighborhood radius |
| `--cache-dir` | `.mrf_cache` | Directory for the cached model |
| `--refresh-cache` | false | Rebuild even if a valid cache already exists |

## Architecture

```
src/
├── dataset.py      # Dataset loading, augmentation, potential estimation
├── model_cache.py  # Disk cache (joblib) for the trained model — class
│                   # frequencies, pairwise potentials, causal/full neighborhood tables
├── mrf_model.py    # Unary/pairwise potentials, graph-cut MAP inference (currently unused
│                   # by the CLI, kept for potential future use)
├── synthesis.py    # Nonparametric Gibbs sampler with multiscale init
├── inpainting.py   # Inpainting via the same nonparametric synthesis, constrained to holes
├── scale2x.py      # EPX (Scale2x) upscaler for mask-pixel → N×N blocks with rounded edges
├── evaluate.py     # Class freq error, GLCM, component stats, tileability
└── main.py         # CLI entry point
```

## Method

1. **Potential estimation**: Direction-specific 3×3 pairwise compatibility matrices (`-log P(c2|c1)`) estimated from all 4-connected neighbor pairs in training masks. Unary potentials come from empirical class frequencies.

2. **Nonparametric Gibbs synthesis**: For each pixel, extract its neighborhood window, look up the empirical conditional distribution from training data (with KDTree fallback for unseen patterns), and sample a label. When `--ratio` is set, the canvas starts with exact target counts and each update includes a global count penalty so final saved images stay close to the requested proportions.

3. **Multiscale initialization**: Synthesize at 1/4 resolution first, upsample via nearest neighbor, then refine at full resolution to prevent large-scale structural failures.

4. **Gibbs refinement**: Multiple full sweeps in random order using the complete (non-causal) neighborhood to improve global consistency.

5. **Configurable tileability**: Neighborhood lookups wrap modularly along whichever axes `--tileable` selects (`none`/`horizontal`/`vertical`/`both`); non-tileable axes clamp to the edge instead. `both` (the default) makes output masks tile seamlessly in every direction.

6. **Inpainting**: Unknown pixels are randomly initialized, then filled with the *same* coarse-to-fine + Gibbs-refinement algorithm as `synthesize` (steps 2-4 above), except every pass only ever resamples pixels inside the unknown region — known pixels are fixed context, never touched. The result is a genuine sample from the texture model rather than a smoothness-maximizing MAP estimate, so it looks like real bark texture instead of a flat blob.

7. **Model caching**: The pieces derived only from the training dataset and `--radius` (class frequencies, pairwise potentials, causal/full neighborhood tables + compact unique-pattern KDTrees) are cached to disk after the first build and loaded on later runs — see "Model caching" above.
