# WFC Mask Research Pipeline

This repository contains a phase-based, reproducible pipeline for:

1. Loading categorical mask data (`0=background`, `1=slepice`, `2=mehanske poskodbe`).
2. Computing dataset statistics and saving transparent analysis artifacts.
3. Extracting overlapping patterns for Wave Function Collapse (WFC).
4. Generating new masks at arbitrary size (default `100x100`).
5. Comparing generated masks against original dataset statistics.

## Run

```bash
python run_pipeline.py \
  --dataset-dir masks-unwrap \
  --output-dir outputs \
  --pattern-size 3 \
  --min-pattern-weight 2 \
  --max-patterns 512 \
  --target-size 100 \
  --tile-mode none \
  --num-generations 3 \
  --seed 42
```

Tileability options:
- `--tile-mode none`: normal bounded generation (default).
- `--tile-mode torus`: repeats seamlessly in both X and Y.
- `--tile-mode cylinder_x`: repeats seamlessly only in X.
- `--tile-mode cylinder_y`: repeats seamlessly only in Y.

## Phases

- `phase1`: dataset ingestion and manifest.
- `phase2`: statistics and visualization outputs.
- `phase3`: WFC pattern library extraction and diagnostics.
- `phase4`: WFC generation with saved intermediate snapshots.
- `phase5`: generated-vs-real quantitative comparison.

Run specific phases:

```bash
python run_pipeline.py --phases phase1,phase2
```

## Output structure

All artifacts are saved under `outputs/`:

- `phase1_dataset/`: dataset manifest, overview, sample gallery.
- `phase2_statistics/`: statistical summaries and PNG graphs.
- `phase3_patterns/`: pattern library (`npz`), compatibility reports, visuals.
- `phase4_wfc/`: generated masks and step-wise WFC snapshots.
- `phase5_comparison/`: generated dataset stats and drift comparison plots.

## Notes

- The pipeline is designed to be transparent and modular for iterative research.
- Only local dependencies are used (`numpy`, `Pillow`) to keep reproducibility simple.
