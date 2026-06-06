# 02 — Preprocess

Turn raw cylindrical-log photos (and their hand masks) into clean square
patches ready for the diffusion model. Run the steps in order:

| Step | Folder | Does |
|------|--------|------|
| 1 | [01_rotate-and-crop](01_rotate-and-crop) | Straighten the log, crop to it, unwrap the cylinder to a flat image. |
| 2 | [02_lighting-normalization](02_lighting-normalization) | Remove the vertical brightness gradient (dark top/bottom). |
| 3 | [03_crop-dataset](03_crop-dataset) | Drop the deformed top/bottom edges, keep the middle. |
| 4 | [04_patches](04_patches) | Cut into square patches (orig/1024/512) and convert to DiffInfinite layout. |

[DATASET_PIPELINE_NOTES.md](DATASET_PIPELINE_NOTES.md) — original notes, including the
DiffInfinite WSL setup tutorial and the bark-dataset-format conversion checklist.
