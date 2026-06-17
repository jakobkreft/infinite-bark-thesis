# 02 — Preprocess

Turn raw cylindrical-log photos into clean, flat, evenly-lit bark images ready for
segmentation. Run the steps in order:

| Step | Folder | Does |
|------|--------|------|
| 1 | [01_log-segmentation](01_log-segmentation) | Segment the log out of the background (1-class DeepLabV3) to get a log mask. |
| 2 | [02_rotate-and-crop](02_rotate-and-crop) | Use the log mask to straighten, crop to the log, and unwrap the cylinder to a flat image. |
| 3 | [03_lighting-normalization](03_lighting-normalization) | Remove the vertical brightness gradient (dark top/bottom). |
| 4 | [04_crop-dataset](04_crop-dataset) | Drop the deformed top/bottom edges, keep the middle. |

Patch extraction into DiffInfinite training format happens **after** segmentation —
see [05_diffusion/prepare-patches](../05_diffusion/prepare-patches).

[DATASET_PIPELINE_NOTES.md](DATASET_PIPELINE_NOTES.md) — original notes, including the
DiffInfinite WSL setup tutorial and the bark-dataset-format conversion checklist.
