# 02 — Dataset prep

Turn the hand-painted labels into a model-ready segmentation dataset
(`step0` raw labels → `step1` full-res pairs → `step2` train/val/test split).

| File | Does |
|------|------|
| `prepare_dataset.py` | step0→step1: crop images to the labeled region, upscale the low-res masks to full resolution (NEAREST). |
| `prepare_dataset_smooth.py` | Same as above but smooth Scale2x mask upscaling (`step1_smooth`). |
| `prepare_dataset_step2.py` | Stratified train/val/test split + an inference set. |
| `visualize_dataset.py`, `visualize_dataset_step2.py` | Sanity-check overlays of images + masks. |
| `visualize_mask_comparison.py` | Compare NEAREST vs Scale2x mask upscaling side by side. |

Generated `dataset_step*` folders are not committed. Output feeds
[03_ai-segment-bark](../03_ai-segment-bark).
