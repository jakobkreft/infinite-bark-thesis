# 02 — Dataset prep

Turn the hand-painted labels into a model-ready segmentation dataset
(`step0` raw labels → `step1` full-res image/mask pairs → `step2` train/val/test split).

| File | Does |
|------|------|
| `prepare_dataset.py` | step0→step1: crop each image to the labelled region and upscale its low-res mask to full resolution. |
| `prepare_dataset_step2.py` | step1→step2: stratified train/val/test split (plus an inference set). |
| `visualize_dataset.py` | Publication figures for step1 (sample / mask / overlay grids, class + resolution distributions). |
| `visualize_dataset_step2.py` | Publication figures for the train/val/test split. |
| `visualize_mask_comparison.py` | Compare NEAREST vs Scale2x mask upscaling side by side. |

Generated `dataset_step*` folders are not committed. The full-res pairs are later cut
into training patches in
[05_diffusion/prepare-patches](../../05_diffusion/prepare-patches), which is what the
segmenter in [03_ai-segment-bark](../03_ai-segment-bark) and the diffusion model both train on.
