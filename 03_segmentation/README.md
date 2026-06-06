# 03 — Segmentation

Label and segment bark-surface damage on the unwrapped images. This produces the
ground-truth masks that the rest of the project learns from. Order:

| Step | Folder | Does |
|------|--------|------|
| 1 | [01_hand-segment-bark](01_hand-segment-bark) | GUI tool to hand-paint masks (creates the labels). |
| 2 | [02_dataset-seg-bark](02_dataset-seg-bark) | Build a train/val/test dataset from the hand labels. |
| 3 | [03_ai-segment-bark](03_ai-segment-bark) | DeepLabV3+ model — train, evaluate, infer (the main segmenter). |

**Classes:** `0` background · `1` slepice (pruning wounds / knots) · `2` mechanical damage.
