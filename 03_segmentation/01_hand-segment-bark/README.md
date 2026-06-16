# 01 — Hand segmentation

Tkinter GUI tools to hand-paint segmentation masks over images.

| File | Does |
|------|------|
| `segmentation_tool_with_epx.py` | **Main labeling tool.** it saves the low res masks suitable for MRF and also masks at same resolution as the images upscaled with scale2x epx algorithm. |

Run: `python segmentation_tool_with_epx.py`, then pick the image and mask folders in the UI.