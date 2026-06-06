# 01 — Hand segmentation

Tkinter GUI tools to hand-paint segmentation masks over images.

| File | Does |
|------|------|
| `segmentation_tool_lowres.py` | **Main labeling tool.** Coarse 20-column grid, 3 classes (background / slepice / mechanical damage). Produces the low-res masks used downstream. |
| `segmentation_tool.py` | Full-resolution brush tool with a finer 8-class scheme (earlier / more detailed labeling). |

Run: `python segmentation_tool_lowres.py`, then pick the image and mask folders in the UI.
The low-res masks feed [02_dataset-seg-bark](../02_dataset-seg-bark).
