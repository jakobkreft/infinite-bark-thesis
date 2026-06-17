# 01 — Hand segmentation

Tkinter GUI to hand-paint bark segmentation masks over the unwrapped images.

**Classes:** `0` background · `1` slepice / pruning wounds (cyan) · `2` mechanical damage (orange).

| File | Does |
|------|------|
| `segmentation_tool_with_epx.py` | **Main labeling tool.** Paints a coarse low-res mask (1 mask pixel ≈ 50×50 image pixels — well suited to the MRF generator) and also saves a full-resolution mask upscaled with the label-preserving Scale2x (EPX) algorithm. |

Run: `python segmentation_tool_with_epx.py`, then pick the image and mask folders in the UI.
