# 01 — Log segmentation

Isolate the tree log from the background so the next step can crop to it. This is a
**single-class** problem (log vs. background) — and is **not** the bark-damage
segmentation in [03_segmentation](../../03_segmentation), which comes later and labels
defects on the already-cropped bark.

| File | Does |
|------|------|
| `segment_tool.py` | Tkinter tool: click points around the log to hand-label a small training set (image + binary log mask). |
| `train.py` | Fine-tune a 1-class DeepLabV3-ResNet50 on the hand-labelled set (`dataset_small/{train,test}`), saving `best_model.pth`. |
| `inference.py` | Run the trained model over all photos to predict log masks (`dataset_big/images` → `dataset_big/masks`) + overlays. |

Flow: hand-label a few logs → train → predict masks for every image. Those predicted
masks feed [02_rotate-and-crop](../02_rotate-and-crop), which uses them to straighten,
crop and unwrap each log.
