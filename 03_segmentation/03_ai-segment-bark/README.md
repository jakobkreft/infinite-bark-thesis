# 03 — AI bark segmentation

Train a semantic segmentation network to label bark-surface damage, using
[segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
(U-Net by default; DeepLabV3+ / UnetPlusPlus / FPN selectable in the CONFIG block)
with a ResNet encoder pretrained on ImageNet.

**Classes:** `0` bark · `1` knot / slepice (cyan) · `2` mechanical damage (orange).

`train_seg.py` runs the whole thing end to end:
1. **5-fold cross-validation** on the train set → mean/std metrics.
2. **Final model** trained on (almost) all train data, with a small val split for best-checkpoint selection.
3. **Evaluation** on the held-out test set.
4. Saves thesis-ready plots (training curves, CV summary, confusion matrix, per-class IoU, qualitative grid) + JSON metrics to `segmentation_results/`.

It reads the 512×512 patch dataset `diffinfinite-bark_512-train/` and
`diffinfinite-bark_512-test/` (`bark_XXXX.jpg` + `bark_XXXX_mask.png`) produced by
[05_diffusion/prepare-patches](../../05_diffusion/prepare-patches). Edit the CONFIG
block at the top of the script for architecture, encoder, image size, epochs, etc.

Run: `python train_seg.py`
