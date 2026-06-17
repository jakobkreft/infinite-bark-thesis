# 05 — Diffusion

Bark texture synthesis with DiffInfinite. The diffusion code is the
[`diffinfinite-bark`](https://github.com/jakobkreft/diffinfinite-bark) fork of
[`marcoaversa/diffinfinite`](https://github.com/marcoaversa/diffinfinite),
included here as a **git submodule**.

| Folder | What |
|--------|------|
| [diffinfinite-bark/](diffinfinite-bark) | The diffusion model (submodule) — train on the patch dataset, sample new textures conditioned on semantic masks. |
| [prepare-patches/](prepare-patches) | Build the DiffInfinite training dataset: cut the segmented bark images + masks into `bark_XXXX.jpg` / `bark_XXXX_mask.png` patches. |

After cloning, fetch the submodule with `git submodule update --init --recursive`.

Training data comes from [03_segmentation](../03_segmentation) (bark images + their
damage masks) via [prepare-patches](prepare-patches). Conditioning masks at
generation time come from [04_masks](../04_masks). Setup/training notes
(DiffInfinite WSL install, bark-dataset-format conversion) are in
[`../02_preprocess/DATASET_PIPELINE_NOTES.md`](../02_preprocess/DATASET_PIPELINE_NOTES.md).
