# Prepare patches — DiffInfinite training data

Cut the segmented bark images + masks into square patches and write them directly
in DiffInfinite flat-folder format (`bark_XXXX.jpg` + `bark_XXXX_mask.png` +
`class_to_int.yml`) at three resolutions (original / 1024 / 512).

Input is the preprocessed bark images from [02_preprocess](../../02_preprocess)
paired with their damage masks from [03_segmentation](../../03_segmentation);
the output is the dataset that the [diffinfinite-bark](../diffinfinite-bark)
submodule trains on. (The script also trims distorted top/bottom edges and
controls patch rows/overlap — edit the CONFIG block at the top.)

Run: `python make_diffinf_patches.py`
