# 03 — Crop edges

Crop away the deformed top and bottom ~1/8 of each image (and its mask), keeping
the middle band where the unwrapped bark is least distorted.

`crop_dataset.py` — crops images + masks together and writes them to an output dataset.

Run: `python crop_dataset.py --image_dir <dir> --mask_dir <dir> --output_dir <dir>`
