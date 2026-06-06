# 04 — Patches

Cut the full images + masks into square patches, then package them the way
DiffInfinite expects.

| File | Does |
|------|------|
| `extract_patches.py` | Tile each image/mask into square patches covering the whole image, at original, 1024, and 512 resolutions. |
| `prepare_diffinfinite.py` | Rename patches to `bark_XXXX.jpg` / `bark_XXXX_mask.png` in a flat folder and write `class_to_int.yml` (bark=0, knot=1, defect=2). |

Run: `python extract_patches.py` then `python prepare_diffinfinite.py`.
Output is the dataset fed to [05_diffusion](../../05_diffusion).
