# 01 — Rotate & crop & unwrap

Turn photos of cylindrical logs into flat, level bark images. Works on
image + mask pairs (the mask marks the log region).

| File | Does |
|------|------|
| `rotate_and_crop_logs.py` | **Main.** Fit lines to the log's top/bottom edges, rotate it level, crop to the log. |
| `unwrap_logs.py` | Flatten the cylindrical perspective (arc → flat strip). Run after cropping. |
| `rotate_logs.py` | Rotate-only (earlier version, superseded by `rotate_and_crop_logs.py`). |
| `visualize_lines.py` | Debug: draw the fitted top/bottom edge lines on a random mask. |
| `test_cv2.py` | Sanity check — OpenCV install + that image/mask paths load. |

Run: `python rotate_and_crop_logs.py` then `python unwrap_logs.py`
(expects `dataset_big/images` + `dataset_big/masks`; outputs not committed).
