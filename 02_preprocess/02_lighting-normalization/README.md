# 02 — Lighting normalization

Remove the vertical brightness gradient (dark at top/bottom, bright in the
middle) caused by photographing a round log. Works on the CIELAB **L** channel
only, so color and texture are preserved.

`normalize_lighting.py` — estimates a smoothed per-row luminance profile and
applies a per-row gain to flatten it.

Run: `python normalize_lighting.py --input <dir> --output <dir>`
