"""
Scale2x (EPX) integer upscaler for discrete-label masks.

EPX compares a pixel's 4-connected neighbors to decide whether each of its
2x2 output corners inherits from a matching neighbor (rounding diagonals)
or keeps the center pixel. It works directly on integer label arrays since
it uses equality only — no interpolation.

To reach an arbitrary integer factor, apply Scale2x repeatedly until the
intermediate resolution meets or exceeds the target, then nearest-neighbor
resample (up or down) to the exact target size. NN is used so labels stay
discrete.
"""

import math
import numpy as np


def scale2x(mask: np.ndarray) -> np.ndarray:
    """One EPX pass. Input (H, W) integer array -> (2H, 2W) same dtype."""
    if mask.ndim != 2:
        raise ValueError(f"scale2x expects 2D array, got shape {mask.shape}")

    h, w = mask.shape
    P = mask
    padded = np.pad(P, 1, mode="edge")
    A = padded[:-2, 1:-1]
    B = padded[1:-1, 2:]
    C = padded[1:-1, :-2]
    D = padded[2:, 1:-1]

    tl = np.where((C == A) & (C != D) & (A != B), A, P)
    tr = np.where((A == B) & (A != C) & (B != D), B, P)
    bl = np.where((D == C) & (D != B) & (C != A), C, P)
    br = np.where((B == D) & (B != A) & (D != C), D, P)

    out = np.empty((2 * h, 2 * w), dtype=mask.dtype)
    out[0::2, 0::2] = tl
    out[0::2, 1::2] = tr
    out[1::2, 0::2] = bl
    out[1::2, 1::2] = br
    return out


def _nn_resample(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbor resample that preserves integer labels."""
    h, w = mask.shape
    row_idx = np.round(np.linspace(0, h - 1, target_h)).astype(np.int64)
    col_idx = np.round(np.linspace(0, w - 1, target_w)).astype(np.int64)
    return mask[row_idx][:, col_idx]


def compute_aligned_dims(h_req, w_req, pixel_scale, target_multiple, mode):
    """
    Pick synthesis dims (h_synth, w_synth) and final crop dims (h_final, w_final)
    so that h_final and w_final are multiples of target_multiple. The final output
    is produced by synthesizing at (h_synth, w_synth), EPX-upscaling by pixel_scale,
    then center-cropping to (h_final, w_final) — preserving the exact 1:pixel_scale
    mapping everywhere inside the crop.

    mode:
      'ceil'  — round requested size up to the next multiple (may synth slightly larger)
      'floor' — round down (final is smaller than requested; at least one multiple)
      'off'   — no alignment; returns inputs unchanged
    """
    if pixel_scale <= 1 or target_multiple <= 1 or mode == "off":
        return h_req, w_req, h_req * pixel_scale, w_req * pixel_scale

    req_h = h_req * pixel_scale
    req_w = w_req * pixel_scale

    if mode == "ceil":
        h_final = math.ceil(req_h / target_multiple) * target_multiple
        w_final = math.ceil(req_w / target_multiple) * target_multiple
    elif mode == "floor":
        h_final = max(target_multiple, (req_h // target_multiple) * target_multiple)
        w_final = max(target_multiple, (req_w // target_multiple) * target_multiple)
    else:
        raise ValueError(f"Unknown size mode: {mode!r}")

    h_synth = math.ceil(h_final / pixel_scale)
    w_synth = math.ceil(w_final / pixel_scale)
    return h_synth, w_synth, h_final, w_final


def center_crop(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop a 2D array to (target_h, target_w). Requires target <= shape."""
    h, w = arr.shape
    if target_h > h or target_w > w:
        raise ValueError(
            f"center_crop target ({target_h}, {target_w}) exceeds array {arr.shape}"
        )
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return arr[top:top + target_h, left:left + target_w].copy()


def upscale_epx_to_factor(mask: np.ndarray, factor: int) -> np.ndarray:
    """
    Upscale a label mask so each input pixel becomes a factor x factor block,
    with EPX-rounded diagonals. factor must be a positive integer.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return mask.copy()

    h, w = mask.shape
    target_h, target_w = h * factor, w * factor

    n_passes = max(1, math.ceil(math.log2(factor)))
    current = mask
    for _ in range(n_passes):
        current = scale2x(current)

    if current.shape != (target_h, target_w):
        current = _nn_resample(current, target_h, target_w)

    print(
        f"EPX upscale: {h}x{w} -> {target_h}x{target_w} "
        f"({n_passes} EPX passes + NN resample to exact factor {factor})"
    )
    return current
