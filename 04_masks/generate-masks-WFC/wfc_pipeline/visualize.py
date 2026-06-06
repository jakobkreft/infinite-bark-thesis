from __future__ import annotations

from pathlib import Path
from typing import Sequence
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import PALETTE, ensure_dir


def _font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Default fallback is portable and available.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def mask_to_rgb(mask: np.ndarray, palette: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    palette = palette or PALETTE
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        out[mask == cls] = color
    unknown = (out.sum(axis=2) == 0)
    out[unknown] = palette.get(255, (255, 0, 255))
    return out


def save_mask_png(mask: np.ndarray, path: Path, scale: int = 8) -> None:
    ensure_dir(path.parent)
    rgb = mask_to_rgb(mask)
    img = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
    img.save(path)


def save_mask_gallery(
    masks: Sequence[np.ndarray],
    path: Path,
    titles: Sequence[str] | None = None,
    cols: int = 8,
    scale: int = 6,
    max_items: int = 64,
    title: str | None = None,
) -> None:
    ensure_dir(path.parent)
    if not masks:
        raise ValueError("No masks provided for gallery")

    masks = list(masks[:max_items])
    if titles is None:
        titles = [f"#{i}" for i in range(len(masks))]
    else:
        titles = list(titles[: len(masks)])

    max_h = max(m.shape[0] for m in masks)
    max_w = max(m.shape[1] for m in masks)
    rows = math.ceil(len(masks) / cols)

    cell_w = max_w * scale + 18
    cell_h = max_h * scale + 30
    top_pad = 36 if title else 12

    canvas = Image.new("RGB", (cols * cell_w + 12, rows * cell_h + top_pad + 12), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((12, 8), title, fill=(20, 20, 20), font=_font(18))

    for i, (mask, label) in enumerate(zip(masks, titles)):
        r = i // cols
        c = i % cols
        x0 = 12 + c * cell_w
        y0 = top_pad + r * cell_h

        h, w = mask.shape
        rgb = mask_to_rgb(mask)
        tile = Image.fromarray(rgb, mode="RGB").resize((w * scale, h * scale), resample=Image.NEAREST)
        canvas.paste(tile, (x0, y0))
        draw.rectangle((x0 - 1, y0 - 1, x0 + w * scale, y0 + h * scale), outline=(180, 180, 180), width=1)
        draw.text((x0, y0 + h * scale + 4), str(label), fill=(40, 40, 40), font=_font(12))

    canvas.save(path)


def save_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    path: Path,
    title: str,
    y_label: str,
    bar_color: tuple[int, int, int] = (66, 133, 244),
) -> None:
    ensure_dir(path.parent)
    width, height = 1200, 700
    left, right, top, bottom = 90, 40, 80, 150
    plot_w = width - left - right
    plot_h = height - top - bottom

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((left, 24), title, fill=(20, 20, 20), font=_font(24))
    draw.line((left, top, left, top + plot_h), fill=(50, 50, 50), width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(50, 50, 50), width=2)

    if not values:
        canvas.save(path)
        return

    vmax = max(float(v) for v in values)
    vmax = vmax if vmax > 0 else 1.0

    n = len(values)
    gap = max(4, int(plot_w * 0.02 / max(1, n // 10 + 1)))
    bar_w = max(4, (plot_w - gap * (n + 1)) // n)

    for i, (lbl, val) in enumerate(zip(labels, values)):
        x0 = left + gap + i * (bar_w + gap)
        x1 = x0 + bar_w
        bar_h = int((float(val) / vmax) * plot_h)
        y0 = top + plot_h - bar_h
        draw.rectangle((x0, y0, x1, top + plot_h), fill=bar_color, outline=(45, 90, 170))

        txt = f"{val:.2f}" if isinstance(val, float) else str(val)
        draw.text((x0, y0 - 16), txt, fill=(30, 30, 30), font=_font(11))

        lx = x0
        ly = top + plot_h + 8
        draw.text((lx, ly), str(lbl), fill=(40, 40, 40), font=_font(12))

    draw.text((18, top + plot_h // 2), y_label, fill=(50, 50, 50), font=_font(14))
    canvas.save(path)


def save_histogram(
    values: Sequence[float],
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    bins: int = 25,
    color: tuple[int, int, int] = (76, 175, 80),
    log_x: bool = False,
) -> None:
    ensure_dir(path.parent)
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        save_text_report([title, "No data"], path)
        return

    work = arr.copy()
    if log_x:
        work = np.log10(np.clip(work, 1e-6, None))

    hist, edges = np.histogram(work, bins=bins)
    labels = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if log_x:
            lo = 10 ** lo
            hi = 10 ** hi
        labels.append(f"{lo:.1f}-{hi:.1f}")

    save_bar_chart(labels, hist.tolist(), path, title=title, y_label=y_label, bar_color=color)


def save_heatmap(
    matrix: np.ndarray,
    path: Path,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    normalize_rows: bool = False,
) -> None:
    ensure_dir(path.parent)
    mat = np.array(matrix, dtype=np.float64)
    if normalize_rows:
        rs = mat.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        mat = mat / rs

    rows, cols = mat.shape
    cell = 90
    left, top = 180, 120
    width = left + cols * cell + 80
    height = top + rows * cell + 120

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 24), title, fill=(20, 20, 20), font=_font(24))

    vmin = float(mat.min())
    vmax = float(mat.max())
    denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0

    for r in range(rows):
        for c in range(cols):
            val = float(mat[r, c])
            t = (val - vmin) / denom
            color = _viridis_like(t)
            x0 = left + c * cell
            y0 = top + r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=(200, 200, 200))
            txt = f"{val:.3f}" if normalize_rows else f"{int(round(val))}"
            tw, th = draw.textbbox((0, 0), txt, font=_font(14))[2:4]
            draw.text((x0 + (cell - tw) / 2, y0 + (cell - th) / 2), txt, fill=(0, 0, 0), font=_font(14))

    for i, lbl in enumerate(col_labels):
        draw.text((left + i * cell + 8, top - 28), str(lbl), fill=(30, 30, 30), font=_font(14))
    for i, lbl in enumerate(row_labels):
        draw.text((20, top + i * cell + 30), str(lbl), fill=(30, 30, 30), font=_font(14))

    canvas.save(path)


def _viridis_like(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    # Simple handcrafted blue-green-yellow gradient.
    if t < 0.5:
        a = t / 0.5
        r = int(35 + a * (53 - 35))
        g = int(40 + a * (183 - 40))
        b = int(130 + a * (121 - 130))
    else:
        a = (t - 0.5) / 0.5
        r = int(53 + a * (252 - 53))
        g = int(183 + a * (231 - 183))
        b = int(121 + a * (37 - 121))
    return (r, g, b)


def save_text_report(lines: Sequence[str], path: Path, title: str | None = None) -> None:
    ensure_dir(path.parent)
    width = 1200
    line_h = 24
    top_pad = 56 if title else 20
    height = top_pad + line_h * (len(lines) + 2)
    canvas = Image.new("RGB", (width, max(height, 220)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((20, 16), title, fill=(15, 15, 15), font=_font(24))

    y = top_pad
    for line in lines:
        draw.text((20, y), line, fill=(35, 35, 35), font=_font(16))
        y += line_h
    canvas.save(path)


def save_pattern_gallery(
    patterns: Sequence[np.ndarray],
    weights: Sequence[int],
    path: Path,
    title: str,
    max_items: int = 64,
    scale: int = 20,
) -> None:
    if not patterns:
        raise ValueError("No patterns for gallery")
    n = min(max_items, len(patterns))
    idx = np.argsort(np.array(weights))[::-1][:n].tolist()
    chosen = [patterns[i] for i in idx]
    titles = [f"id={i} w={weights[i]}" for i in idx]
    save_mask_gallery(chosen, path=path, titles=titles, cols=8, scale=scale, max_items=n, title=title)


def save_fraction_boxplot_like(
    fractions: dict[int, Sequence[float]],
    path: Path,
    title: str,
) -> None:
    # Lightweight percentile visualization for class fractions.
    ensure_dir(path.parent)
    width, height = 1000, 560
    left, top, bottom = 130, 90, 90
    plot_h = height - top - bottom
    plot_w = width - left - 40

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), title, fill=(20, 20, 20), font=_font(24))

    cls_ids = sorted(fractions.keys())
    n = len(cls_ids)
    spacing = plot_w // max(1, n)

    draw.line((left, top, left, top + plot_h), fill=(40, 40, 40), width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(40, 40, 40), width=2)

    def y_of(v: float) -> int:
        return int(top + plot_h - v * plot_h)

    for cls_idx, cls in enumerate(cls_ids):
        vals = np.array(fractions[cls], dtype=np.float64)
        if vals.size == 0:
            continue
        q10, q25, q50, q75, q90 = np.percentile(vals, [10, 25, 50, 75, 90]).tolist()
        x = left + cls_idx * spacing + spacing // 2
        box_w = max(40, spacing // 3)

        color = PALETTE.get(cls, (120, 120, 120))

        draw.line((x, y_of(q10), x, y_of(q90)), fill=(80, 80, 80), width=2)
        draw.rectangle((x - box_w // 2, y_of(q75), x + box_w // 2, y_of(q25)), fill=color, outline=(50, 50, 50))
        draw.line((x - box_w // 2, y_of(q50), x + box_w // 2, y_of(q50)), fill=(20, 20, 20), width=2)

        draw.text((x - 26, top + plot_h + 14), f"class {cls}", fill=(30, 30, 30), font=_font(14))
        draw.text((x - 36, y_of(q90) - 18), f"{q90:.2f}", fill=(40, 40, 40), font=_font(12))

    canvas.save(path)
