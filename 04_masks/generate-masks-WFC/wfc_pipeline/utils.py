from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Iterable
import numpy as np

# Fixed color map for reproducibility.
PALETTE = {
    0: (22, 24, 29),      # background
    1: (235, 167, 52),    # slepice
    2: (67, 142, 247),    # mehanske poskodbe
    255: (220, 0, 120),   # unknown / debug
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def class_name(cls: int) -> str:
    return {0: "background", 1: "slepice", 2: "mehanske_poskodbe"}.get(cls, f"class_{cls}")


def mask_values_present(mask: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(mask).tolist())
