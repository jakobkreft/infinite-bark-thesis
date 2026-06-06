from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Callable
import math
import numpy as np

from .patterns import PatternLibrary


SnapshotCallback = Callable[[int, np.ndarray, str], None]

VALID_TILE_MODES = {"none", "torus", "cylinder_x", "cylinder_y"}


def normalize_tile_mode(tile_mode: str) -> str:
    mode = str(tile_mode).strip().lower()
    aliases = {
        "periodic": "torus",
        "cylinder": "cylinder_x",
        "cyl_x": "cylinder_x",
        "cyl_y": "cylinder_y",
        "x": "cylinder_x",
        "y": "cylinder_y",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_TILE_MODES:
        raise ValueError(
            f"Unsupported tile_mode={tile_mode!r}. Expected one of {sorted(VALID_TILE_MODES)}"
        )
    return mode


@dataclass
class WFCGenerationResult:
    success: bool
    mask: np.ndarray | None
    collapsed_pattern_ids: np.ndarray | None
    steps: int
    restart_index: int
    message: str


class WFCGenerator:
    def __init__(
        self,
        library: PatternLibrary,
        target_height: int,
        target_width: int,
        seed: int = 42,
        tile_mode: str = "none",
    ) -> None:
        self.lib = library
        self.n = library.pattern_size
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.tile_mode = normalize_tile_mode(tile_mode)

        self.periodic_x = self.tile_mode in {"torus", "cylinder_x"}
        self.periodic_y = self.tile_mode in {"torus", "cylinder_y"}

        if self.target_height <= 0 or self.target_width <= 0:
            raise ValueError("target_height and target_width must be > 0")
        if (not self.periodic_y) and self.target_height < self.n:
            raise ValueError(
                f"Non-periodic Y requires target_height >= pattern_size ({self.target_height} < {self.n})"
            )
        if (not self.periodic_x) and self.target_width < self.n:
            raise ValueError(
                f"Non-periodic X requires target_width >= pattern_size ({self.target_width} < {self.n})"
            )

        # Periodic axis: each pixel location is a pattern anchor (wrap-around overlap).
        self.patch_h = self.target_height if self.periodic_y else self.target_height - self.n + 1
        self.patch_w = self.target_width if self.periodic_x else self.target_width - self.n + 1

        self.weights = library.weights.astype(np.float64)
        self.num_patterns = int(library.patterns.shape[0])
        self.seed = int(seed)

    def generate(
        self,
        max_restarts: int = 20,
        snapshot_interval: int = 150,
        snapshot_cb: SnapshotCallback | None = None,
    ) -> WFCGenerationResult:
        for restart in range(max_restarts):
            rng = np.random.default_rng(self.seed + restart)
            wave = np.ones((self.patch_h, self.patch_w, self.num_patterns), dtype=bool)
            domain_sizes = np.full((self.patch_h, self.patch_w), self.num_patterns, dtype=np.int32)

            if snapshot_cb is not None:
                snapshot_cb(0, wave, f"restart_{restart}_start")

            steps = 0
            while True:
                if np.all(domain_sizes == 1):
                    ids = np.argmax(wave, axis=2)
                    mask = self.reconstruct_from_pattern_ids(ids)
                    if snapshot_cb is not None:
                        snapshot_cb(steps, wave, f"restart_{restart}_success")
                    return WFCGenerationResult(
                        success=True,
                        mask=mask,
                        collapsed_pattern_ids=ids,
                        steps=steps,
                        restart_index=restart,
                        message="success",
                    )

                pos = self._select_cell(domain_sizes, wave, rng)
                if pos is None:
                    break
                y, x = pos

                success = self._collapse_cell(wave, domain_sizes, y, x, rng)
                steps += 1
                if not success:
                    if snapshot_cb is not None:
                        snapshot_cb(steps, wave, f"restart_{restart}_collapse_failed")
                    break

                queue = deque([(y, x)])
                ok = self._propagate(wave, domain_sizes, queue)
                if (snapshot_cb is not None) and (steps % max(1, snapshot_interval) == 0):
                    snapshot_cb(steps, wave, f"restart_{restart}_progress")
                if not ok:
                    if snapshot_cb is not None:
                        snapshot_cb(steps, wave, f"restart_{restart}_contradiction")
                    break

        return WFCGenerationResult(
            success=False,
            mask=None,
            collapsed_pattern_ids=None,
            steps=0,
            restart_index=max_restarts - 1,
            message=f"failed_after_{max_restarts}_restarts",
        )

    def _select_cell(
        self,
        domain_sizes: np.ndarray,
        wave: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int] | None:
        unresolved = domain_sizes > 1
        if not np.any(unresolved):
            return None

        min_size = int(domain_sizes[unresolved].min())
        candidates = np.argwhere(domain_sizes == min_size)
        if candidates.shape[0] == 1:
            y, x = candidates[0]
            return (int(y), int(x))

        # When candidate set is limited, entropy tie-break gives cleaner textures.
        if min_size <= 4 and candidates.shape[0] <= 256:
            best_entropy = float("inf")
            best_pos = None
            for y, x in candidates:
                ent = self._entropy(wave[int(y), int(x)])
                ent += float(rng.random()) * 1e-8
                if ent < best_entropy:
                    best_entropy = ent
                    best_pos = (int(y), int(x))
            if best_pos is not None:
                return best_pos

        pick = int(rng.integers(0, candidates.shape[0]))
        y, x = candidates[pick]
        return (int(y), int(x))

    def _entropy(self, domain: np.ndarray) -> float:
        ids = np.flatnonzero(domain)
        if ids.size <= 1:
            return 0.0
        w = self.weights[ids]
        w_sum = float(w.sum())
        if w_sum <= 0:
            return 0.0
        # Weighted Shannon entropy.
        return math.log(w_sum) - float(np.sum(w * np.log(np.clip(w, 1e-12, None)))) / w_sum

    def _collapse_cell(
        self,
        wave: np.ndarray,
        domain_sizes: np.ndarray,
        y: int,
        x: int,
        rng: np.random.Generator,
    ) -> bool:
        domain = wave[y, x]
        ids = np.flatnonzero(domain)
        if ids.size == 0:
            return False
        if ids.size == 1:
            domain_sizes[y, x] = 1
            return True

        w = self.weights[ids]
        w_sum = float(w.sum())
        if w_sum <= 0:
            probs = np.ones(ids.size, dtype=np.float64) / ids.size
        else:
            probs = w / w_sum

        chosen = int(rng.choice(ids, p=probs))
        domain[:] = False
        domain[chosen] = True
        domain_sizes[y, x] = 1
        return True

    def _neighbor_index(self, y: int, x: int, dy: int, dx: int) -> tuple[int, int] | None:
        ny = y + dy
        nx = x + dx

        if ny < 0 or ny >= self.patch_h:
            if self.periodic_y:
                ny %= self.patch_h
            else:
                return None

        if nx < 0 or nx >= self.patch_w:
            if self.periodic_x:
                nx %= self.patch_w
            else:
                return None

        return int(ny), int(nx)

    def _propagate(
        self,
        wave: np.ndarray,
        domain_sizes: np.ndarray,
        queue: deque[tuple[int, int]],
    ) -> bool:
        directions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        while queue:
            y, x = queue.popleft()
            domain = wave[y, x]
            dsize = int(domain_sizes[y, x])
            if dsize <= 0:
                return False

            for direction, (dy, dx) in directions.items():
                neighbor = self._neighbor_index(y, x, dy, dx)
                if neighbor is None:
                    continue
                ny, nx = neighbor

                neighbor_domain = wave[ny, nx]
                neighbor_size = int(domain_sizes[ny, nx])
                if neighbor_size <= 0:
                    return False

                if dsize == 1:
                    pid = int(np.argmax(domain))
                    allowed = self.lib.compat[direction][pid]
                else:
                    # Fast vectorized union of compat rows for current domain.
                    allowed = np.any(self.lib.compat[direction][domain], axis=0)

                new_neighbor_domain = neighbor_domain & allowed
                new_size = int(new_neighbor_domain.sum())
                if new_size <= 0:
                    return False

                if new_size == neighbor_size:
                    continue

                wave[ny, nx] = new_neighbor_domain
                domain_sizes[ny, nx] = new_size
                queue.append((ny, nx))

        return True

    def reconstruct_from_pattern_ids(self, ids: np.ndarray) -> np.ndarray:
        class_count = max(1, int(np.max(self.lib.patterns)) + 1)
        votes = np.zeros((self.target_height, self.target_width, class_count), dtype=np.int32)

        for y in range(self.patch_h):
            for x in range(self.patch_w):
                pid = int(ids[y, x])
                pattern = self.lib.patterns[pid]
                for dy in range(self.n):
                    yy = y + dy
                    if self.periodic_y:
                        yy %= self.target_height
                    elif yy < 0 or yy >= self.target_height:
                        continue

                    for dx in range(self.n):
                        xx = x + dx
                        if self.periodic_x:
                            xx %= self.target_width
                        elif xx < 0 or xx >= self.target_width:
                            continue

                        pv = int(pattern[dy, dx])
                        if 0 <= pv < class_count:
                            votes[yy, xx, pv] += 1

        out = np.argmax(votes, axis=2).astype(np.uint8)
        empty = votes.sum(axis=2) == 0
        if np.any(empty):
            out[empty] = 0
        return out

    def reconstruct_preview_from_wave(self, wave: np.ndarray) -> np.ndarray:
        ids = np.zeros((self.patch_h, self.patch_w), dtype=np.int32)
        for y in range(self.patch_h):
            for x in range(self.patch_w):
                domain = wave[y, x]
                options = np.flatnonzero(domain)
                if options.size == 0:
                    ids[y, x] = 0
                elif options.size == 1:
                    ids[y, x] = int(options[0])
                else:
                    # Most likely according to pattern weight.
                    best = options[np.argmax(self.weights[options])]
                    ids[y, x] = int(best)
        return self.reconstruct_from_pattern_ids(ids)

    def entropy_map(self, wave: np.ndarray) -> np.ndarray:
        e = np.zeros((self.patch_h, self.patch_w), dtype=np.float64)
        for y in range(self.patch_h):
            for x in range(self.patch_w):
                domain = wave[y, x]
                if not np.any(domain):
                    e[y, x] = -1.0
                else:
                    e[y, x] = self._entropy(domain)
        return e

    def collapsed_fraction(self, wave: np.ndarray) -> float:
        counts = wave.sum(axis=2)
        return float(np.mean(counts == 1))
