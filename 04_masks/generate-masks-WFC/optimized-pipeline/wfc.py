"""wfc.py – Wave Function Collapse generator for overlapping-tile textures.

Supports three boundary modes:
  - "none"       flat (no wrapping)
  - "torus"      periodic in both axes
  - "cylinder_x" periodic in X only
  - "cylinder_y" periodic in Y only

Distribution-guided generation
-------------------------------
Standard WFC only enforces local pattern compatibility, which produces
distributions heavily biased toward the most-frequent patterns (background in
our dataset). Two mechanisms fix this:

1. Pattern reweighting (applied once before generation, in pipeline.py):
   Rescales base pattern weights using multiplicative importance factors so
   the EXPECTED class distribution from the pattern library matches the
   real-data target. This is the most impactful fix.

2. Per-step log-ratio guidance (Dirichlet-inspired):
   At each collapse step, multiplies candidate weights by a per-pattern
   reward derived from the running class fraction vs target:

       log_ratio[c] = log( target_frac[c] / current_frac[c] )
       reward[p]    = exp( β · Σ_c log_ratio[c] · pat_frac[p, c] )

   Using log(target/current) instead of (target-current) gives ~20× stronger
   correction when a class is severely underrepresented — the correct
   Dirichlet-posterior gradient. β=1 corresponds to exact importance
   weighting; β>1 amplifies the correction.

Speed notes
-----------
Cell selection uses domain size (number of valid patterns remaining) as a
proxy for Shannon entropy. This avoids allocating and log-computing a
[ph×pw×P] float array every step (~5 M log ops for P=512, called ~9600 times).
Shannon entropy is still computed for snapshots, just not in the hot path.

Propagation uses in-queue deduplication (prevents re-processing the same cell)
and inlined boundary checks (avoids per-iteration function call overhead).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from patterns import PatternLibrary

SnapshotCallback = Callable[[int, "np.ndarray", str], None]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WFCResult:
    success: bool
    mask: np.ndarray | None          # (H, W) uint8
    pattern_ids: np.ndarray | None   # (patch_h, patch_w) int32
    steps: int
    restarts: int
    message: str
    # Guidance tracking: list of (step, class_fracs_array) sampled periodically
    guidance_history: list[tuple[int, np.ndarray]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class WFCGenerator:
    """Overlapping-tile WFC generator with optional class-distribution guidance.

    Parameters
    ----------
    library:           pre-built PatternLibrary (may already be reweighted)
    height, width:     dimensions of the output mask
    tile_mode:         "none" | "torus" | "cylinder_x" | "cylinder_y"
    seed:              base random seed
    target_fracs:      desired class pixel fractions, shape (n_classes,).
                       None = no guidance (plain WFC).
    guidance_strength (β): log-ratio reward scale.
                       0 = plain WFC.  1 = Dirichlet importance weight.
                       3–5 = strong.  Higher values increase distribution
                       matching but may raise contradiction rate.
    log_interval:      print a progress line every this many steps (0 = silent)
    """

    VALID_MODES = {"none", "torus", "cylinder_x", "cylinder_y"}

    def __init__(
        self,
        library: PatternLibrary,
        height: int,
        width: int,
        tile_mode: str = "none",
        seed: int = 42,
        target_fracs: np.ndarray | None = None,
        guidance_strength: float = 0.0,
        log_interval: int = 500,
    ) -> None:
        tile_mode = tile_mode.strip().lower()
        tile_mode = {"periodic": "torus", "cylinder": "cylinder_x",
                     "cyl_x": "cylinder_x", "cyl_y": "cylinder_y"}.get(tile_mode, tile_mode)
        if tile_mode not in self.VALID_MODES:
            raise ValueError(f"Unknown tile_mode={tile_mode!r}")

        self.lib = library
        self.n   = library.pattern_size
        self.H, self.W = int(height), int(width)
        self.tile_mode = tile_mode
        self.periodic_x = tile_mode in {"torus", "cylinder_x"}
        self.periodic_y = tile_mode in {"torus", "cylinder_y"}
        self.seed = int(seed)
        self.log_interval = int(log_interval)

        self.ph = self.H if self.periodic_y else self.H - self.n + 1
        self.pw = self.W if self.periodic_x else self.W - self.n + 1

        self.P = int(library.patterns.shape[0])
        self.weights = library.weights.astype(np.float64)
        self.n_classes = int(library.patterns.max()) + 1

        if (not self.periodic_y) and self.H < self.n:
            raise ValueError(f"height={self.H} < pattern_size={self.n} in non-periodic Y")
        if (not self.periodic_x) and self.W < self.n:
            raise ValueError(f"width={self.W} < pattern_size={self.n} in non-periodic X")

        # ---- Guidance setup ------------------------------------------------
        self.guidance_strength = float(guidance_strength)
        self.target_fracs: np.ndarray | None = None
        self.pat_class_fracs: np.ndarray | None = None  # [P, n_classes]

        if target_fracs is not None and self.guidance_strength > 0.0:
            tf = np.array(target_fracs, dtype=np.float64)
            if tf.shape != (self.n_classes,):
                raise ValueError(
                    f"target_fracs shape {tf.shape} != (n_classes={self.n_classes},)"
                )
            self.target_fracs = tf / tf.sum()

            # [P, n_classes] – class pixel fraction for each pattern
            pats = library.patterns  # [P, n, n]
            pcf = np.zeros((self.P, self.n_classes), dtype=np.float64)
            for c in range(self.n_classes):
                pcf[:, c] = (pats == c).mean(axis=(1, 2))
            self.pat_class_fracs = pcf


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        max_restarts: int = 20,
        snapshot_interval: int = 200,
        snapshot_cb: SnapshotCallback | None = None,
    ) -> WFCResult:
        """Run WFC.  Returns a WFCResult (success=True → .mask is set)."""
        total_cells = self.ph * self.pw

        for restart in range(max_restarts):
            rng = np.random.default_rng(self.seed + restart)

            wave   = np.ones((self.ph, self.pw, self.P), dtype=bool)
            domain = np.full((self.ph, self.pw), self.P, dtype=np.int32)

            # Running class distribution estimate for guidance
            running_counts = np.zeros(self.n_classes, dtype=np.float64)
            n_collapsed    = 0

            guidance_history: list[tuple[int, np.ndarray]] = []

            t0 = time.perf_counter()
            last_log = 0

            if snapshot_cb is not None:
                snapshot_cb(0, wave, f"r{restart}_start")

            steps = 0
            while True:
                # Check completion: all cells have exactly 1 valid pattern
                if (domain == 1).all():
                    ids  = np.argmax(wave, axis=2).astype(np.int32)
                    mask = self._reconstruct(ids)
                    if snapshot_cb:
                        snapshot_cb(steps, wave, f"r{restart}_done")
                    elapsed = time.perf_counter() - t0
                    print(f"    Done: {steps} steps in {elapsed:.1f}s "
                          f"({steps/max(elapsed,1e-6):.0f} steps/s, "
                          f"{restart} restart(s))")
                    return WFCResult(
                        success=True, mask=mask, pattern_ids=ids,
                        steps=steps, restarts=restart, message="ok",
                        guidance_history=guidance_history,
                    )

                pos = self._select(domain, rng)
                if pos is None:
                    break

                y, x = pos
                chosen = self._collapse_guided(
                    wave, domain, y, x, rng,
                    running_counts, n_collapsed,
                )
                if chosen < 0:
                    print(f"    Contradiction at step {steps}, restart {restart+1}/{max_restarts}")
                    break  # contradiction → restart

                # Update running class distribution estimate
                if self.pat_class_fracs is not None:
                    running_counts += self.pat_class_fracs[chosen]
                    n_collapsed    += 1

                steps += 1

                # Propagate constraints
                queue: deque[tuple[int, int]] = deque([(y, x)])
                if not self._propagate(wave, domain, queue):
                    print(f"    Propagation contradiction at step {steps}, restart {restart+1}/{max_restarts}")
                    break

                # Record guidance snapshot
                if n_collapsed > 0 and steps % max(1, snapshot_interval) == 0:
                    if running_counts.sum() > 0:
                        guidance_history.append(
                            (steps, running_counts.copy() / running_counts.sum())
                        )
                    if snapshot_cb:
                        snapshot_cb(steps, wave, f"r{restart}_s{steps}")

                # Progress logging
                if self.log_interval > 0 and (steps - last_log) >= self.log_interval:
                    elapsed = time.perf_counter() - t0
                    frac = n_collapsed / total_cells
                    rate = steps / max(elapsed, 1e-6)
                    eta = (1.0 - frac) / frac * elapsed if frac > 0 else float("inf")
                    cf_str = ""
                    if n_collapsed > 0 and self.pat_class_fracs is not None:
                        cf = running_counts / running_counts.sum()
                        cf_str = "  fracs=[" + " ".join(
                            f"c{c}:{cf[c]:.2f}" for c in range(self.n_classes)
                        ) + "]"
                        if self.target_fracs is not None:
                            def_str = "  deficit=[" + " ".join(
                                f"c{c}:{self.target_fracs[c]-cf[c]:+.2f}"
                                for c in range(self.n_classes)
                            ) + "]"
                            cf_str += def_str
                    print(f"    step {steps:5d}  collapsed {frac:5.1%}  "
                          f"{rate:6.0f} steps/s  ETA {eta:5.0f}s  restart {restart}{cf_str}")
                    last_log = steps

        return WFCResult(
            success=False, mask=None, pattern_ids=None,
            steps=steps, restarts=max_restarts - 1,
            message=f"failed after {max_restarts} restarts",
            guidance_history=[],
        )


    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select(
        self, domain: np.ndarray, rng: np.random.Generator
    ) -> tuple[int, int] | None:
        """Select the cell with the fewest valid patterns (minimum domain size).

        Uses domain count as entropy proxy — O(ph*pw) with no log operations,
        no large array allocation. Shannon entropy (expensive) is computed only
        for snapshot visualisations, not here in the hot path.
        """
        unresolved = domain > 1
        if not unresolved.any():
            return None

        # Cells with smaller domain have higher 'priority' (collapse them first)
        scores = np.where(unresolved, domain.astype(np.float64), np.inf)
        scores[domain <= 0] = np.inf
        scores += rng.random(scores.shape) * 1e-6  # random tiebreaking
        idx = int(np.argmin(scores))
        return divmod(idx, self.pw)


    def _collapse_guided(
        self,
        wave: np.ndarray,
        domain: np.ndarray,
        y: int, x: int,
        rng: np.random.Generator,
        running_counts: np.ndarray,
        n_collapsed: int,
    ) -> int:
        """Collapse cell (y,x), applying log-ratio guidance if configured.

        Returns the chosen pattern id, or -1 on contradiction.

        Guidance uses log(target/current) as the reward signal — the correct
        Dirichlet-posterior gradient. When class c is at fraction 0.01 vs
        target 0.28, the log-ratio signal is log(28) ≈ 3.3, which is ~12×
        stronger than the linear deficit 0.27. This prevents background from
        overwhelming rare classes.
        """
        options = np.flatnonzero(wave[y, x])
        if options.size == 0:
            return -1
        if options.size == 1:
            domain[y, x] = 1
            return int(options[0])

        w = self.weights[options].copy()

        # ---- Log-ratio guidance (Dirichlet-inspired) --------------------------
        if (self.pat_class_fracs is not None
                and self.target_fracs is not None
                and n_collapsed > 0):
            total = running_counts.sum()
            if total > 0:
                current_fracs = running_counts / total
                # log(target/current): large when class is under-represented,
                # negative when over-represented, zero when on target.
                # clip to avoid log(0) and extreme values.
                eps = 1e-6
                log_ratio = np.log(
                    np.clip(
                        self.target_fracs / np.clip(current_fracs, eps, None),
                        eps, 1e6
                    )
                )
                # reward[i] = exp(β · dot(log_ratio, pat_frac[options[i]]))
                reward_scores = self.pat_class_fracs[options] @ log_ratio  # [k]
                # clip to avoid exp overflow/underflow
                reward = np.exp(
                    np.clip(self.guidance_strength * reward_scores, -30.0, 30.0)
                )
                w = w * reward
        # -----------------------------------------------------------------------

        s = w.sum()
        probs = w / s if s > 0 else np.ones(options.size) / options.size

        # Guard against NaN from extreme β values
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            probs = np.ones(options.size) / options.size

        chosen = int(rng.choice(options, p=probs))
        wave[y, x, :] = False
        wave[y, x, chosen] = True
        domain[y, x] = 1
        return chosen


    def _propagate(
        self, wave: np.ndarray, domain: np.ndarray,
        queue: deque[tuple[int, int]]
    ) -> bool:
        """Arc-consistency propagation (AC-3 variant).

        Optimisations vs naive version:
        - in_queue flag prevents re-adding cells already in the queue,
          bounding total work to O(cells * P) per collapse step.
        - Boundary checks inlined (avoids per-iteration _neighbor() call).
        - Local variable caching for frequently accessed attributes.
        """
        ph = self.ph
        pw = self.pw
        periodic_x = self.periodic_x
        periodic_y = self.periodic_y
        compat = self.lib.compat
        # (dy, dx, direction_key)
        DIRS = ((-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right"))

        # Track in-queue state to avoid redundant re-processing
        in_queue = np.zeros((ph, pw), dtype=bool)
        for iy, ix in queue:
            in_queue[iy, ix] = True

        while queue:
            y, x = queue.popleft()
            in_queue[y, x] = False

            if domain[y, x] <= 0:
                return False

            for dy, dx, direction in DIRS:
                # Compute neighbour coordinates with inlined boundary logic
                ny = y + dy
                nx = x + dx
                if ny < 0:
                    if periodic_y:
                        ny = ph - 1
                    else:
                        continue
                elif ny >= ph:
                    if periodic_y:
                        ny = 0
                    else:
                        continue
                if nx < 0:
                    if periodic_x:
                        nx = pw - 1
                    else:
                        continue
                elif nx >= pw:
                    if periodic_x:
                        nx = 0
                    else:
                        continue

                if domain[ny, nx] <= 0:
                    return False

                # Allowed patterns: union of compat entries for all current patterns
                if domain[y, x] == 1:
                    pid = int(np.argmax(wave[y, x]))
                    allowed = compat[direction][pid]  # [P] bool, fast index
                else:
                    # Fancy-index: compat[d][bool_mask] → [k, P], then any
                    allowed = np.any(compat[direction][wave[y, x]], axis=0)  # [P] bool

                new_wave = wave[ny, nx] & allowed
                new_size = int(new_wave.sum())

                if new_size <= 0:
                    return False          # contradiction
                if new_size == domain[ny, nx]:
                    continue              # no change, neighbour not affected

                wave[ny, nx] = new_wave
                domain[ny, nx] = new_size

                # Only enqueue if not already waiting
                if not in_queue[ny, nx]:
                    queue.append((ny, nx))
                    in_queue[ny, nx] = True

        return True


    # ------------------------------------------------------------------
    # Reconstruction (vectorised)
    # ------------------------------------------------------------------

    def _reconstruct(self, ids: np.ndarray) -> np.ndarray:
        """Reconstruct (H, W) mask from pattern-id grid via majority vote."""
        votes = np.zeros((self.H, self.W, self.n_classes), dtype=np.int32)
        patterns = self.lib.patterns

        row_idx = (np.arange(self.ph, dtype=np.int32)[:, None]
                   * np.ones(self.pw, dtype=np.int32)[None, :])
        col_idx = (np.arange(self.pw, dtype=np.int32)[None, :]
                   * np.ones(self.ph, dtype=np.int32)[:, None])

        for dy in range(self.n):
            for dx in range(self.n):
                pv = patterns[ids, dy, dx]
                yy = row_idx + dy
                xx = col_idx + dx

                if self.periodic_y:
                    yy = yy % self.H
                    ym = np.ones((self.ph, self.pw), dtype=bool)
                else:
                    ym = yy < self.H

                if self.periodic_x:
                    xx = xx % self.W
                    xm = np.ones((self.ph, self.pw), dtype=bool)
                else:
                    xm = xx < self.W

                valid = ym & xm
                np.add.at(votes, (yy[valid], xx[valid], pv[valid]), 1)

        out = np.argmax(votes, axis=2).astype(np.uint8)
        out[votes.sum(axis=2) == 0] = 0
        return out


    def preview_from_wave(self, wave: np.ndarray) -> np.ndarray:
        """Quick reconstruction using most-likely pattern per cell."""
        weighted = wave * self.weights[None, None, :]
        ids = np.argmax(weighted, axis=2).astype(np.int32)
        return self._reconstruct(ids)


    # ------------------------------------------------------------------
    # Shannon entropy map (for visualisation only — not used in hot path)
    # ------------------------------------------------------------------

    def entropy_map(self, wave: np.ndarray) -> np.ndarray:
        """Return (ph, pw) float64 Shannon entropy map.  -1 = contradiction.

        NOTE: this is expensive (O(ph*pw*P) with log). Only call for
        snapshot visualisations, not in the collapse-selection hot path.
        """
        w = self.weights[None, None, :] * wave
        w_sum = w.sum(axis=2)
        log_w = np.where(w > 0, np.log(np.where(w > 0, w, 1.0)), 0.0)
        w_log_w_sum = (w * log_w).sum(axis=2)
        has_options = w_sum > 0
        ent = np.where(
            has_options,
            np.log(np.where(has_options, w_sum, 1.0))
            - w_log_w_sum / np.where(has_options, w_sum, 1.0),
            0.0,
        )
        ent[~wave.any(axis=2)] = -1.0
        return ent


    def collapsed_fraction(self, wave: np.ndarray) -> float:
        return float((wave.sum(axis=2) == 1).mean())
