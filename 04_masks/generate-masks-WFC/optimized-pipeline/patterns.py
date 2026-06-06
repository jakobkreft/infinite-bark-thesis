"""patterns.py – WFC overlapping-tile pattern extraction.

Key optimisations:
- Compatibility matrix built with vectorised NumPy broadcasting (no Python loops).
- Pattern deduplication uses tobytes() hashing (fast).
- PatternLibrary.reweight_for_target(): multiplicative importance reweighting
  so the weighted-mean class distribution matches a target (real-data) distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")


@dataclass
class PatternLibrary:
    """Immutable pattern library produced by extract_patterns()."""
    pattern_size: int
    patterns: np.ndarray           # [P, n, n]  uint8
    weights:  np.ndarray           # [P]        float64 – frequency (may be reweighted)
    compat:   dict[str, np.ndarray] # dir -> [P, P] bool


    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "pattern_library.npz",
            pattern_size=np.array([self.pattern_size], dtype=np.int32),
            patterns=self.patterns,
            weights=self.weights.astype(np.float64),
            compat_up=self.compat["up"],
            compat_down=self.compat["down"],
            compat_left=self.compat["left"],
            compat_right=self.compat["right"],
        )
        summary = {
            "pattern_size": self.pattern_size,
            "n_patterns": int(self.patterns.shape[0]),
            "weight_min": float(self.weights.min()),
            "weight_max": float(self.weights.max()),
            "compat_density": {d: float(self.compat[d].mean()) for d in DIRECTIONS},
        }
        with (out_dir / "pattern_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)


    @staticmethod
    def load(npz_path: Path) -> "PatternLibrary":
        data = np.load(npz_path, allow_pickle=False)
        return PatternLibrary(
            pattern_size=int(data["pattern_size"][0]),
            patterns=data["patterns"].astype(np.uint8),
            weights=data["weights"].astype(np.float64),
            compat={
                "up":    data["compat_up"].astype(bool),
                "down":  data["compat_down"].astype(bool),
                "left":  data["compat_left"].astype(bool),
                "right": data["compat_right"].astype(bool),
            },
        )


    # ------------------------------------------------------------------
    # Class composition helpers
    # ------------------------------------------------------------------

    def compute_class_fracs(self, n_classes: int = 3) -> np.ndarray:
        """Return [P, n_classes] array of class pixel fractions per pattern.

        pcf[p, c] = fraction of pixels in pattern p that belong to class c.
        """
        P = self.patterns.shape[0]
        pcf = np.zeros((P, n_classes), dtype=np.float64)
        for c in range(n_classes):
            pcf[:, c] = (self.patterns == c).mean(axis=(1, 2))
        return pcf

    def corpus_class_fracs(self, n_classes: int = 3) -> np.ndarray:
        """Weighted-mean class fractions across all patterns (the 'corpus distribution').

        This is the expected class distribution WFC would produce if it drew
        patterns independently according to the base weights.
        """
        pcf = self.compute_class_fracs(n_classes)
        w_norm = self.weights.astype(np.float64)
        w_norm = w_norm / w_norm.sum()
        return (w_norm[:, None] * pcf).sum(axis=0)  # [n_classes]


    # ------------------------------------------------------------------
    # Reweighting
    # ------------------------------------------------------------------

    def reweight_for_target(
        self,
        target_fracs: np.ndarray,  # [n_classes] desired class pixel fractions
        clip_min: float = 0.05,    # min scale factor relative to median weight
        clip_max: float = 50.0,    # max scale factor relative to median weight
    ) -> "PatternLibrary":
        """Return a new PatternLibrary with multiplicatively reweighted weights.

        Method: importance-ratio reweighting (one step of iterative proportional
        fitting in log space).

            log(new_w[p]) = log(w[p]) + Σ_c  pat_frac[p,c] · log(target[c] / corpus[c])

        Equivalently:
            new_w[p] = w[p] · ∏_c  (target[c] / corpus[c])^pat_frac[p,c]

        This ensures that the weighted-mean class distribution of the reweighted
        library equals target_fracs (to first-order approximation). Patterns
        containing underrepresented classes are upweighted; all-background
        patterns are downweighted.

        Parameters
        ----------
        target_fracs : desired class fractions (will be L1-normalised)
        clip_min     : minimum weight = clip_min × median(original_weights)
        clip_max     : maximum weight = clip_max × median(original_weights)
        """
        tf = np.asarray(target_fracs, dtype=np.float64)
        tf = tf / tf.sum()
        n_cls = len(tf)

        pcf = self.compute_class_fracs(n_classes=n_cls)  # [P, C]

        w = self.weights.astype(np.float64)
        w_norm = w / w.sum()
        corpus = (w_norm[:, None] * pcf).sum(axis=0)  # [C]

        eps = 1e-9
        log_ratio = np.log(np.clip(tf / np.clip(corpus, eps, None), eps, None))  # [C]
        log_scale = pcf @ log_ratio   # [P] — log importance weight per pattern
        scale = np.exp(log_scale)     # [P]

        new_w = w * scale

        # Clip relative to median to prevent extreme weights destabilising WFC
        med = float(np.median(new_w))
        if med > 0:
            new_w = np.clip(new_w, clip_min * med, clip_max * med)

        # Report shift in corpus distribution
        new_w_norm = new_w / new_w.sum()
        new_corpus = (new_w_norm[:, None] * pcf).sum(axis=0)
        print(f"    Pattern reweighting:"
              f"\n      corpus before: {_fmt_fracs(corpus)}"
              f"\n      corpus after:  {_fmt_fracs(new_corpus)}"
              f"\n      target:        {_fmt_fracs(tf)}")

        return PatternLibrary(
            pattern_size=self.pattern_size,
            patterns=self.patterns,
            weights=new_w,
            compat=self.compat,
        )


def _fmt_fracs(fracs: np.ndarray) -> str:
    return " ".join(f"c{i}:{fracs[i]:.3f}" for i in range(len(fracs)))


# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------

def extract_patterns(
    masks: list[np.ndarray],
    pattern_size: int = 3,
    augment_symmetry: bool = True,
    min_weight: int = 2,
    max_patterns: int = 512,
) -> PatternLibrary:
    """Extract all n×n overlapping tile patterns from *masks*.

    Parameters
    ----------
    masks:            list of 2-D uint8 masks
    pattern_size:     side length *n* of each tile pattern
    augment_symmetry: include all 8 dihedral transforms (4 rots + 4 flips)
    min_weight:       discard patterns that appear fewer times than this
    max_patterns:     keep only the *max_patterns* most frequent patterns
                      (0 = keep all). For pattern_size > 3, consider using
                      1024–2048 to retain more structural diversity.
    """
    if pattern_size < 2:
        raise ValueError("pattern_size must be >= 2")

    pattern_to_id: dict[bytes, int] = {}
    patterns: list[np.ndarray] = []
    weights: list[int] = []

    n_positions_total = 0
    for mask in masks:
        variants = _augmented(mask) if augment_symmetry else [mask]
        for v in variants:
            h, w = v.shape
            if h < pattern_size or w < pattern_size:
                continue
            for y in range(h - pattern_size + 1):
                for x in range(w - pattern_size + 1):
                    p = v[y : y + pattern_size, x : x + pattern_size]
                    key = p.tobytes()
                    pid = pattern_to_id.get(key)
                    if pid is None:
                        pattern_to_id[key] = len(patterns)
                        patterns.append(p.copy())
                        weights.append(1)
                    else:
                        weights[pid] += 1
                    n_positions_total += 1

    if not patterns:
        raise RuntimeError("No patterns extracted – check dataset dimensions vs pattern_size.")

    pat_arr = np.stack(patterns, axis=0).astype(np.uint8)
    w_arr   = np.array(weights, dtype=np.float64)

    print(f"    Extracted {len(patterns)} unique patterns from {n_positions_total} positions")

    # Filter by minimum frequency
    keep = w_arr >= max(1, min_weight)
    if not keep.any():
        raise RuntimeError(
            f"No patterns survive min_weight={min_weight}.  "
            "Try lowering --min-weight."
        )
    pat_arr, w_arr = pat_arr[keep], w_arr[keep]
    print(f"    After min_weight={min_weight} filter: {pat_arr.shape[0]} patterns")

    # Keep top-K by frequency
    if max_patterns > 0 and pat_arr.shape[0] > max_patterns:
        top_k = np.argsort(w_arr)[::-1][:max_patterns]
        pat_arr, w_arr = pat_arr[top_k], w_arr[top_k]
        print(f"    After top-{max_patterns} filter: {pat_arr.shape[0]} patterns")

    compat = _compute_compat(pat_arr)
    return PatternLibrary(pattern_size=pattern_size, patterns=pat_arr,
                          weights=w_arr, compat=compat)


def _augmented(mask: np.ndarray) -> list[np.ndarray]:
    """Return up to 8 dihedral transforms of *mask*, deduplicated."""
    seen: dict[bytes, np.ndarray] = {}
    for k in range(4):
        r = np.rot90(mask, k)
        seen.setdefault(r.tobytes(), r)
        f = np.fliplr(r)
        seen.setdefault(f.tobytes(), f)
    return list(seen.values())


def _compute_compat(patterns: np.ndarray) -> dict[str, np.ndarray]:
    """Build 4-directional boolean compatibility matrices.

    compat["right"][a, b] = True  iff  pattern b can be placed
    immediately to the right of pattern a  (their overlap is consistent).

    Vectorised: compare slices of all P patterns simultaneously.
    """
    P, n, _ = patterns.shape

    right_a = patterns[:, :, 1:]   # [P, n, n-1]
    right_b = patterns[:, :, :-1]  # [P, n, n-1]
    down_a  = patterns[:, 1:, :]   # [P, n-1, n]
    down_b  = patterns[:, :-1, :]  # [P, n-1, n]

    compat_right = _all_match(right_a, right_b)
    compat_down  = _all_match(down_a,  down_b)
    compat_left  = compat_right.T
    compat_up    = compat_down.T

    return {
        "right": compat_right,
        "left":  compat_left,
        "down":  compat_down,
        "up":    compat_up,
    }


def _all_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return [P_a, P_b] bool matrix: all(a[i] == b[j]) for every (i, j) pair."""
    a_exp = a[:, None, ...]   # [P, 1, ...]
    b_exp = b[None, :, ...]   # [1, P, ...]
    match = np.all(a_exp == b_exp, axis=tuple(range(2, a_exp.ndim)))  # [P, P]
    return match.astype(bool)
