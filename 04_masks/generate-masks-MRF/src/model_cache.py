"""
Persistent cache for the pieces of the MRF model that depend only on the
training dataset and the neighborhood radius: class frequencies, pairwise
potentials, and the nonparametric causal/full neighborhood lookup tables
(each with a KDTree fallback over unique observed neighborhoods).

Building these from scratch means looping in Python over every pixel of
every augmented training mask (millions of iterations) and fitting KDTrees
for nearest-neighbor fallback — tens of seconds to minutes, and identical
for any run against the same dataset/radius. `load_or_build_model` builds
this once and caches it to disk via joblib.
"""

import hashlib
import os
import time

import joblib
import numpy as np

from .dataset import BarkMaskDataset, _causal_offsets, _full_neighborhood_offsets

DEFAULT_CACHE_DIR = ".mrf_cache"

# Bump this if TrainedModel's fields or the table format ever change, so
# stale caches from an older code version are rebuilt instead of misread.
CACHE_VERSION = 2


class TrainedModel:
    """Everything derivable from (training dataset, radius) alone.

    Independent of canvas size, seed, target ratio, temperature, or
    tileability — those are applied at synthesis/inpainting time.
    """

    def __init__(self, radius, class_freqs, pairwise,
                 causal_di, causal_dj, causal_table, causal_kdtree, causal_kd_counts,
                 full_di, full_dj, full_table, full_kdtree, full_kd_counts):
        self.version = CACHE_VERSION
        self.radius = radius
        self.class_freqs = class_freqs
        self.pairwise = pairwise
        self.causal_di = causal_di
        self.causal_dj = causal_dj
        self.causal_table = causal_table
        self.causal_kdtree = causal_kdtree
        self.causal_kd_counts = causal_kd_counts
        self.full_di = full_di
        self.full_dj = full_dj
        self.full_table = full_table
        self.full_kdtree = full_kdtree
        self.full_kd_counts = full_kd_counts

    @classmethod
    def build(cls, dataset, radius, verbose=True):
        # Local import: synthesis.py doesn't need to import this module, so
        # only this direction of the dependency exists.
        from .synthesis import _build_table

        class_freqs = dataset.estimate_class_frequencies()
        pairwise = dataset.estimate_pairwise_potentials()

        causal_offsets = _causal_offsets(radius)
        causal_di = np.array([o[0] for o in causal_offsets], dtype=np.int32)
        causal_dj = np.array([o[1] for o in causal_offsets], dtype=np.int32)
        if verbose:
            print("Building causal neighborhood lookup table...")
        causal_table, causal_kdtree, causal_kd_counts = _build_table(
            dataset, causal_di, causal_dj
        )

        full_offsets = _full_neighborhood_offsets(radius)
        full_di = np.array([o[0] for o in full_offsets], dtype=np.int32)
        full_dj = np.array([o[1] for o in full_offsets], dtype=np.int32)
        if verbose:
            print("Building full-neighborhood table for Gibbs refinement...")
        full_table, full_kdtree, full_kd_counts = _build_table(
            dataset, full_di, full_dj
        )

        return cls(
            radius, class_freqs, pairwise,
            causal_di, causal_dj, causal_table, causal_kdtree, causal_kd_counts,
            full_di, full_dj, full_table, full_kdtree, full_kd_counts,
        )

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Write to a temp file and rename, so a crash mid-write (or a
        # concurrent reader) never sees a truncated cache file.
        tmp_path = f"{path}.tmp{os.getpid()}"
        joblib.dump(self, tmp_path, compress=3)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError(f"{path} does not contain a TrainedModel")
        if getattr(model, "version", None) != CACHE_VERSION:
            raise ValueError(
                f"{path} was built by an incompatible cache version "
                f"({getattr(model, 'version', None)} != {CACHE_VERSION})"
            )
        return model


def dataset_fingerprint(folder):
    """
    Cheap content fingerprint (filenames + sizes + mtimes) so the cache can
    be invalidated automatically when the dataset changes, without reading
    and decoding every training image just to check.
    """
    files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    h = hashlib.sha1()
    for f in files:
        st = os.stat(os.path.join(folder, f))
        h.update(f.encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()[:16]


def cache_path(cache_dir, data_folder, radius):
    """Deterministic cache file path for a given (data_folder, radius)."""
    fingerprint = dataset_fingerprint(data_folder)
    base = os.path.basename(os.path.normpath(data_folder))
    filename = f"{base}_r{radius}_v{CACHE_VERSION}_{fingerprint}.joblib"
    return os.path.join(cache_dir, filename)


def load_or_build_model(data_folder, radius, cache_dir=DEFAULT_CACHE_DIR,
                         use_cache=True, refresh=False, verbose=True):
    """
    Return (trained_model, dataset) for (data_folder, radius), transparently
    caching to disk keyed on the dataset's contents and the radius.

    On a cache hit, the training masks are never loaded at all — `dataset`
    is returned as None in that case, and the caller should lazily construct
    a `BarkMaskDataset` only if it separately needs one (e.g. for --evaluate).
    """
    path = None
    if use_cache:
        path = cache_path(cache_dir, data_folder, radius)
        if not refresh and os.path.exists(path):
            try:
                t0 = time.time()
                model = TrainedModel.load(path)
                if verbose:
                    print(f"Loaded cached MRF model from {path} "
                          f"({time.time() - t0:.3f}s)")
                return model, None
            except Exception as e:
                if verbose:
                    print(f"Cache at {path} unusable ({e}); rebuilding...")

    dataset = BarkMaskDataset(data_folder)
    model = TrainedModel.build(dataset, radius, verbose=verbose)

    if use_cache:
        t0 = time.time()
        model.save(path)
        if verbose:
            print(f"Saved MRF model cache to {path} ({time.time() - t0:.1f}s)")

    return model, dataset
