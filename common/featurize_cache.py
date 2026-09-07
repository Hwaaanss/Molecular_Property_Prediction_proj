"""Disk-backed, parallel featurization cache for the dual-branch datasets.

``smiles_to_graph_dual`` embeds a 3D conformer and runs an MMFF94 optimization
per molecule. That costs roughly 10-100 ms each, so featurizing HIV (41k
molecules) single-threaded takes hours -- and the ablation sweep would repeat it
for every (condition, seed) cell. This module featurizes each dataset exactly
once, across the job's CPU budget, and stores the result under ``data/cache/``.

The cache is keyed by the SMILES content itself (a hash of the SMILES column)
plus a format version, so editing a CSV or bumping the featurizer invalidates it
automatically -- there is no stale-cache failure mode that silently trains on the
wrong molecules.

Layout: one ``.npz`` per dataset holding every molecule's arrays concatenated,
with offset indices. Loading is a single mmap-friendly read rather than 93k
small unpickles, and slicing per molecule is a view, not a copy.

Build a cache ahead of time with:
    python scripts/precompute_features.py --datasets all --workers 8
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Bump whenever the tuple written by _featurize_one changes shape or meaning.
# The version is part of the cache filename, so an older cache is simply never
# found and gets rebuilt; DualFeatureCache also refuses to load an .npz that is
# missing arrays this version requires, so a hand-copied v1 file cannot slip in.
#   dualv1 -> dualv2: added the molecule-level fingerprint array.
CACHE_VERSION = "dualv2"
_EMPTY_DIMS = {"x_chem": 19, "x_phys": 5, "edge_attr": 7}
_REQUIRED_ARRAYS = (
    "x_chem", "x_phys", "edge_index", "edge_attr", "fp",
    "node_offsets", "edge_offsets", "valid", "fp_valid",
)


def default_cache_dir() -> Path:
    from common.config import get_project_root

    return get_project_root() / "data" / "cache"


def _smiles_fingerprint(smiles_list: list[str]) -> str:
    digest = hashlib.sha256()
    for smiles in smiles_list:
        digest.update(str(smiles).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def cache_path_for(data_path: str | Path, smiles_column: str, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
    smiles_list = pd.read_csv(data_path, usecols=[smiles_column])[smiles_column].tolist()
    stem = Path(data_path).stem
    return cache_dir / f"{stem}_{CACHE_VERSION}_{_smiles_fingerprint(smiles_list)}.npz"


def _pin_worker_threads() -> None:
    """Pool initializer: one compute thread per featurization worker."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _featurize_one(smiles: str):
    """Worker body. Returns numpy arrays (never torch tensors, which would send
    a shared-memory file descriptor per result and exhaust the fd limit)."""
    from common.data import smiles_to_graph_dual

    try:
        result = smiles_to_graph_dual(smiles)
    except Exception:
        return None
    if result is None:
        return None
    x_chem, x_phys, edge_index, edge_attr, fp = result
    return (
        x_chem.numpy().astype(np.float32),
        x_phys.numpy().astype(np.float32),
        edge_index.numpy().astype(np.int64),
        edge_attr.numpy().astype(np.float32),
        fp.numpy().astype(np.float32),
    )


def resolve_workers(requested: int | None = None) -> int:
    """Featurization processes, always clamped to the job's CPU budget."""
    from common.resources import featurize_workers

    return max(1, featurize_workers(requested))


class DualFeatureCache:
    """Featurized dual-branch graphs for one CSV, sliceable by row index."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        missing = [key for key in _REQUIRED_ARRAYS if key not in arrays]
        if missing:
            raise ValueError(
                f"Feature cache is missing {', '.join(missing)}: it was written by an "
                f"older featurizer (cache format is now {CACHE_VERSION}). Rebuild it with "
                "`python scripts/precompute_features.py --datasets <name> --force`."
            )
        self.x_chem = arrays["x_chem"]
        self.x_phys = arrays["x_phys"]
        self.edge_index = arrays["edge_index"]
        self.edge_attr = arrays["edge_attr"]
        self.fp = arrays["fp"]
        self.node_offsets = arrays["node_offsets"]
        self.edge_offsets = arrays["edge_offsets"]
        self.valid = arrays["valid"].astype(bool)
        self.fp_valid = arrays["fp_valid"].astype(bool)

    def __len__(self) -> int:
        return int(self.valid.shape[0])

    @property
    def num_failed(self) -> int:
        return int((~self.valid).sum())

    @property
    def num_fingerprint_failed(self) -> int:
        """Molecules with no usable fingerprint, i.e. an all-zero row.

        Independent of ``num_failed``: fingerprints are computed from the 2D
        graph, so a conformer failure never zeroes one. In practice only an
        unparseable SMILES (or a raising generator) produces an all-zero row --
        every parseable molecule sets at least one pattern/Morgan bit.
        """
        return int((~self.fp_valid).sum())

    def get(self, idx: int):
        """Return (x_chem, x_phys, edge_index, edge_attr, fp) as numpy views, or None."""
        if not self.valid[idx]:
            return None
        node_start, node_end = int(self.node_offsets[idx]), int(self.node_offsets[idx + 1])
        edge_start, edge_end = int(self.edge_offsets[idx]), int(self.edge_offsets[idx + 1])
        return (
            self.x_chem[node_start:node_end],
            self.x_phys[node_start:node_end],
            self.edge_index[:, edge_start:edge_end],
            self.edge_attr[edge_start:edge_end],
            self.fp[idx],
        )

    def get_fingerprint(self, idx: int) -> np.ndarray:
        """Fingerprint row for ``idx``, available even when the graph is not."""
        return self.fp[idx]

    @classmethod
    def from_results(cls, results: list) -> "DualFeatureCache":
        from common.data import DEFAULT_FINGERPRINT_DIM

        count = len(results)
        node_counts = np.zeros(count, dtype=np.int64)
        edge_counts = np.zeros(count, dtype=np.int64)
        valid = np.zeros(count, dtype=bool)
        for idx, result in enumerate(results):
            if result is None:
                continue
            valid[idx] = True
            node_counts[idx] = result[0].shape[0]
            edge_counts[idx] = result[2].shape[1]

        node_offsets = np.concatenate([[0], np.cumsum(node_counts)]).astype(np.int64)
        edge_offsets = np.concatenate([[0], np.cumsum(edge_counts)]).astype(np.int64)
        total_nodes, total_edges = int(node_offsets[-1]), int(edge_offsets[-1])
        fp_dim = next(
            (int(result[4].shape[0]) for result in results if result is not None),
            DEFAULT_FINGERPRINT_DIM,
        )

        arrays = {
            "x_chem": np.zeros((total_nodes, _EMPTY_DIMS["x_chem"]), dtype=np.float32),
            "x_phys": np.zeros((total_nodes, _EMPTY_DIMS["x_phys"]), dtype=np.float32),
            "edge_index": np.zeros((2, total_edges), dtype=np.int64),
            "edge_attr": np.zeros((total_edges, _EMPTY_DIMS["edge_attr"]), dtype=np.float32),
            # One dense row per molecule (not offset-sliced): the fingerprint is
            # a fixed-width molecule-level vector, and keeping it addressable by
            # row is what lets a molecule with no usable graph still return one.
            "fp": np.zeros((count, fp_dim), dtype=np.float32),
            "node_offsets": node_offsets,
            "edge_offsets": edge_offsets,
            "valid": valid,
            "fp_valid": np.zeros(count, dtype=bool),
        }
        for idx, result in enumerate(results):
            if result is None:
                continue
            x_chem, x_phys, edge_index, edge_attr, fp = result
            arrays["x_chem"][node_offsets[idx]:node_offsets[idx + 1]] = x_chem
            arrays["x_phys"][node_offsets[idx]:node_offsets[idx + 1]] = x_phys
            arrays["edge_index"][:, edge_offsets[idx]:edge_offsets[idx + 1]] = edge_index
            arrays["edge_attr"][edge_offsets[idx]:edge_offsets[idx + 1]] = edge_attr
            arrays["fp"][idx] = fp
            arrays["fp_valid"][idx] = bool(np.any(fp))
        return cls(arrays)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Must keep the .npz suffix: np.savez appends one when it is missing,
        # which would leave the temp file under a name we never rename.
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp,
            x_chem=self.x_chem,
            x_phys=self.x_phys,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            fp=self.fp,
            node_offsets=self.node_offsets,
            edge_offsets=self.edge_offsets,
            valid=self.valid,
            fp_valid=self.fp_valid,
        )
        tmp.replace(path)  # atomic: a killed job never leaves a half-written cache

    @classmethod
    def load(cls, path: Path) -> "DualFeatureCache":
        with np.load(path) as handle:
            try:
                return cls({key: handle[key] for key in handle.files})
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from None


def build_cache(
    data_path: str | Path,
    smiles_column: str = "smiles",
    workers: int | None = None,
    force: bool = False,
    cache_dir: Path | None = None,
    verbose: bool = True,
) -> DualFeatureCache:
    """Featurize one CSV (in parallel) and cache it, or load an existing cache."""
    path = cache_path_for(data_path, smiles_column, cache_dir)
    if path.exists() and not force:
        cache = DualFeatureCache.load(path)
        if verbose:
            print(f"  [cache] hit  {path.name}  n={len(cache)}  failed={cache.num_failed}  "
                  f"fp_failed={cache.num_fingerprint_failed}")
        return cache

    smiles_list = pd.read_csv(data_path, usecols=[smiles_column])[smiles_column].tolist()
    n_workers = resolve_workers(workers)
    if verbose:
        print(f"  [cache] miss {path.name}  featurizing {len(smiles_list)} molecules "
              f"on {n_workers} worker(s)...", flush=True)

    started = time.time()
    if n_workers == 1:
        results = [_featurize_one(smiles) for smiles in smiles_list]
    else:
        from multiprocessing import Pool

        chunk = max(1, min(64, len(smiles_list) // (n_workers * 8) or 1))
        results = []
        progress_every = max(1000, len(smiles_list) // 20)
        # Workers inherit the parent's OMP_NUM_THREADS, so without this each of
        # them would open its own multi-thread BLAS pool and the pass would use
        # workers x threads cores instead of workers.
        with Pool(processes=n_workers, initializer=_pin_worker_threads) as pool:
            for done, result in enumerate(
                pool.imap(_featurize_one, smiles_list, chunksize=chunk), start=1
            ):
                results.append(result)
                if verbose and (done % progress_every == 0 or done == len(smiles_list)):
                    rate = done / max(time.time() - started, 1e-6)
                    remaining = (len(smiles_list) - done) / max(rate, 1e-6)
                    print(f"    {done}/{len(smiles_list)}  {rate:.0f} mol/s  "
                          f"eta {remaining/60:.1f} min", flush=True)

    cache = DualFeatureCache.from_results(results)
    cache.save(path)
    if verbose:
        print(f"  [cache] built {path.name} in {(time.time()-started)/60:.1f} min  "
              f"failed={cache.num_failed}/{len(cache)}  "
              f"fp_failed={cache.num_fingerprint_failed}/{len(cache)}")
    return cache
