"""Featurize every dataset once, in parallel, into data/cache/.

The dual-branch featurizer embeds a 3D conformer and runs MMFF94 per molecule.
Doing that inside each training run would repeat hours of CPU work for every
(condition, seed) cell -- 80 times over for a full sweep. This script does it
once per dataset across all cores; training runs then hit the cache.

Run this before the sweep:
    python scripts/precompute_features.py --datasets all --workers 8

It is idempotent (an existing, content-matching cache is left alone) and safe to
interrupt: caches are written atomically, one dataset at a time.

Sizing note: HIV (41k) and CEP (30k) dominate. Expect the cache to be a few
hundred MB per large dataset under data/cache/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Must run before numpy/rdkit load (see common/resources).
from common.resources import apply_thread_limits  # noqa: E402

apply_thread_limits(verbose=True)

from common.datasets import (  # noqa: E402
    DATASETS,
    TASK_TYPES,
    datasets_for_task,
    get_dataset_spec,
)
from common.featurize_cache import build_cache, cache_path_for, resolve_workers  # noqa: E402


def expand(names: list[str]) -> list[str]:
    resolved: list[str] = []
    for entry in names:
        key = entry.lower()
        if key == "all":
            resolved.extend(DATASETS.keys())
        elif key in TASK_TYPES:
            resolved.extend(datasets_for_task(key))
        else:
            resolved.append(key)
    return list(dict.fromkeys(resolved))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        help="Dataset names, a task type (classification/regression), or all.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel featurization workers. Default: all CPU cores.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a cache exists.")
    args = parser.parse_args()

    names = expand(args.datasets)
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        parser.error(f"Unknown dataset(s): {', '.join(unknown)}")

    workers = resolve_workers(args.workers)
    print(f"Featurizing {len(names)} dataset(s) on {workers} worker(s): {', '.join(names)}\n")

    started = time.time()
    total_failed = 0
    total_fp_failed = 0
    for name in names:
        spec = get_dataset_spec(name)
        data_path = spec.data_path()
        if not data_path.exists():
            print(f"[{name}] SKIP — {data_path} not found; run scripts/download_data.py first")
            continue
        print(f"[{name}] {data_path.name}  ({spec.task_type})")
        cache = build_cache(
            data_path,
            smiles_column=spec.smiles_column,
            workers=workers,
            force=args.force,
        )
        total_failed += cache.num_failed
        total_fp_failed += cache.num_fingerprint_failed
        path = cache_path_for(data_path, spec.smiles_column)
        size_mb = path.stat().st_size / 1024**2 if path.exists() else 0.0
        print(f"  -> {path.name}  {size_mb:.0f} MB  "
              f"conformer failures: {cache.num_failed}/{len(cache)}  "
              f"fingerprint failures: {cache.num_fingerprint_failed}/{len(cache)}\n")

    print(f"Done in {(time.time()-started)/60:.1f} min. "
          f"Molecules without a usable conformer (zero-filled at train time): {total_failed}")
    # Reported separately because the two are independent: fingerprints come
    # from the 2D graph, so a conformer failure still yields a real fingerprint.
    print(f"Molecules without a usable fingerprint (unparseable SMILES): {total_fp_failed}")


if __name__ == "__main__":
    main()
