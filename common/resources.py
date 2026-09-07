"""Hard CPU/RAM budget for the whole pipeline.

The host reports 128 cores, but the job is only entitled to a small slice of
them; exceeding the allocation gets the process killed by the cluster manager.
Left alone, PyTorch sizes its intra-op thread pool from ``os.cpu_count()`` (64
threads here) and every DataLoader worker inherits an OpenMP pool of the same
size, so a single training run would blow past an 8-core budget instantly.

This module pins every thread pool to the declared budget. Import it -- and call
:func:`apply_thread_limits` -- **before importing torch or numpy**, because the
OpenMP/MKL runtimes read their environment variables once, at load time. Nothing
here imports torch, so it is safe as the very first import in a script.

Budget knobs (env):
    DIKAT_CPU_BUDGET      total cores the job may use   (default 8)
    DIKAT_LOADER_WORKERS  DataLoader worker processes   (default budget-1, max 4)
    DIKAT_FEATURIZE_WORKERS  parallel featurization procs (default budget)
"""
from __future__ import annotations

import os

DEFAULT_CPU_BUDGET = 8

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def cpu_budget() -> int:
    """Total cores this job may occupy."""
    raw = os.environ.get("DIKAT_CPU_BUDGET")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_CPU_BUDGET


def loader_workers() -> int:
    """DataLoader worker processes.

    One core is reserved for the main process, which does the GPU feeding; more
    than four workers stops helping once features come from the cache, since the
    workers then only collate.
    """
    raw = os.environ.get("DIKAT_LOADER_WORKERS")
    if raw is not None and raw.isdigit():
        return min(int(raw), max(0, cpu_budget() - 1))
    return max(0, min(4, cpu_budget() - 1))


def featurize_workers(requested: int | None = None) -> int:
    """Processes for the one-off featurization pass (no GPU work in flight).

    Capped one below the budget: the parent process also collects results, so
    running exactly ``budget`` workers would push the job over its allocation.
    """
    ceiling = max(1, cpu_budget() - 1)
    if requested is not None and requested > 0:
        return min(requested, ceiling)
    raw = os.environ.get("DIKAT_FEATURIZE_WORKERS")
    if raw and raw.isdigit() and int(raw) > 0:
        return min(int(raw), ceiling)
    return ceiling


def apply_thread_limits(verbose: bool = False) -> int:
    """Pin every native thread pool to the budget. Safe to call repeatedly.

    Returns the per-process thread count. Worker processes get 1 thread each:
    the budget is shared with the loader workers, and BLAS inside a collate
    worker gains nothing from extra threads.
    """
    budget = cpu_budget()
    workers = loader_workers()
    # Main process keeps what the workers do not take, and at least one thread.
    main_threads = max(1, budget - workers)

    for name in _THREAD_ENV_VARS:
        os.environ.setdefault(name, str(main_threads))

    try:
        import torch

        torch.set_num_threads(main_threads)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        # set_num_interop_threads raises if the pool is already initialised;
        # the env vars above still bound the damage.
        pass

    if verbose:
        print(f"[cpu] budget={budget} cores | main threads={main_threads} | "
              f"loader workers={workers}")
    return main_threads


def worker_init(_worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: one compute thread per worker."""
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
