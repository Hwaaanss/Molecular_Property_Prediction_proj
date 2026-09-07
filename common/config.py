from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


DEFAULT_SEED = 42
DEFAULT_DATASET_NAME = "tox21"
DEFAULT_TARGET_COLUMNS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
NUM_CLASSES = len(DEFAULT_TARGET_COLUMNS)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def project_path(value: str | Path) -> Path:
    """Resolve a CLI path argument against the project root, not the cwd.

    Every path this project prints, documents or defaults to is written
    relative to the repository root (``ablation/runs_v5``,
    ``dual_kd_gnn/optuna/...``). Argparse would resolve those against whatever
    directory the command happened to be launched from, so the same documented
    command writes results to a different place depending on where you stand.
    Anchoring relative values to the project root makes the commands in
    commands.md mean one thing. Absolute paths are returned untouched.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (get_project_root() / path)


def ablation_runs_root(split_type: str | None = None, legacy: bool = False) -> Path:
    """Directory holding ablation runs for one split protocol.

    Layout is ``ablation/runs/<split_type>/<condition>/<dataset>_seed<N>/``, one
    subtree per protocol so runs under different splits never overwrite each
    other. ``legacy=True`` (or ``split_type=None``) returns the flat pre-refactor
    path ``ablation/runs/``, whose ``<condition>/`` folders hold the label-aware
    results and are read-only from here on.
    """
    root = get_project_root() / "ablation" / "runs"
    return root if (legacy or split_type is None) else root / split_type


def configure_performance(verbose: bool = True) -> None:
    """Enable the throughput knobs that are safe for this workload.

    TF32 matmuls and convolutions are a large free speedup on Ampere (A100) and
    do not change any reported metric beyond float noise -- the model already
    trains under autocast AMP. cuDNN benchmark mode pays off because batch shapes
    repeat across epochs.

    TF32 shifts results by roughly 1e-3 relative -- well inside seed-to-seed
    noise, but it does mean a TF32 run is not bit-comparable with a non-TF32 one.
    Set ``DIKAT_TF32=0`` to keep strict FP32 matmuls when that matters.

    Idempotent, so scripts may call it unconditionally at startup.
    """
    if not torch.cuda.is_available():
        return
    import os

    use_tf32 = os.environ.get("DIKAT_TF32", "1") != "0"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    if verbose:
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[perf] {name} ({total_gb:.0f} GB) | "
              f"TF32 {'on' if use_tf32 else 'off'} | cudnn.benchmark on")


def is_mps_available() -> bool:
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()


def get_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{requested_device}' but CUDA is not available. "
                "Refusing to silently fall back to CPU. "
                "Pass --device cpu explicitly to run on CPU."
            )
        if requested_device == "mps" and not is_mps_available():
            raise RuntimeError(
                "Requested device 'mps' but MPS is not available. "
                "Pass --device cpu explicitly to run on CPU."
            )
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = DEFAULT_SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    slug: str
    uses_dual_features: bool
    builder: Callable[..., torch.nn.Module]
    default_hparams: dict[str, Any] = field(default_factory=dict)
    add_model_arguments: Callable[[Any], None] | None = None
    collect_model_kwargs: Callable[[Any], dict[str, Any]] | None = None
    collect_hparam_overrides: Callable[[Any], dict[str, Any]] | None = None
    notes: str = ""
