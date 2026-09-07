"""Multi-seed sweep over the (encoder, ablation, dataset, seed) grid.

Ablation grid
  Four factors can each be removed: the 3D (physical) branch, the cross-modal
  InfoNCE, the codebook head, and the fingerprint branch. The intact model plus
  all 15 non-empty removal subsets give 16 conditions, and each is instantiated
  once per message-passing encoder (gcn, gine, gat) -- 48 conditions in total,
  named ``<conv>_full_model`` and ``<conv>_no_<factors>`` (factors always listed
  in the order 3d, infonce, codebook, fp).

  The intact model has the fingerprint branch ON and the head's projection
  block-diagonal, so the fingerprint really is a third block of the quadratic
  form rather than something a full projection has already mixed away.

Output layout
  ``<runs-root>/<split_type>/<condition>/<dataset>_seed<N>/metrics.json``

  ``--runs-root`` defaults to ``ablation/runs_v2``. The pre-refactor results
  under ``ablation/runs`` came from a different architecture (a transformer
  stage that no longer exists) and must not be pooled with these, so they get
  their own tree and their own summary CSV.

Idempotency: existing metrics.json files are detected and skipped, so this
script is safe to restart after interruption.

Usage on server
  conda activate dualgnn
  nohup python -u scripts/seed_expansion.py --split-type scaffold \
      --datasets bbbp bace sider --conditions fast --seeds 0 1 2 3 42 \
      --skip-random --device cuda > logs/fast.log 2>&1 &
  tail -f logs/fast.log

After completion
  python ablation/main.py --runs-dir ablation/runs_v2
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Must run before torch/numpy load: their OpenMP pools read the environment once,
# at import time, and would otherwise size themselves from the host's 128 cores.
from common.resources import apply_thread_limits  # noqa: E402

apply_thread_limits(verbose=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common.config import (  # noqa: E402
    configure_performance,
    set_seed,
    get_device,
    project_path,
)
from common.data import (  # noqa: E402
    SPLIT_TYPE_CHOICES,
    SPLIT_TYPES,
    resolve_split_type,
    split_label,
    count_auc_defined_tasks,
    create_dual_datasets,
    split_indices,
    target_statistics,
)
from common.datasets import (  # noqa: E402
    DATASETS,
    TASK_TYPES,
    datasets_for_task,
    resolve_target_columns,
)
from common.featurize_cache import build_cache  # noqa: E402
from common.io_utils import save_history_rows, save_json  # noqa: E402
from common.plotting import plot_training_curves  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from dual_kd_gnn.configs import (  # noqa: E402
    load_run_config,
    sanitize_hparams,
    sanitize_model_kwargs,
)
from dual_kd_gnn.model import GNN_CONV_TYPES, DualDistillationModel as Model  # noqa: E402

# Configuration
NEW_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
DATASETS_ALL = list(DATASETS.keys())
# The five originally-benchmarked classification sets, kept as a named group so
# `--datasets original` still reproduces the pre-expansion sweep exactly.
DATASETS_ORIGINAL = ["bace", "bbbp", "clintox", "sider", "tox21"]
DATASETS_RANDOM = ["sider", "tox21", "clintox"]
SCAFFOLD_SPLIT_TYPES = ("deterministic_scaffold", "random_scaffold", "label_aware_scaffold")
# Results of this (post-refactor) architecture live in their own tree; see the
# module docstring. Overridable with --runs-root.
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "ablation" / "runs_v2"
RUNS_ROOT = DEFAULT_RUNS_ROOT
# Optional per-(dataset, condition) config lookup, set by --config-template.
# Used to replay the winner of a per-cell Optuna study: the shared
# dual_kd_gnn/optuna/<dataset>_xkd/best_config.json is one config per dataset,
# but the tuning runs here are one study per (dataset, encoder, fingerprint)
# cell, so each cell needs its own file.
CONFIG_TEMPLATE: str | None = None
RANDOM_RUNS_DIR_NAME = "full_model_random"

# ---------------------------------------------------------------------------
# Ablation grid.
#
# Four removable factors. Each entry is applied on top of the intact model's
# (model_kwargs, hparams), which already come from the dataset's tuned config.
#
#   3d       -- zero the physical node features, so the geometric branch sees
#               nothing. The branch itself stays wired up (parameter count is
#               unchanged), which is what makes it a clean ablation.
#   infonce  -- drop the cross-modal InfoNCE term from phase 1.
#   codebook -- replace the quadratic codebook head with a plain linear head.
#               Both rank and prototype count go to zero: prototypes and task
#               weights *are* the head, so there is no half-way house.
#   fp       -- remove the fingerprint branch entirely (third block gone).
# ---------------------------------------------------------------------------
FACTOR_ORDER = ("3d", "infonce", "codebook", "fp")
FACTOR_OVERRIDES = {
    "3d":       lambda m, h: ({**m, "zero_phys_branch": True}, h),
    "infonce":  lambda m, h: (m, {**h, "cross_distill_weight": 0.0}),
    "codebook": lambda m, h: ({**m, "ih_rank": 0, "ih_num_prototypes": 0}, h),
    "fp":       lambda m, h: ({**m, "use_fingerprint": False}, h),
}

# The intact model. ih_block_proj is on for the same reason the fingerprint is:
# a third fusion block is only interpretable as a block if the head's projection
# is block-diagonal, otherwise the projection mixes geometry, topology and
# fingerprint back together before the quadratic form ever sees them.
# ih_symmetric is pinned False here too, so a --config-template pointing at a
# hand-written config cannot reintroduce the PSD head by accident.
BASE_MODEL_OVERRIDES = {"use_fingerprint": True, "ih_block_proj": True, "ih_symmetric": False}


def condition_name(conv: str, removed: tuple[str, ...]) -> str:
    return f"{conv}_full_model" if not removed else f"{conv}_no_" + "_".join(removed)


def make_override(conv: str, removed: tuple[str, ...]):
    def apply(model_kwargs: dict, hparams: dict):
        model_kwargs = {**model_kwargs, **BASE_MODEL_OVERRIDES, "gnn_conv": conv}
        for factor in removed:
            model_kwargs, hparams = FACTOR_OVERRIDES[factor](model_kwargs, hparams)
        return model_kwargs, hparams
    return apply


def _build_ablation_overrides() -> dict:
    overrides = {}
    for conv in GNN_CONV_TYPES:
        for size in range(len(FACTOR_ORDER) + 1):
            for removed in itertools.combinations(FACTOR_ORDER, size):
                overrides[condition_name(conv, removed)] = make_override(conv, removed)
    return overrides


ABLATION_OVERRIDES = _build_ablation_overrides()

# Named groups accepted by --conditions, so the common sweeps do not need all
# 48 names spelled out.
FULL_MODELS = tuple(f"{conv}_full_model" for conv in GNN_CONV_TYPES)
# The 6-cell screen: every encoder, fingerprint on and off, nothing else removed.
FAST_CELLS = tuple(
    name
    for conv in GNN_CONV_TYPES
    for name in (f"{conv}_full_model", f"{conv}_no_fp")
)


def _cells_by_removal_count(conv: str, sizes: tuple[int, ...]) -> tuple[str, ...]:
    """Conditions for one encoder that remove exactly ``sizes`` factors each."""
    return tuple(
        condition_name(conv, removed)
        for size in sizes
        for removed in itertools.combinations(FACTOR_ORDER, size)
    )


# "one factor out, two factors out, everything out" -- the 12-condition grid
# (1 + 4 + 6 + 1). ``<conv>_all`` is the full 16 including the four
# three-factor cells; ``paper`` is the 12 grid across all three encoders.
PAPER_SIZES = (0, 1, 2, 4)
CONDITION_GROUPS = {
    "all": tuple(ABLATION_OVERRIDES),
    "fast": FAST_CELLS,
    "full_models": FULL_MODELS,
    "paper": tuple(n for conv in GNN_CONV_TYPES
                   for n in _cells_by_removal_count(conv, PAPER_SIZES)),
    **{f"{conv}_all": tuple(n for n in ABLATION_OVERRIDES if n.startswith(f"{conv}_"))
       for conv in GNN_CONV_TYPES},
    **{f"{conv}_paper": _cells_by_removal_count(conv, PAPER_SIZES)
       for conv in GNN_CONV_TYPES},
    # Individual removal-count slices, for sharding a sweep by cost.
    **{f"{conv}_r{size}": _cells_by_removal_count(conv, (size,))
       for conv in GNN_CONV_TYPES for size in (1, 2, 3, 4)},
}


def expand_condition_names(names: list[str], parser=None) -> list[str]:
    """Resolve group aliases ('all', 'fast', 'full_models', '<conv>_all')."""
    resolved: list[str] = []
    for entry in names:
        key = entry.lower()
        resolved.extend(CONDITION_GROUPS[key] if key in CONDITION_GROUPS else [key])
    resolved = list(dict.fromkeys(resolved))
    unknown = [name for name in resolved if name not in ABLATION_OVERRIDES]
    if unknown:
        message = (f"Unknown condition(s): {', '.join(unknown)}. "
                   f"Groups: {', '.join(CONDITION_GROUPS)}. "
                   f"Names: {', '.join(ABLATION_OVERRIDES)}.")
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return resolved


def load_best_config(dataset: str) -> dict:
    p = PROJECT_ROOT / "dual_kd_gnn" / "optuna" / f"{dataset}_xkd" / "best_config.json"
    return json.loads(p.read_text(encoding="utf-8"))


def run_dir_for(split_type: str, condition: str, dataset: str, seed: int) -> Path:
    return RUNS_ROOT / split_type / condition / f"{dataset}_seed{seed}"


_FEATURE_CACHES: dict[str, object] = {}


def feature_cache_for(spec) -> object:
    """One featurization per dataset, reused by every run in this process."""
    key = str(spec.data_path())
    if key not in _FEATURE_CACHES:
        _FEATURE_CACHES[key] = build_cache(
            spec.data_path(), smiles_column=spec.smiles_column, verbose=True
        )
    return _FEATURE_CACHES[key]


# ---------------------------------------------------------------------------
# VRAM budget.
#
# Target card is a 24 GB RTX 3090, three of them, one training process each.
# Two things drive peak memory: the node tensor [N, H] (N is roughly
# batch_size x 25 atoms) and the head's per-task projections [B, K, r], which
# are the reason a 617-task set cannot use the same batch size as a 1-task one.
# The caps below are deliberately conservative -- an OOM 40 hours into a sweep
# costs far more than a slightly smaller batch -- and OOM_RETRY halves the batch
# again at run time if a cell still does not fit.
# ---------------------------------------------------------------------------
VRAM_SAFE_BATCH_DEFAULT = 256
MIN_BATCH_SIZE = 16


def max_batch_size(num_classes: int, gnn_hidden: int, ceiling: int | None = None) -> int:
    """Largest batch this (task count, width) is allowed on a 24 GB card."""
    if ceiling is None:
        raw = os.environ.get("DIKAT_MAX_BATCH")
        ceiling = int(raw) if raw and raw.isdigit() else VRAM_SAFE_BATCH_DEFAULT
    limit = ceiling
    if num_classes >= 256:        # ToxCast (617 tasks)
        limit = min(limit, 64)
    elif num_classes >= 32:       # nothing today, but SIDER-scale panels grow
        limit = min(limit, 128)
    if gnn_hidden >= 512:
        limit = min(limit, 128)
    return max(MIN_BATCH_SIZE, limit)


def batch_size_ladder(batch_size: int) -> list[int]:
    """Batch sizes to try in order, halving down to MIN_BATCH_SIZE."""
    ladder, size = [], int(batch_size)
    while size >= MIN_BATCH_SIZE:
        ladder.append(size)
        size //= 2
    return ladder or [MIN_BATCH_SIZE]


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_cuda() -> None:
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


@torch.no_grad()
def head_diagnostics(model) -> dict:
    """Block norms and prototype assignments of the trained head.

    These are the paper's two interpretability artifacts and they are cheap to
    read off a finished model, so they are recorded here rather than requiring
    every run's weights to be kept and re-loaded later.

    ``quadratic_block_norms`` refuses to report when a full (mixing) projection
    destroyed the block partition; that is a legitimate configuration (it is
    what ih_block_proj=False means), so it is recorded as unavailable rather
    than failing the run.
    """
    head = getattr(model, "classifier", None)
    if head is None:
        return {}
    diagnostics: dict = {
        "uses_codebook": bool(getattr(head, "use_codebook", False)),
        "blocks_preserved": bool(getattr(head, "blocks_preserved", False)),
        "block_names": list(getattr(head, "block_names", ())),
        "effective_block_dims": list(getattr(head, "effective_block_dims", ())),
    }
    try:
        norms = head.quadratic_block_norms()
        diagnostics["block_norms"] = {key: float(value) for key, value in norms.items()}
        total = sum(diagnostics["block_norms"].values())
        diagnostics["block_norms_fraction"] = (
            {key: float(value) / total for key, value in diagnostics["block_norms"].items()}
            if total > 0 else {}
        )
    except Exception as exc:
        diagnostics["block_norms"] = None
        diagnostics["block_norms_unavailable"] = str(exc)
    return diagnostics


def train_one(
    condition: str,
    dataset: str,
    seed: int,
    split_type: str,
    device: torch.device,
    run_dir: Path | None = None,
    max_batch: int | None = None,
    save_weights: bool = False,
) -> dict:
    """Train and save one (condition, dataset, seed, split_type) cell. Idempotent."""
    if split_type not in SPLIT_TYPES:
        raise ValueError(f"Unknown split_type '{split_type}'. Expected one of {SPLIT_TYPES}.")
    if run_dir is None:
        run_dir = run_dir_for(split_type, condition, dataset, seed)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return {"status": "skipped", "path": str(metrics_path)}

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    spec = DATASETS[dataset]
    data_path = str(spec.data_path())
    target_columns = (
        list(spec.target_columns) if spec.target_columns else resolve_target_columns(spec, data_path)
    )
    num_classes = len(target_columns)

    task_type = spec.task_type
    is_regression = spec.is_regression

    dataframe = pd.read_csv(data_path)
    train_idx, val_idx, test_idx = split_indices(
        dataframe,
        split_type=split_type,
        target_columns=target_columns,
        smiles_column=spec.smiles_column,
        seed=seed,
        task_type=task_type,
    )
    print(
        f"  split '{split_type}' ({task_type}): "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
        flush=True,
    )

    cache = feature_cache_for(spec)
    train_ds, val_ds, test_ds = create_dual_datasets(
        data_path, target_columns, train_idx, val_idx, test_idx,
        smiles_column=spec.smiles_column, task_type=task_type, feature_cache=cache,
    )

    # Regression standardizes targets with train-split statistics only.
    target_mean, target_scale = (
        target_statistics(dataframe, target_columns, train_idx) if is_regression else (None, None)
    )

    run_config = resolve_run_config(dataset, condition, num_molecules=len(dataframe))
    model_kwargs = dict(run_config["model_kwargs"])
    hparams = dict(run_config["hparams"])
    model_kwargs, hparams = ABLATION_OVERRIDES[condition](model_kwargs, hparams)

    # Cap the tuned batch size to what a 24 GB card holds for this task count and
    # width. The cap is applied before the first attempt so the common case never
    # pays for an OOM; the ladder below is the fallback for what the cap misses.
    requested_batch = int(hparams.get("batch_size", 64))
    capped_batch = min(
        requested_batch,
        max_batch_size(num_classes, int(model_kwargs.get("gnn_hidden", 256)), ceiling=max_batch),
    )
    if capped_batch != requested_batch:
        print(f"  [vram] batch_size {requested_batch} -> {capped_batch} "
              f"({num_classes} tasks, hidden={model_kwargs.get('gnn_hidden', 256)})",
              flush=True)

    model = trainer = test_metrics = None
    oom_history: list[int] = []
    t0 = time.time()
    for attempt_batch in batch_size_ladder(capped_batch):
        hparams["batch_size"] = attempt_batch
        try:
            set_seed(seed)  # re-seed so a retry is still a reproducible run
            model = Model(num_classes=num_classes, **model_kwargs)
            trainer = Trainer(
                model=model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                device=device,
                num_classes=num_classes,
                task_type=task_type,
                target_mean=target_mean,
                target_scale=target_scale,
                **hparams,
            )
            trainer.train()
            test_metrics = trainer.evaluate_full(test_ds, batch_size=attempt_batch)
            break
        except Exception as exc:
            if not _is_oom(exc):
                raise
            oom_history.append(attempt_batch)
            print(f"  [vram] OOM at batch_size={attempt_batch}; halving and retrying",
                  flush=True)
            model = trainer = None
            _free_cuda()
    if test_metrics is None:
        raise RuntimeError(
            f"{condition}/{dataset}/seed{seed}: out of memory even at "
            f"batch_size={MIN_BATCH_SIZE} (tried {oom_history})"
        )
    elapsed = time.time() - t0

    primary_metric = float(test_metrics["rmse"] if is_regression else test_metrics["roc_auc"])
    metrics = {
        "ablation_name": condition,
        "dataset_name": f"{dataset}_{condition}_{split_type}_seed{seed}",
        "dataset": dataset,
        "task_type": task_type,
        "metric_name": spec.metric_name,
        "greater_is_better": spec.greater_is_better,
        # Published protocol name, so downstream tables need no translation.
        "split_protocol": split_label(split_type),
        "split_type_key": split_type,
        "split_ratio": "80_10_10",
        "config_source": run_config["config_source"],
        "target_columns": target_columns if num_classes <= 32 else target_columns[:32] + ["..."],
        "num_targets": num_classes,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        # AUC definability is a classification-only concept; null for regression.
        "n_tasks_auc_defined_val":
            count_auc_defined_tasks(dataframe, target_columns, val_idx) if not is_regression else None,
        "n_tasks_auc_defined_test":
            count_auc_defined_tasks(dataframe, target_columns, test_idx) if not is_regression else None,
        "best_val_metric": float(trainer.best_val_metric),
        "best_val_auc": float(trainer.best_val_auc),  # kept for older readers
        "best_epoch": int(trainer.best_epoch),
        "test_metric": primary_metric,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "uses_dual_features": True,
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
        "batch_size": int(hparams["batch_size"]),
        "oom_retry_batch_sizes": oom_history,
        # Head interpretability, read off the best checkpoint: how much of the
        # learned quadratic form sits in each block pair, including the
        # geo-topo and geo-fp cross terms the paper's third claim rests on.
        "head_diagnostics": head_diagnostics(model),
        # Recorded so the aggregated CSV can be read without re-deriving the
        # condition's meaning from its name.
        "ablation_settings": {
            "gnn_conv": model_kwargs.get("gnn_conv", "gcn"),
            "zero_phys_branch": model_kwargs.get("zero_phys_branch", False),
            "distill_weight": hparams.get("distill_weight"),
            "cross_distill_weight": hparams.get("cross_distill_weight"),
            "ih_rank": model_kwargs.get("ih_rank"),
            "ih_num_prototypes": model_kwargs.get("ih_num_prototypes"),
            "use_fingerprint": model_kwargs.get("use_fingerprint", False),
            "fp_stage": model_kwargs.get("fp_stage", "head"),
            "ih_block_proj": model_kwargs.get("ih_block_proj", False),
        },
    }
    if is_regression:
        metrics.update({
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_r2": float(test_metrics["r2"]),
        })
    else:
        metrics["test_roc_auc"] = float(test_metrics["roc_auc"])

    # Per-epoch history: the source for the aggregated learning-curve figure.
    # Phase is recorded per row, so stage 1 and stage 2 stay separable.
    history_rows = trainer.build_history_rows()
    save_history_rows(run_dir, history_rows)
    plot_training_curves(
        history_rows,
        run_dir / "training_curves.png",
        title=f"{condition} | {dataset} | seed {seed}",
        metric_name=spec.metric_name,
        stage1_epochs=len(trainer.pretrain_train_losses),
    )
    save_json(run_dir / "run_metadata.json", {
        "condition": condition,
        "dataset": dataset,
        "seed": seed,
        "split_type": split_type,
        "data_path": str(Path(data_path).resolve()),
        "smiles_column": spec.smiles_column,
        "model_kwargs": model_kwargs,
        "hparams": hparams,
        "config_source": run_config["config_source"],
        "device": str(device),
    })
    assignment = None
    head = getattr(model, "classifier", None)
    if head is not None and getattr(head, "use_codebook", False):
        assignment = head.get_assignment_probabilities().detach().cpu().numpy()
        np.save(run_dir / "assignment_probs.npy", assignment)
    if save_weights:
        torch.save(trainer.best_state, run_dir / "model_weights.pt")

    # metrics.json is written LAST, on purpose: its presence is what marks this
    # cell complete for the idempotent skip, so it must not exist until every
    # other artifact for the run is already on disk. A sweep killed mid-run then
    # redoes that one cell instead of resuming with a half-written directory.
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return {
        "status": "done",
        "elapsed": elapsed,
        "metric": primary_metric,
        "metric_name": spec.metric_name,
        "path": str(metrics_path),
    }


def resolve_run_config(dataset: str, condition: str, num_molecules: int) -> dict:
    """Per-cell Optuna config when --config-template names one, else the default.

    A missing template file is not an error: it means that cell was never tuned,
    so the dataset's shared config is the honest fallback. The choice is recorded
    in metrics.json via config_source, so a reader can tell the two apart.
    """
    def _load(path: Path, source: str) -> dict | None:
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            return None
        config = json.loads(path.read_text(encoding="utf-8"))
        return {
            "model_kwargs": sanitize_model_kwargs(dict(config["model_kwargs"])),
            "hparams": sanitize_hparams(dict(config["hparams"])),
            "config_source": source,
        }

    if CONFIG_TEMPLATE:
        exact = Path(str(CONFIG_TEMPLATE).format(dataset=dataset, condition=condition))
        loaded = _load(exact, f"optuna_tuned:{exact.name}")
        if loaded is not None:
            return loaded

        # An ablation must differ from its reference in exactly one thing: the
        # removed factor. Falling straight through to the dataset-wide
        # <dataset>_xkd config would give the ablation a different width, a
        # different learning rate and a different dropout than the full model it
        # is subtracted from, so the paired delta would carry the hyperparameter
        # change as well. Inherit the same encoder's full-model config instead --
        # that is what "everything else held fixed" means.
        conv = condition.split("_", 1)[0]
        if conv in GNN_CONV_TYPES and not condition.endswith("full_model"):
            sibling = Path(str(CONFIG_TEMPLATE).format(
                dataset=dataset, condition=f"{conv}_full_model"))
            loaded = _load(sibling, f"optuna_tuned:{sibling.name}(full_model_sibling)")
            if loaded is not None:
                print(f"  [config] {condition}: no per-cell study — inheriting "
                      f"{conv}_full_model's tuned config", flush=True)
                return loaded

        print(f"  [config] no tuned config at {exact} and no full-model sibling — "
              f"falling back to the dataset default", flush=True)
    return load_run_config(dataset, num_molecules=num_molecules)


def expand_dataset_names(names: list[str], parser=None) -> list[str]:
    """Resolve group aliases ('all', 'classification', 'regression', 'original')."""
    resolved: list[str] = []
    for entry in names:
        key = entry.lower()
        if key == "all":
            resolved.extend(DATASETS_ALL)
        elif key == "original":
            resolved.extend(DATASETS_ORIGINAL)
        elif key in TASK_TYPES:
            resolved.extend(datasets_for_task(key))
        else:
            resolved.append(key)
    resolved = list(dict.fromkeys(resolved))
    unknown = [name for name in resolved if name not in DATASETS]
    if unknown:
        message = (f"Unknown dataset(s): {', '.join(unknown)}. "
                   f"Choices: {', '.join(DATASETS_ALL)}, classification, regression, original, all.")
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=NEW_SEEDS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS_ALL,
                        metavar="NAME",
                        help="Dataset names, or a group: 'all', 'classification', 'regression', "
                             f"'original' (the pre-expansion five). Names: {', '.join(DATASETS_ALL)}.")
    parser.add_argument("--conditions", nargs="+", default=["all"], metavar="NAME",
                        help="Condition names, or a group: "
                             f"{', '.join(CONDITION_GROUPS)}. "
                             "Names are '<conv>_full_model' / '<conv>_no_<factors>' with "
                             f"conv in {{{', '.join(GNN_CONV_TYPES)}}} and factors drawn "
                             f"from {{{', '.join(FACTOR_ORDER)}}}.")
    parser.add_argument("--config-template", default=None, metavar="PATH",
                        help="Path template for per-cell Optuna configs, with {dataset} and "
                             "{condition} placeholders, e.g. "
                             "'dual_kd_gnn/optuna/{dataset}_{condition}_v2/best_config.json'. "
                             "Cells without a file fall back to the dataset default.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                        help="Root of the results tree. Default: ablation/runs_v2 "
                             "(ablation/runs holds pre-refactor results and is not written).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batch-size", type=int, default=None, metavar="N",
                        help="Ceiling on batch size regardless of what the tuned config says. "
                             "Default: DIKAT_MAX_BATCH, else 256, further capped by task count "
                             "and encoder width for a 24 GB card. A cell that still OOMs halves "
                             "its batch and retries automatically.")
    parser.add_argument("--save-weights", action="store_true",
                        help="Also write model_weights.pt per run (~30 MB each; a full 12-dataset "
                             "sweep is tens of GB). Off by default — the block norms and "
                             "prototype assignments are already saved without it.")
    parser.add_argument("--split-type", default="scaffold",
                        choices=[*SPLIT_TYPE_CHOICES, "all"],
                        help="Split protocol for the main sweep, or 'all' to run every "
                             "protocol in turn. Default: scaffold "
                             "('deterministic_scaffold' means the same).")
    parser.add_argument("--skip-scaffold", action="store_true",
                        help="Skip the main sweep for scaffold-family protocols "
                             f"({', '.join(SCAFFOLD_SPLIT_TYPES)}).")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip everything under the 'random' protocol: the main sweep "
                             "when --split-type is random, and the matched-protocol "
                             "full_model random comparison runs.")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="Run only shard I of N (0-based), e.g. --shard 0/5. Jobs are "
                             "dealt round-robin so every shard gets a similar mix of "
                             "dataset sizes. Each run writes its own directory, so shards "
                             "may run concurrently on different GPUs without coordination.")
    args = parser.parse_args()

    shard_index, shard_count = 0, 1
    if args.shard:
        try:
            shard_index, shard_count = (int(part) for part in args.shard.split("/"))
        except ValueError:
            parser.error(f"--shard expects I/N (e.g. 0/5), got '{args.shard}'")
        if not 0 <= shard_index < shard_count:
            parser.error(f"--shard index must satisfy 0 <= I < N, got {args.shard}")
    args.datasets = expand_dataset_names(args.datasets, parser)
    args.conditions = expand_condition_names(args.conditions, parser)

    global RUNS_ROOT, CONFIG_TEMPLATE
    # Relative --runs-root means "relative to the project", not to wherever the
    # command was launched from; --config-template already worked that way.
    RUNS_ROOT = project_path(args.runs_root)
    CONFIG_TEMPLATE = args.config_template

    requested = (
        list(SPLIT_TYPES) if args.split_type == "all"
        else [resolve_split_type(args.split_type)]
    )
    sweep_split_types = [
        split_type for split_type in requested
        if not (args.skip_scaffold and split_type in SCAFFOLD_SPLIT_TYPES)
        and not (args.skip_random and split_type == "random")
    ]

    device = get_device(args.device)
    configure_performance()
    grouped = {
        task_type: [d for d in args.datasets if DATASETS[d].task_type == task_type]
        for task_type in TASK_TYPES
    }
    print(f"Device: {device}")
    print(f"Runs root: {RUNS_ROOT}")
    if CONFIG_TEMPLATE:
        print(f"Config template: {CONFIG_TEMPLATE}")
    print(f"Seeds:  {args.seeds}")
    print(f"Conditions ({len(args.conditions)}):  {args.conditions}")
    print(f"Datasets:    classification={grouped['classification']} regression={grouped['regression']}")
    print(f"Split types: {sweep_split_types or '(none — all skipped)'}")

    # Build the full job list first so it can be sharded and counted exactly.
    # Dataset-major ordering plus round-robin dealing means each shard gets a
    # similar mix of cheap and expensive datasets.
    jobs: list[tuple[str, str, str, int, Path | None]] = []
    for split_type in sweep_split_types:
        for condition in args.conditions:
            for dataset in args.datasets:
                # label_aware_scaffold buckets scaffolds by class balance, so it
                # has no meaning for continuous targets.
                if split_type == "label_aware_scaffold" and DATASETS[dataset].is_regression:
                    continue
                for seed in args.seeds:
                    jobs.append((split_type, condition, dataset, seed, None))

    # Matched-protocol comparison against baselines that report random splits.
    # Keeps its historical ablation/runs/full_model_random/ directory: that path
    # has only ever held random-split runs, so it needs no protocol separation,
    # and reusing it keeps the existing runs there idempotently skippable.
    random_datasets = [] if args.skip_random else [d for d in args.datasets if d in DATASETS_RANDOM]
    for dataset in random_datasets:
        for seed in args.seeds:
            jobs.append(("random", FULL_MODELS[0], dataset, seed,
                         RUNS_ROOT / RANDOM_RUNS_DIR_NAME / f"{dataset}_seed{seed}"))

    all_jobs = len(jobs)
    if shard_count > 1:
        jobs = [job for index, job in enumerate(jobs) if index % shard_count == shard_index]
        print(f"Shard {shard_index}/{shard_count}: {len(jobs)} of {all_jobs} jobs")
    total_target = len(jobs)
    print(f"Target new runs (upper bound; idempotent skips reduce actual work): {total_target}\n")

    done = skipped = failed = 0

    def record(tag: str, condition: str, dataset: str, seed: int, split_type: str,
               run_dir: Path | None = None) -> None:
        nonlocal done, skipped, failed
        try:
            result = train_one(condition, dataset, seed, split_type, device,
                               run_dir=run_dir, max_batch=args.max_batch_size,
                               save_weights=args.save_weights)
        except Exception as e:
            print(f"{tag} FAILED: {e}", flush=True)
            failed += 1
            return
        if result["status"] == "skipped":
            skipped += 1
            print(f"{tag} skip (exists)", flush=True)
        else:
            done += 1
            print(f"{tag} done  {result['metric_name']}={result['metric']:.4f}  "
                  f"({result['elapsed']/60:.1f} min)  [{done+skipped}/{total_target}]", flush=True)

    for split_type, condition, dataset, seed, run_dir in jobs:
        label = "full_model_random" if run_dir is not None else split_type
        record(f"[{label:22s}][{condition:16s}][{dataset:9s}][seed={seed:3d}]",
               condition, dataset, seed, split_type, run_dir=run_dir)

    print(f"\nSummary: done={done}, skipped={skipped}, failed={failed}, target={total_target}")
    if done > 0:
        primary = sweep_split_types[0] if sweep_split_types else "random"
        print("\nNext steps:")
        print("  1. Regenerate the summary CSVs:")
        print(f"       python ablation/main.py --runs-dir {RUNS_ROOT}")
        print("  2. Re-run Wilcoxon tests with n=15:")
        print(f"       python scripts/revision_experiments.py --task c1 --split-type {primary}")
        print("  3. Recompute CIs:")
        print(f"       python scripts/compute_ci.py --split-type {primary}")


if __name__ == "__main__":
    main()
