from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common.config import (
    DEFAULT_SEED,
    ModelSpec,
    get_device,
    get_project_root,
    set_seed,
)
from common.data import (
    SPLIT_TYPE_CHOICES,
    create_datasets,
    resolve_split_type,
    split_indices,
    split_label,
    target_statistics,
)
from common.datasets import (
    DEFAULT_DATASET,
    available_datasets,
    get_dataset_spec,
    resolve_target_columns,
)
from common.featurize_cache import build_cache
from common.io_utils import save_run_artifacts
from common.plotting import plot_training_curves
from common.trainer import Trainer


OVERRIDABLE_HPARAMS = [
    "batch_size",
    "lr",
    "weight_decay",
    "num_epochs",
    "patience",
    "gcn_pretrain_epochs",
    "head_epochs",
    "pretrain_lr",
    "head_lr",
    "ema_decay",
    "ema_decay_init",
    "distill_weight",
    "cross_distill_weight",
]


def add_dataset_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=available_datasets(),
        help="Registered dataset to train on. Resolves data path, SMILES column, and target tasks.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the CSV path inferred from --dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Label used for saved run directories. Defaults to --dataset.",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Override the SMILES column inferred from --dataset.",
    )
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=None,
        help="Override the target columns inferred from --dataset.",
    )
    return parser


def add_general_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (single run).")
    parser.add_argument(
        "--ablation-name",
        default=None,
        help=(
            "Ablation label (e.g. 'a1_no_phys'). When set, artifacts are saved to "
            "ablation/runs/<name>/<dataset>/ instead of dual_kd_gnn/runs/<dataset>/."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run with multiple seeds and report mean±std AUROC. Overrides --seed. E.g.: --seeds 42 0 1 2 3",
    )
    parser.add_argument(
        "--split-type",
        default="scaffold",
        choices=SPLIT_TYPE_CHOICES,
        help="Split protocol. Default: scaffold (Bemis-Murcko; 'deterministic_scaffold' "
             "means the same). label_aware_scaffold is classification-only.",
    )
    parser.add_argument("--device", default="cuda", help="Compute device. Defaults to cuda; pass cpu/mps/cuda:N to override.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--gcn-pretrain-epochs", type=int, default=None)
    parser.add_argument("--head-epochs", type=int, default=None,
                        help="Phase-2 (codebook head fine-tuning) epoch budget.")
    parser.add_argument("--pretrain-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None,
                        help="Phase-2 learning rate.")
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--ema-decay-init", type=float, default=None)
    parser.add_argument("--distill-weight", type=float, default=None)
    parser.add_argument("--cross-distill-weight", type=float, default=None)
    return parser


def add_shared_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_dataset_arguments(parser)
    add_general_training_arguments(parser)
    return parser


def build_single_model_parser(spec: ModelSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train {spec.name}.")
    add_shared_training_arguments(parser)
    if spec.add_model_arguments is not None:
        spec.add_model_arguments(parser)
    return parser


def collect_override_hparams(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "gcn_pretrain_epochs": args.gcn_pretrain_epochs,
        "head_epochs": args.head_epochs,
        "pretrain_lr": args.pretrain_lr,
        "head_lr": args.head_lr,
        "ema_decay": args.ema_decay,
        "ema_decay_init": args.ema_decay_init,
        "distill_weight": args.distill_weight,
        "cross_distill_weight": args.cross_distill_weight,
    }


def merge_hparams(defaults: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    hparams = dict(defaults)
    for key, value in (overrides or {}).items():
        if value is not None and key in defaults:
            hparams[key] = value
    return hparams


_FEATURE_CACHES: dict[str, Any] = {}


def feature_cache_for(data_path: str, smiles_column: str):
    """One featurization per CSV, reused by every run in this process.

    The dual featurizer embeds a conformer and runs MMFF94 per molecule; without
    this the cost is paid again for every seed, which is hours on HIV and CEP.
    """
    key = str(Path(data_path).resolve())
    if key not in _FEATURE_CACHES:
        _FEATURE_CACHES[key] = build_cache(data_path, smiles_column=smiles_column, verbose=True)
    return _FEATURE_CACHES[key]


def run_experiment(
    spec: ModelSpec,
    data_path: str,
    dataset_name: str,
    seed: int,
    device_name: str | None,
    target_columns: list[str],
    model_dir: Path,
    overrides: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    smiles_column: str = "smiles",
    ablation_name: str | None = None,
    task_type: str = "classification",
    split_type: str = "deterministic_scaffold",
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device(device_name)
    hparams = merge_hparams(spec.default_hparams, overrides)
    model_kwargs = model_kwargs or {}
    num_classes = len(target_columns)
    split_type = resolve_split_type(split_type)
    is_regression = task_type == "regression"

    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name} | targets ({num_classes}): {', '.join(target_columns)}")
    print(f"Task type: {task_type} | split: {split_label(split_type)}")
    feature_cache = (
        feature_cache_for(data_path, smiles_column) if spec.uses_dual_features else None
    )
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_path=data_path,
        target_columns=target_columns,
        seed=seed,
        dual=spec.uses_dual_features,
        smiles_column=smiles_column,
        split_type=split_type,
        task_type=task_type,
        feature_cache=feature_cache,
    )

    # Regression trains on targets standardized with train-split statistics only
    # and reports RMSE back in the original units. Without these the Trainer
    # would fall back to masked BCE + ROC-AUC on continuous targets, which runs
    # without raising and produces a meaningless number.
    target_mean = target_scale = None
    if is_regression:
        dataframe = pd.read_csv(data_path)
        train_idx, _, _ = split_indices(
            dataframe,
            split_type=split_type,
            target_columns=target_columns,
            smiles_column=smiles_column,
            seed=seed,
            task_type=task_type,
        )
        target_mean, target_scale = target_statistics(dataframe, target_columns, train_idx)

    model = spec.builder(num_classes=num_classes, **model_kwargs)
    train_start = time.time()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        num_classes=len(target_columns),
        task_type=task_type,
        target_mean=target_mean,
        target_scale=target_scale,
        **hparams,
    )
    trainer.train()
    batch_size = int(hparams.get("batch_size", 64))
    test_metrics = trainer.evaluate_full(test_dataset, batch_size=batch_size)
    test_auc = float(test_metrics["rmse"] if is_regression else test_metrics["roc_auc"])
    elapsed_seconds = time.time() - train_start

    history_rows = trainer.build_history_rows()

    if ablation_name:
        run_dir = get_project_root() / "ablation" / "runs" / ablation_name / dataset_name
    else:
        run_dir = model_dir / "runs" / dataset_name

    # Field names match what seed_expansion.py writes, so single runs and sweep
    # runs land in the same aggregated table instead of relying on fallbacks.
    metrics: dict[str, Any] = {
        "model_name": spec.name,
        "model_slug": spec.slug,
        "dataset_name": dataset_name,
        "dataset": dataset_name.split("_seed")[0],
        "task_type": task_type,
        "metric_name": "rmse" if is_regression else "roc_auc",
        "greater_is_better": not is_regression,
        "split_protocol": split_label(split_type),
        "split_type_key": split_type,
        "target_columns": target_columns,
        "num_targets": len(target_columns),
        "best_val_metric": float(trainer.best_val_metric),
        "best_val_auc": float(trainer.best_val_auc),  # kept for older readers
        "best_epoch": int(trainer.best_epoch),
        "test_metric": test_auc,
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "uses_dual_features": spec.uses_dual_features,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "seed": seed,
    }
    if is_regression:
        metrics.update({
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_r2": float(test_metrics["r2"]),
        })
    else:
        metrics["test_roc_auc"] = float(test_metrics["roc_auc"])
    if ablation_name is not None:
        metrics["ablation_name"] = ablation_name
        metrics["ablation_settings"] = {
            "zero_phys_branch": model_kwargs.get("zero_phys_branch", False),
            "distill_weight": hparams.get("distill_weight"),
            "cross_distill_weight": hparams.get("cross_distill_weight"),
            "ih_rank": model_kwargs.get("ih_rank"),
            "ih_num_prototypes": model_kwargs.get("ih_num_prototypes"),
            "use_fingerprint": model_kwargs.get("use_fingerprint", False),
            "fp_stage": model_kwargs.get("fp_stage"),
            "ih_block_proj": model_kwargs.get("ih_block_proj", False),
            "gnn_conv": model_kwargs.get("gnn_conv", "gcn"),
        }

    save_run_artifacts(
        run_dir=run_dir,
        history_rows=history_rows,
        metrics=metrics,
        metadata={
            "data_path": str(Path(data_path).resolve()),
            "smiles_column": smiles_column,
            "hparams": hparams,
            "model_kwargs": model_kwargs,
            "device": str(device),
            "model_notes": spec.notes,
        },
    )
    torch.save(trainer.best_state, run_dir / "model_weights.pt")
    plot_training_curves(history_rows, run_dir / "training_curves.png", title=f"{spec.slug} | {dataset_name}")

    label = trainer.metric_name.upper()
    print(f"Saved run artifacts to: {run_dir}")
    print(f"  Best Val {label}: {trainer.best_val_metric:.4f}")
    print(f"  Test {label}:     {test_auc:.4f}")
    return metrics


def resolve_dataset_inputs(args: argparse.Namespace) -> tuple[str, str, list[str], str]:
    """Resolve (data_path, dataset_name, target_columns, smiles_column).

    Registered ``--dataset`` metadata provides defaults; explicit CLI flags
    (``--data-path``, ``--target-columns``, ``--smiles-column``,
    ``--dataset-name``) override them.
    """
    spec = get_dataset_spec(args.dataset)
    data_path = args.data_path or str(spec.data_path())
    if not Path(data_path).exists():
        raise SystemExit(
            f"Dataset file not found: {data_path}\n"
            f"Download it first, e.g.: python scripts/download_data.py {spec.name}"
        )
    smiles_column = args.smiles_column or spec.smiles_column
    target_columns = (
        list(args.target_columns)
        if args.target_columns is not None
        else resolve_target_columns(spec, data_path)
    )
    dataset_name = args.dataset_name or spec.name
    return data_path, dataset_name, target_columns, smiles_column


def run_multi_seed(
    spec: ModelSpec,
    data_path: str,
    dataset_name: str,
    seeds: list[int],
    device_name: str | None,
    target_columns: list[str],
    model_dir: Path,
    overrides: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    smiles_column: str = "smiles",
    ablation_name: str | None = None,
    task_type: str = "classification",
    split_type: str = "deterministic_scaffold",
) -> dict[str, Any]:
    """Run the same experiment over multiple seeds and report mean ± std.

    The aggregated quantity is ``test_metric`` — ROC-AUC for classification,
    RMSE for regression — so this works for both task types.
    """
    metric_name = "rmse" if task_type == "regression" else "roc_auc"
    aucs: list[float] = []
    for seed in seeds:
        run_ds_name = f"{dataset_name}_seed{seed}"
        metrics = run_experiment(
            spec=spec,
            data_path=data_path,
            dataset_name=run_ds_name,
            seed=seed,
            device_name=device_name,
            target_columns=target_columns,
            model_dir=model_dir,
            overrides=overrides,
            model_kwargs=model_kwargs,
            smiles_column=smiles_column,
            ablation_name=ablation_name,
            task_type=task_type,
            split_type=split_type,
        )
        aucs.append(metrics["test_metric"])

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    sep = "=" * 60
    label = metric_name.upper()
    print(f"\n{sep}")
    print(f"Multi-seed results ({len(seeds)} seeds) on {dataset_name}:")
    print(f"  Seeds:            {seeds}")
    print(f"  Per-seed {label:<8}: {', '.join(f'{v:.4f}' for v in aucs)}")
    print(f"  Mean ± Std:       {mean_auc:.4f} ± {std_auc:.4f}")
    print(sep)
    return {
        "dataset_name": dataset_name,
        "task_type": task_type,
        "metric_name": metric_name,
        "mean_test_metric": mean_auc,
        "std_test_metric": std_auc,
        "per_seed_metrics": aucs,
        # Legacy keys, so existing readers of this return value keep working.
        "mean_test_roc_auc": mean_auc,
        "std_test_roc_auc": std_auc,
        "per_seed_aucs": aucs,
        "seeds": seeds,
    }


def run_from_cli(spec: ModelSpec, model_dir: Path) -> dict[str, Any]:
    parser = build_single_model_parser(spec)
    args = parser.parse_args()
    data_path, dataset_name, target_columns, smiles_column = resolve_dataset_inputs(args)
    model_kwargs = spec.collect_model_kwargs(args) if spec.collect_model_kwargs is not None else {}
    hparam_overrides = spec.collect_hparam_overrides(args) if spec.collect_hparam_overrides is not None else {}
    hparam_overrides.update(
        {
            key: value
            for key, value in collect_override_hparams(args).items()
            if value is not None
        }
    )
    seeds = args.seeds if args.seeds is not None else [args.seed]
    ablation_name = getattr(args, "ablation_name", None)
    # Task type comes from the dataset registry, never from a flag: it decides
    # the loss, the reported metric and the early-stopping direction together,
    # and letting those three disagree is how a regression set silently gets
    # trained with BCE.
    task_type = get_dataset_spec(args.dataset).task_type
    split_type = resolve_split_type(getattr(args, "split_type", "scaffold"))
    if len(seeds) > 1:
        return run_multi_seed(
            spec=spec,
            data_path=data_path,
            dataset_name=dataset_name,
            seeds=seeds,
            device_name=args.device,
            target_columns=target_columns,
            model_dir=model_dir,
            overrides=hparam_overrides,
            model_kwargs=model_kwargs,
            smiles_column=smiles_column,
            ablation_name=ablation_name,
            task_type=task_type,
            split_type=split_type,
        )
    return run_experiment(
        spec=spec,
        data_path=data_path,
        dataset_name=dataset_name,
        seed=seeds[0],
        device_name=args.device,
        target_columns=target_columns,
        model_dir=model_dir,
        overrides=hparam_overrides,
        model_kwargs=model_kwargs,
        smiles_column=smiles_column,
        ablation_name=ablation_name,
        task_type=task_type,
        split_type=split_type,
    )
