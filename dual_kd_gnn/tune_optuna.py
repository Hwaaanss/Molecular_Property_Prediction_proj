from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Must run before torch/numpy load (see common/resources).
from common.resources import apply_thread_limits  # noqa: E402

apply_thread_limits(verbose=True)

try:
    import optuna
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise SystemExit("Optuna is required for this script. Install it with: pip install optuna") from exc

import pandas as pd

from common.config import DEFAULT_SEED, configure_performance, get_device, project_path, set_seed
from common.data import create_datasets, split_indices, target_statistics
from common.datasets import (
    DEFAULT_DATASET,
    available_datasets,
    get_dataset_spec,
    resolve_target_columns,
)
from common.featurize_cache import build_cache
from common.io_utils import ensure_dir, save_json, save_run_artifacts
from common.plotting import plot_training_curves
from common.trainer import Trainer
from dual_kd_gnn.configs import sanitize_hparams, sanitize_model_kwargs
from dual_kd_gnn.model import GNN_CONV_TYPES, DualDistillationModel


MODEL_NAME = "dual_distillation"
MODEL_SLUG = "dual_kd_gnn"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune dual_kd_gnn with Optuna and save reproducible results.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=available_datasets(),
        help="Registered dataset to tune on. Resolves data path, SMILES column, and target tasks.",
    )
    parser.add_argument("--data-path", default=None, help="Override the CSV path inferred from --dataset.")
    parser.add_argument("--dataset-name", default=None, help="Label used for saved run directories. Defaults to --dataset.")
    parser.add_argument("--smiles-column", default=None, help="Override the SMILES column inferred from --dataset.")
    parser.add_argument("--target-columns", nargs="+", default=None, help="Override the target columns inferred from --dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda", help="Compute device. Defaults to cuda; pass cpu/mps/cuda:N to override.")
    parser.add_argument("--study-name", default="dual_kd_gnn_optuna")
    parser.add_argument("--storage", default=None, help="Optuna storage URL. Defaults to a local SQLite DB.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / MODEL_SLUG / "optuna")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None, help="Maximum tuning time in seconds.")
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--pruner", choices=["hyperband", "median", "none"], default="hyperband")
    parser.add_argument("--pruner-warmup-steps", type=int, default=8)
    parser.add_argument("--gcn-pretrain-epochs", type=int, default=150)
    parser.add_argument("--head-epochs", type=int, default=150,
                        help="Phase-2 (codebook head fine-tuning) epoch budget.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--gnn-conv",
        choices=list(GNN_CONV_TYPES),
        default="gcn",
        help="Message-passing convolution, fixed for the whole study. Not searched: "
             "it changes every parameter shape, so one study per encoder keeps the "
             "trials of a study comparable with each other.",
    )
    fingerprint = parser.add_mutually_exclusive_group()
    fingerprint.add_argument("--use-fingerprint", dest="use_fingerprint",
                             action="store_true", default=True,
                             help="Tune with the fingerprint branch on (default).")
    fingerprint.add_argument("--no-fingerprint", dest="use_fingerprint",
                             action="store_false",
                             help="Tune with the fingerprint branch removed.")
    parser.add_argument("--replay-best", type=Path, default=None, help="Train once from a saved best_config.json.")
    parser.add_argument("--replay-run-name", default=None, help="Run directory name for --replay-best.")
    return parser


def build_sampler(args: argparse.Namespace) -> optuna.samplers.BaseSampler:
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.seed)
    return optuna.samplers.TPESampler(seed=args.seed, multivariate=True)


def build_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=args.pruner_warmup_steps)
    return optuna.pruners.HyperbandPruner(min_resource=max(args.pruner_warmup_steps, 1), reduction_factor=3)


def storage_url_for(args: argparse.Namespace, study_dir: Path) -> str:
    if args.storage:
        return args.storage
    db_path = study_dir / "study.db"
    return f"sqlite:///{db_path.resolve()}"


# ---------------------------------------------------------------------------
# Per-dataset Optuna search space.
#
# BASE_SEARCH_SPACE defines the default range for every tunable parameter.
# DATASET_SEARCH_SPACES holds per-dataset overrides: only the listed keys
# replace the base entry; everything else falls back to BASE_SEARCH_SPACE.
# resolve_search_space(dataset) merges the two, and run_tuning() selects the
# dataset automatically from --dataset, so no extra flag is needed.
#
# Entry formats consumed by suggest_param():
#   {"type": "categorical", "choices": [...]}
#   {"type": "int",   "low": int,   "high": int}
#   {"type": "float", "low": float, "high": float, "log": bool (default False)}
# ---------------------------------------------------------------------------

# Structural knobs (widths, depths, prototype counts, dropout levels) are drawn
# from explicit discrete sets rather than continuous ranges. Two reasons: a
# continuous width is meaningless, and with a 5-trial budget a continuous range
# mostly buys near-duplicate configurations. Genuinely continuous, genuinely
# sensitive quantities -- the two learning rates, weight decay, the distillation
# weight -- stay log-uniform.
BASE_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    # model kwargs (structural: discrete)
    "gnn_hidden": {"type": "categorical", "choices": [128, 192, 256, 384]},
    "gnn_layers": {"type": "int", "low": 2, "high": 4},
    "gnn_dropout": {"type": "categorical", "choices": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "fusion_dropout": {"type": "categorical", "choices": [0.0, 0.1, 0.2, 0.3, 0.4]},
    "ih_rank": {"type": "categorical", "choices": [16, 32, 64]},
    # ih_symmetric is not searched: the symmetric (PSD) head cannot represent a
    # signed molecule-dependent output now that the linear term is gone, and it
    # collapses to bias-only on imbalanced or regression targets. Pinned False.
    "ih_proj_dim": {"type": "categorical", "choices": [0, 128, 256]},
    "ih_num_prototypes": {"type": "categorical", "choices": [3, 4, 6, 8, 12]},
    "ih_assignment_mode": {"type": "categorical", "choices": ["hard", "soft", "sparse"]},
    "ih_diversity_weight": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "info_nce_temperature": {"type": "categorical", "choices": [0.1, 0.2, 0.5]},
    # Only sampled when the study runs with the fingerprint branch on.
    "fp_dim": {"type": "categorical", "choices": [32, 64, 128]},
    "fp_dropout": {"type": "categorical", "choices": [0.1, 0.2, 0.3]},
    # hparams
    "batch_size": {"type": "categorical", "choices": [64, 128]},
    "head_lr": {"type": "float", "low": 1e-5, "high": 3e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 3e-3, "log": True},
    "pretrain_lr": {"type": "float", "low": 1e-5, "high": 3e-3, "log": True},
    "ema_decay": {"type": "float", "low": 0.95, "high": 0.999},
    "ema_decay_init": {"type": "categorical", "choices": [0.90, 0.95, 0.98, 0.99]},
    "distill_weight": {"type": "float", "low": 1e-3, "high": 0.2, "log": True},
    "cross_distill_weight": {"type": "categorical", "choices": [0.02, 0.05, 0.1]},
}

DATASET_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    # bace: small (1.5k) and easy to overfit; the encoders now carry the whole
    # representation, so capacity sits in gnn_hidden rather than in a stack of
    # transformer layers that no longer exists.
    "bace": {
        "gnn_hidden": {"type": "categorical", "choices": [256, 384, 512]},
        "ih_proj_dim": {"type": "categorical", "choices": [0, 128]},
        "ih_num_prototypes": {"type": "categorical", "choices": [4, 6, 8]},
        "info_nce_temperature": {"type": "categorical", "choices": [0.2, 0.5, 1.0]},
        "head_lr": {"type": "float", "low": 1e-4, "high": 2e-3, "log": True},
        "pretrain_lr": {"type": "float", "low": 1e-4, "high": 2e-3, "log": True},
    },
    # tox21: 12 tasks, mildly underfit. Add capacity and more codebook
    # prototypes; keep dropout in the higher band the tuned run liked.
    "tox21": {
        "gnn_hidden": {"type": "categorical", "choices": [128, 192, 256, 384, 512]},
        "gnn_dropout": {"type": "categorical", "choices": [0.3, 0.4, 0.5]},
        "ih_proj_dim": {"type": "categorical", "choices": [128, 256]},
        "ih_num_prototypes": {"type": "categorical", "choices": [8, 12, 16, 24]},
        "ema_decay": {"type": "float", "low": 0.97, "high": 0.999},
        "info_nce_temperature": {"type": "categorical", "choices": [0.2, 0.5, 1.0]},
        "distill_weight": {"type": "float", "low": 5e-4, "high": 0.05, "log": True},
        "head_lr": {"type": "float", "low": 1e-4, "high": 2e-3, "log": True},
    },
    # clintox: kept on BASE_SEARCH_SPACE unchanged (do not adjust).
    "clintox": {},
    # bbbp / sider: not yet specialized; inherit the base space.
    "bbbp": {},
    "sider": {},
}


def resolve_search_space(dataset: str) -> dict[str, dict[str, Any]]:
    """Merge BASE_SEARCH_SPACE with the dataset-specific overrides (if any)."""
    space = {name: dict(spec) for name, spec in BASE_SEARCH_SPACE.items()}
    for name, spec in DATASET_SEARCH_SPACES.get(dataset, {}).items():
        space[name] = dict(spec)
    return space


def suggest_param(trial: optuna.Trial, name: str, space: dict[str, dict[str, Any]]) -> Any:
    spec = space[name]
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    raise ValueError(f"Unknown search-space type for {name!r}: {kind!r}")


def sample_model_kwargs(
    trial: optuna.Trial, space: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    model_kwargs = {
        "gnn_hidden": suggest_param(trial, "gnn_hidden", space),
        "gnn_layers": suggest_param(trial, "gnn_layers", space),
        "gnn_dropout": suggest_param(trial, "gnn_dropout", space),
        "fusion_dropout": suggest_param(trial, "fusion_dropout", space),
        "ih_rank": suggest_param(trial, "ih_rank", space),
        "ih_symmetric": False,
        "ih_proj_dim": suggest_param(trial, "ih_proj_dim", space),
        "ih_num_prototypes": suggest_param(trial, "ih_num_prototypes", space),
        "ih_assignment_mode": suggest_param(trial, "ih_assignment_mode", space),
        "ih_diversity_weight": suggest_param(trial, "ih_diversity_weight", space),
        "info_nce_temperature": suggest_param(trial, "info_nce_temperature", space),
        # Fixed for the study, not searched: the encoder defines the parameter
        # shapes, and the block-diagonal projection is what makes the head's
        # block decomposition readable at all.
        "gnn_conv": args.gnn_conv,
        "ih_block_proj": True,
        "use_fingerprint": bool(args.use_fingerprint),
    }
    if args.use_fingerprint:
        model_kwargs["fp_dim"] = suggest_param(trial, "fp_dim", space)
        model_kwargs["fp_dropout"] = suggest_param(trial, "fp_dropout", space)
    return model_kwargs


def sample_hparams(
    trial: optuna.Trial, space: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    head_lr = suggest_param(trial, "head_lr", space)
    return {
        "batch_size": suggest_param(trial, "batch_size", space),
        "lr": head_lr,
        "weight_decay": suggest_param(trial, "weight_decay", space),
        "num_epochs": max(args.gcn_pretrain_epochs, args.head_epochs),
        "patience": args.patience,
        "gcn_pretrain_epochs": args.gcn_pretrain_epochs,
        "head_epochs": args.head_epochs,
        "pretrain_lr": suggest_param(trial, "pretrain_lr", space),
        "head_lr": head_lr,
        "ema_decay": suggest_param(trial, "ema_decay", space),
        "ema_decay_init": suggest_param(trial, "ema_decay_init", space),
        "distill_weight": suggest_param(trial, "distill_weight", space),
        "cross_distill_weight": suggest_param(trial, "cross_distill_weight", space),
    }


def build_epoch_callback(trial: optuna.Trial):
    def callback(event: dict[str, object]) -> None:
        if event.get("phase") != "stage2_head":
            return
        value = float(event["val_metric"])
        step = int(event["epoch"])
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at head epoch {step} with val metric {value:.6f}")

    return callback


def build_metrics(
    *,
    dataset_name: str,
    target_columns: list[str],
    best_val_auc: float,
    best_epoch: int,
    model: DualDistillationModel,
    elapsed_seconds: float,
    status: str,
    test_roc_auc: float | None = None,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "model_name": MODEL_NAME,
        "model_slug": MODEL_SLUG,
        "dataset_name": dataset_name,
        "target_columns": target_columns,
        "num_targets": len(target_columns),
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "test_roc_auc": math.nan if test_roc_auc is None else test_roc_auc,
        "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "uses_dual_features": True,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "status": status,
    }
    return metrics


def best_seen_by_trainer(trainer: Trainer) -> tuple[float, int]:
    """Best validation score a (possibly pruned) trial reached.

    Direction-aware: ROC-AUC is maximized, RMSE minimized.
    """
    if trainer.val_aucs:
        pick = max if trainer.greater_is_better else min
        best_val = pick(trainer.val_aucs)
        best_epoch = trainer.val_aucs.index(best_val) + 1
        return float(best_val), int(best_epoch)
    return float(trainer.best_val_metric), int(trainer.best_epoch)


def train_once(
    *,
    train_dataset,
    val_dataset,
    test_dataset,
    target_columns: list[str],
    dataset_name: str,
    seed: int,
    device_name: str | None,
    model_kwargs: dict[str, Any],
    hparams: dict[str, Any],
    run_dir: Path,
    metadata: dict[str, object],
    epoch_callback=None,
    evaluate_test: bool = False,
    save_weights: bool = False,
    task_type: str = "classification",
    target_mean=None,
    target_scale=None,
) -> dict[str, object]:
    set_seed(seed)
    device = get_device(device_name)
    model = DualDistillationModel(num_classes=len(target_columns), **model_kwargs)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        num_classes=len(target_columns),
        epoch_callback=epoch_callback,
        task_type=task_type,
        target_mean=target_mean,
        target_scale=target_scale,
        **hparams,
    )

    start = time.time()
    status = "complete"
    try:
        trainer.train()
    except optuna.TrialPruned:
        status = "pruned"
        elapsed = time.time() - start
        best_val_auc, best_epoch = best_seen_by_trainer(trainer)
        metrics = build_metrics(
            dataset_name=dataset_name,
            target_columns=target_columns,
            best_val_auc=best_val_auc,
            best_epoch=best_epoch,
            model=model,
            elapsed_seconds=elapsed,
            status=status,
        )
        save_run_artifacts(run_dir, trainer.build_history_rows(), metrics, metadata={**metadata, "status": status})
        raise

    test_auc = None
    if evaluate_test:
        test_auc = float(trainer.evaluate(test_dataset, batch_size=int(hparams["batch_size"])))

    elapsed = time.time() - start
    metrics = build_metrics(
        dataset_name=dataset_name,
        target_columns=target_columns,
        best_val_auc=float(trainer.best_val_metric),
        best_epoch=int(trainer.best_epoch),
        model=model,
        elapsed_seconds=elapsed,
        status=status,
        test_roc_auc=test_auc,
    )
    history_rows = trainer.build_history_rows()
    save_run_artifacts(run_dir, history_rows, metrics, metadata={**metadata, "status": status})
    if save_weights:
        import torch

        torch.save(trainer.best_state, run_dir / "model_weights.pt")
        plot_training_curves(history_rows, run_dir / "training_curves.png", title=f"{MODEL_SLUG} | {dataset_name}")
    return metrics


def save_trials_csv(study: optuna.Study, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "number",
                "state",
                "value",
                "datetime_start",
                "datetime_complete",
                "duration_seconds",
                "params_json",
                "user_attrs_json",
            ],
        )
        writer.writeheader()
        for trial in study.trials:
            duration = trial.duration.total_seconds() if trial.duration is not None else math.nan
            writer.writerow(
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value if trial.value is not None else math.nan,
                    "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else "",
                    "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else "",
                    "duration_seconds": duration,
                    "params_json": json.dumps(trial.params, ensure_ascii=True, sort_keys=True),
                    "user_attrs_json": json.dumps(trial.user_attrs, ensure_ascii=True, sort_keys=True),
                }
            )


def save_best_config(study: optuna.Study, study_dir: Path, base_config: dict[str, object]) -> Path | None:
    save_trials_csv(study, study_dir / "trials.csv")
    try:
        best_trial = study.best_trial
    except ValueError:
        return None

    config = {
        **base_config,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "model_kwargs": best_trial.user_attrs["model_kwargs"],
        "hparams": best_trial.user_attrs["hparams"],
        "metric": "best_val_metric",
        "metric_name": base_config.get("metric_name", "roc_auc"),
        "direction": base_config.get("direction", "maximize"),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    best_config_path = study_dir / "best_config.json"
    config["replay_command"] = f"python dual_kd_gnn/tune_optuna.py --replay-best {best_config_path}"
    save_json(best_config_path, config)
    save_json(study_dir / "best_trial.json", {"number": best_trial.number, "value": best_trial.value})
    return best_config_path


def run_tuning(args: argparse.Namespace) -> None:
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

    study_dir = ensure_dir(project_path(args.output_dir) / args.study_name)
    storage_url = storage_url_for(args, study_dir)

    # Featurize once for the whole study; every trial reuses the same splits.
    feature_cache = build_cache(data_path, smiles_column=smiles_column)
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_path=data_path,
        target_columns=target_columns,
        seed=args.seed,
        dual=True,
        smiles_column=smiles_column,
        task_type=spec.task_type,
        feature_cache=feature_cache,
    )
    target_mean = target_scale = None
    if spec.is_regression:
        dataframe = pd.read_csv(data_path)
        train_idx, _, _ = split_indices(
            dataframe, target_columns=target_columns, smiles_column=smiles_column,
            seed=args.seed, task_type=spec.task_type,
        )
        target_mean, target_scale = target_statistics(dataframe, target_columns, train_idx)

    sampler = build_sampler(args)
    pruner = build_pruner(args)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        # RMSE is minimized, ROC-AUC maximized.
        direction="maximize" if spec.greater_is_better else "minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # Auto-select the search space from the chosen dataset (falls back to BASE).
    search_space = resolve_search_space(spec.name)
    specialized = spec.name in DATASET_SEARCH_SPACES and DATASET_SEARCH_SPACES[spec.name]
    print(
        f"Search space for '{spec.name}': "
        f"{'dataset-specific overrides applied' if specialized else 'base space (no overrides)'}"
    )

    base_config: dict[str, object] = {
        "study_name": args.study_name,
        "storage": storage_url,
        "dataset": spec.name,
        "data_path": str(Path(data_path).resolve()),
        "dataset_name": dataset_name,
        "smiles_column": smiles_column,
        "target_columns": target_columns,
        "seed": args.seed,
        "device": args.device,
        "search_space": search_space,
        "gnn_conv": args.gnn_conv,
        "use_fingerprint": bool(args.use_fingerprint),
        "task_type": spec.task_type,
        "metric_name": spec.metric_name,
        "direction": "maximize" if spec.greater_is_better else "minimize",
    }
    save_json(study_dir / "study_config.json", base_config)

    def objective(trial: optuna.Trial) -> float:
        model_kwargs = sample_model_kwargs(trial, search_space, args)
        hparams = sample_hparams(trial, search_space, args)
        trial.set_user_attr("model_kwargs", model_kwargs)
        trial.set_user_attr("hparams", hparams)

        trial_dataset_name = f"{dataset_name}_optuna_trial_{trial.number:04d}"
        trial_dir = ensure_dir(study_dir / "trials" / f"trial_{trial.number:04d}")
        save_json(
            trial_dir / "trial_config.json",
            {
                "trial_number": trial.number,
                "dataset_name": trial_dataset_name,
                "model_kwargs": model_kwargs,
                "hparams": hparams,
                "seed": args.seed,
            },
        )

        metadata = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "study_name": args.study_name,
            "trial_number": trial.number,
            "model_kwargs": model_kwargs,
            "hparams": hparams,
            "device": str(get_device(args.device)),
        }
        metrics = train_once(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            target_columns=target_columns,
            dataset_name=trial_dataset_name,
            seed=args.seed,
            device_name=args.device,
            model_kwargs=model_kwargs,
            hparams=hparams,
            run_dir=trial_dir,
            metadata=metadata,
            epoch_callback=build_epoch_callback(trial),
            evaluate_test=False,
            task_type=spec.task_type,
            target_mean=target_mean,
            target_scale=target_scale,
        )
        return float(metrics["best_val_auc"])

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    best_config_path = save_best_config(study, study_dir, base_config)
    if best_config_path is None:
        print(f"No completed trials yet. Study artifacts are in: {study_dir}")
    else:
        print(f"Saved Optuna study artifacts to: {study_dir}")
        print(f"Saved replayable best config to: {best_config_path}")


def run_replay(args: argparse.Namespace) -> None:
    replay_path = project_path(args.replay_best)
    config = json.loads(replay_path.read_text(encoding="utf-8"))
    data_path = args.data_path or config["data_path"]
    dataset_name = args.dataset_name or config["dataset_name"]
    target_columns = list(config["target_columns"])
    smiles_column = args.smiles_column or config.get("smiles_column", "smiles")
    seed = int(config["seed"])
    device_name = args.device if args.device is not None else config.get("device")
    # Pre-refactor best_config.json files still name the removed transformer
    # knobs; sanitize them the same way dual_kd_gnn.configs does.
    model_kwargs = sanitize_model_kwargs(dict(config["model_kwargs"]))
    hparams = sanitize_hparams(dict(config["hparams"]))
    task_type = config.get("task_type", "classification")

    train_dataset, val_dataset, test_dataset = create_datasets(
        data_path=data_path,
        target_columns=target_columns,
        seed=seed,
        dual=True,
        smiles_column=smiles_column,
        task_type=task_type,
        feature_cache=build_cache(data_path, smiles_column=smiles_column),
    )
    target_mean = target_scale = None
    if task_type == "regression":
        dataframe = pd.read_csv(data_path)
        train_idx, _, _ = split_indices(
            dataframe, target_columns=target_columns, smiles_column=smiles_column,
            seed=seed, task_type=task_type,
        )
        target_mean, target_scale = target_statistics(dataframe, target_columns, train_idx)
    run_name = args.replay_run_name or f"{dataset_name}_optuna_best"
    run_dir = PROJECT_ROOT / MODEL_SLUG / "runs" / run_name
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "replay_best_config": str(replay_path.resolve()),
        "data_path": str(Path(data_path).resolve()),
        "smiles_column": smiles_column,
        "model_kwargs": model_kwargs,
        "hparams": hparams,
        "device": str(get_device(device_name)),
    }
    metrics = train_once(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        target_columns=target_columns,
        dataset_name=run_name,
        seed=seed,
        device_name=device_name,
        model_kwargs=model_kwargs,
        hparams=hparams,
        run_dir=run_dir,
        metadata=metadata,
        evaluate_test=True,
        save_weights=True,
        task_type=task_type,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    metric_label = "RMSE" if task_type == "regression" else "ROC-AUC"
    print(f"Saved replay run artifacts to: {run_dir}")
    print(f"  Best Val {metric_label}: {float(metrics['best_val_auc']):.4f}")
    print(f"  Test {metric_label}:     {float(metrics['test_roc_auc']):.4f}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_performance()
    if args.replay_best is not None:
        run_replay(args)
    else:
        run_tuning(args)


if __name__ == "__main__":
    main()
