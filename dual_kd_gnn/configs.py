"""Per-dataset run configuration for DualDistillationModel.

The original five datasets have Optuna-tuned configs under
``dual_kd_gnn/optuna/<dataset>_xkd/best_config.json`` and those are used verbatim
whenever present -- nothing here changes an existing tuned run.

The seven datasets added later (ToxCast, HIV, ESOL, FreeSolv, Lipo, CEP,
Malaria) fall back to :data:`BASE_MODEL_KWARGS` / :data:`BASE_HPARAMS` with a
size-dependent batch size and epoch budget whenever a study has not been run for
them yet. Those defaults are deliberately close to the tuned Tox21 config (the
largest tuned multitask set) rather than being separately invented.

Sizing rationale for the target machine (one A100 80GB, 100GB RAM, 8 CPU cores):
larger sets get a bigger batch so the GPU is not starved by an 8-core input
pipeline, and a smaller epoch budget so a 41k-molecule set does not cost 30x a
1.5k-molecule one. Early stopping still decides the actual length.
"""
from __future__ import annotations

import json
from pathlib import Path

from common.config import get_project_root

# Mirrors the tuned Tox21 architecture, the largest tuned multitask dataset.
BASE_MODEL_KWARGS: dict = {
    "gnn_hidden": 256,
    "gnn_layers": 3,
    "gnn_dropout": 0.45,
    "gnn_conv": "gcn",
    "fusion_dropout": 0.25,
    "ih_rank": 32,
    "ih_symmetric": False,
    "ih_proj_dim": 256,
    "ih_num_prototypes": 8,
    "ih_assignment_mode": "sparse",
    "ih_diversity_weight": 0.0005,
    # On by default now that the head is the only trained predictor: a full
    # projection would mix geometry, topology and fingerprint back together and
    # make the block decomposition unreadable.
    "ih_block_proj": True,
    "info_nce_temperature": 0.2,
}

BASE_HPARAMS: dict = {
    "batch_size": 128,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-4,
    "num_epochs": 150,
    "patience": 10,
    "gcn_pretrain_epochs": 150,
    "head_epochs": 150,
    "pretrain_lr": 6.0e-4,
    "head_lr": 3.0e-4,
    "ema_decay": 0.99,
    "ema_decay_init": 0.95,
    "distill_weight": 0.03,
    "cross_distill_weight": 0.05,
}

# Keys written by pre-refactor Optuna studies. The transformer stage is gone, so
# its architecture knobs have no target to set; tf_dropout is the one that still
# has a home (it was always the fusion-input dropout rate). Dropping these here
# rather than accepting-and-ignoring them in the model keeps the model strict:
# a typo in a hand-written config still raises instead of being swallowed.
LEGACY_MODEL_KWARGS: tuple[str, ...] = ("nhead", "tf_layers", "dim_ff")
# Pinned regardless of what a saved config says. A symmetric head is PSD, and
# with no linear term beside the quadratic form that makes the head unable to
# produce the signed, molecule-dependent output that imbalanced multi-task
# classification and standardized regression both need; it collapses to
# bias-only and cannot recover (see InteractionTensorHead's docstring for the
# measurements). Studies tuned before the head became purely quadratic chose
# ih_symmetric=True for 9 of the 12 datasets, so this has to be overridden at
# load time rather than left to each config. An explicit --ih-symmetric on the
# command line still wins: CLI flags are applied after this.
FORCED_MODEL_KWARGS: dict = {"ih_symmetric": False}
RENAMED_MODEL_KWARGS: dict[str, str] = {"tf_dropout": "fusion_dropout"}
RENAMED_HPARAMS: dict[str, str] = {
    "transformer_epochs": "head_epochs",
    "transformer_lr": "head_lr",
}


def sanitize_model_kwargs(model_kwargs: dict) -> dict:
    """Drop removed keys and apply renames, leaving everything else untouched."""
    clean = {}
    for key, value in model_kwargs.items():
        if key in LEGACY_MODEL_KWARGS:
            continue
        clean[RENAMED_MODEL_KWARGS.get(key, key)] = value
    clean.update(FORCED_MODEL_KWARGS)
    return clean


def sanitize_hparams(hparams: dict) -> dict:
    """Apply the phase-2 hyperparameter renames (transformer_* -> head_*)."""
    clean = {}
    for key, value in hparams.items():
        target = RENAMED_HPARAMS.get(key, key)
        # An explicit head_* entry always wins over the legacy alias.
        if target in clean and key in RENAMED_HPARAMS:
            continue
        clean[target] = value
    return clean

# (max molecules, batch_size, stage epochs, patience). First bucket that fits wins.
SIZE_BUCKETS: tuple[tuple[int, int, int, int], ...] = (
    (2_000, 64, 150, 15),
    (10_000, 128, 150, 12),
    (35_000, 256, 100, 10),
    (10**9, 512, 60, 8),
)


def size_profile(num_molecules: int) -> dict:
    for threshold, batch_size, epochs, patience in SIZE_BUCKETS:
        if num_molecules <= threshold:
            return {
                "batch_size": batch_size,
                "num_epochs": epochs,
                "gcn_pretrain_epochs": epochs,
                "head_epochs": epochs,
                "patience": patience,
            }
    raise AssertionError("SIZE_BUCKETS must end with a catch-all bucket")


def tuned_config_path(dataset: str) -> Path:
    return get_project_root() / "dual_kd_gnn" / "optuna" / f"{dataset}_xkd" / "best_config.json"


def has_tuned_config(dataset: str) -> bool:
    return tuned_config_path(dataset).exists()


def load_run_config(dataset: str, num_molecules: int | None = None) -> dict:
    """Return ``{'model_kwargs':..., 'hparams':..., 'config_source':...}``.

    Uses the Optuna best_config when one exists, otherwise the size-scaled
    defaults. ``config_source`` is recorded in metrics.json so a reader can tell
    tuned results from untuned ones without guessing.
    """
    path = tuned_config_path(dataset)
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        return {
            "model_kwargs": sanitize_model_kwargs(dict(config["model_kwargs"])),
            "hparams": sanitize_hparams(dict(config["hparams"])),
            "config_source": "optuna_tuned",
        }

    hparams = dict(BASE_HPARAMS)
    if num_molecules is not None:
        profile = size_profile(num_molecules)
        hparams.update(profile)
    return {
        "model_kwargs": dict(BASE_MODEL_KWARGS),
        "hparams": hparams,
        "config_source": "default_untuned",
    }
