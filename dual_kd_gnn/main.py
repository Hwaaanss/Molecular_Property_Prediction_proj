from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ModelSpec, project_path
from common.runner import run_from_cli

from dual_kd_gnn.configs import sanitize_hparams, sanitize_model_kwargs
from dual_kd_gnn.model import DualDistillationModel


MODEL_KWARG_NAMES = [
    "gnn_hidden",
    "gnn_layers",
    "gnn_dropout",
    "gnn_conv",
    "fusion_dropout",
    "ih_rank",
    "ih_symmetric",
    "ih_proj_dim",
    "ih_num_prototypes",
    "ih_assignment_mode",
    "ih_tau_init",
    "ih_tau_final",
    "ih_diversity_weight",
    "ih_codebook_init",
    "ih_topk",
    "ih_block_proj",
    "info_nce_temperature",
    "zero_phys_branch",
    "use_fingerprint",
    "fp_input_dim",
    "fp_dim",
    "fp_dropout",
    "fp_stage",
    "zero_fp_branch",
]
HPARAM_NAMES = [
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


def load_best_config(args) -> dict[str, object]:
    if args.best_config is None:
        return {}
    if not hasattr(args, "_best_config_data"):
        args._best_config_data = json.loads(
            project_path(args.best_config).read_text(encoding="utf-8"))
    return args._best_config_data


def add_dual_model_arguments(parser) -> None:
    parser.add_argument(
        "--best-config",
        type=Path,
        default=None,
        help="Path to an Optuna best_config.json containing model_kwargs and hparams.",
    )
    parser.add_argument("--gnn-hidden", type=int, default=None)
    parser.add_argument("--gnn-layers", type=int, default=None)
    parser.add_argument("--gnn-dropout", type=float, default=None)
    parser.add_argument(
        "--gnn-conv",
        choices=["gcn", "gine", "gat"],
        default=None,
        help=(
            "Message-passing convolution for both view encoders. 'gcn' (default) collapses the "
            "bond features to one scalar gate per edge; 'gine' consumes them as a vector, so the "
            "bond length reaches the geometric branch at full width; 'gat' weights neighbours by "
            "attention with the bond features in the attention logit. Changes the state_dict, so "
            "runs are not comparable across this flag."
        ),
    )
    parser.add_argument("--fusion-dropout", type=float, default=None,
                        help="Dropout on the per-view node features before fusion (was --tf-dropout).")
    parser.add_argument("--ih-rank", type=int, default=None)
    ih_symmetric = parser.add_mutually_exclusive_group()
    ih_symmetric.add_argument("--ih-symmetric", dest="ih_symmetric", action="store_true", default=None)
    ih_symmetric.add_argument("--ih-asymmetric", dest="ih_symmetric", action="store_false")
    parser.add_argument("--ih-proj-dim", type=int, default=None)
    parser.add_argument(
        "--ih-num-prototypes",
        type=int,
        default=None,
        help="Number of shared codebook prototypes (M). 0 disables the codebook and uses per-class U_k.",
    )
    parser.add_argument(
        "--ih-assignment-mode",
        choices=["hard", "soft", "sparse"],
        default=None,
        help="Codebook assignment routing: hard (Gumbel-STE), soft (softmax), or sparse (top-k softmax).",
    )
    parser.add_argument("--ih-tau-init", type=float, default=None, help="Initial Gumbel/softmax temperature.")
    parser.add_argument("--ih-tau-final", type=float, default=None, help="Final Gumbel/softmax temperature after annealing.")
    parser.add_argument("--ih-diversity-weight", type=float, default=None, help="Weight of the codebook diversity regularizer.")
    parser.add_argument(
        "--ih-codebook-init",
        choices=["orthogonal", "random"],
        default=None,
        help="Codebook initialization scheme.",
    )
    parser.add_argument("--ih-topk", type=int, default=None, help="Top-k for sparse assignment mode.")
    parser.add_argument(
        "--info-nce-temperature",
        type=float,
        default=None,
        help="Temperature for cross-modal InfoNCE distillation (graph-level CLIP-style).",
    )
    # default=None, not False: these four used to default to False, which is not
    # None, so they passed the "was it given?" filter in collect_dual_model_kwargs
    # and silently overwrote whatever --best-config had set. A saved config with
    # use_fingerprint=true / ih_block_proj=true came back with both switched off.
    # None means "not given"; each flag now has an explicit off switch.
    block_proj = parser.add_mutually_exclusive_group()
    block_proj.add_argument(
        "--ih-block-proj",
        dest="ih_block_proj",
        action="store_true",
        default=None,
        help=(
            "Make the head's projection block-diagonal (one Linear per fusion block) so the "
            "geometry/topology/fingerprint partition survives it. Unset by default: the model "
            "default (off) applies unless --best-config supplies one."
        ),
    )
    block_proj.add_argument(
        "--no-ih-block-proj", dest="ih_block_proj", action="store_false",
        help="Force a single mixing projection, overriding --best-config.",
    )
    phys_branch = parser.add_mutually_exclusive_group()
    phys_branch.add_argument(
        "--no-phys-branch",
        dest="zero_phys_branch",
        action="store_true",
        default=None,
        help="Ablation A1: zero out physical node features (x_phys=0), disabling the 3D physical branch.",
    )
    phys_branch.add_argument(
        "--phys-branch", dest="zero_phys_branch", action="store_false",
        help="Force the 3D branch on, overriding --best-config.",
    )
    fingerprint = parser.add_mutually_exclusive_group()
    fingerprint.add_argument(
        "--use-fingerprint",
        dest="use_fingerprint",
        action="store_true",
        default=None,
        help="Enable the molecule-level fingerprint branch as a third fusion block.",
    )
    fingerprint.add_argument(
        "--no-fingerprint", dest="use_fingerprint", action="store_false",
        help="Remove the fingerprint branch, overriding --best-config.",
    )
    fp_branch = parser.add_mutually_exclusive_group()
    fp_branch.add_argument(
        "--no-fp-branch",
        dest="zero_fp_branch",
        action="store_true",
        default=None,
        help="Ablation: zero out the fingerprint input (fp=0) while keeping the branch wired up.",
    )
    fp_branch.add_argument(
        "--fp-branch", dest="zero_fp_branch", action="store_false",
        help="Force real fingerprint input, overriding --best-config.",
    )
    parser.add_argument("--fp-dim", type=int, default=None,
                        help="Width of the fingerprint block after its encoder (default 64).")
    parser.add_argument("--fp-input-dim", type=int, default=None,
                        help="Raw fingerprint width. Default: common.data.DEFAULT_FINGERPRINT_DIM.")
    parser.add_argument("--fp-dropout", type=float, default=None,
                        help="Dropout inside the fingerprint encoder (default 0.2).")
    parser.add_argument(
        "--fp-stage",
        choices=["head", "stage2", "both"],
        default=None,
        help=(
            "When the fingerprint branch trains. 'head' (default; 'stage2' is the old spelling of "
            "the same thing) keeps it out of encoder pretraining, so the task loss there keeps "
            "pushing the GNN encoders; 'both' opens it from phase 1 and is an ablation only."
        ),
    )


def collect_dual_model_kwargs(args) -> dict[str, object]:
    config = load_best_config(args)
    # sanitize_model_kwargs drops the removed transformer knobs, renames
    # tf_dropout -> fusion_dropout, and re-pins ih_symmetric=False. Without it a
    # pre-refactor <dataset>_xkd config (9 of 12 carry ih_symmetric=true) builds
    # a PSD head, which collapses to bias-only and cannot recover -- see
    # InteractionTensorHead's docstring for the measurements.
    saved = sanitize_model_kwargs(dict(config.get("model_kwargs", {}))) if config else {}
    model_kwargs = {
        name: value
        for name, value in saved.items()
        if name in MODEL_KWARG_NAMES
    }
    model_kwargs.update({
        name: value
        for name in MODEL_KWARG_NAMES
        if (value := getattr(args, name)) is not None
    })
    return model_kwargs


def collect_dual_hparam_overrides(args) -> dict[str, object]:
    config = load_best_config(args)
    # transformer_epochs -> head_epochs, transformer_lr -> head_lr. Without the
    # rename those keys fail the HPARAM_NAMES filter and the phase-2 schedule of
    # every pre-refactor config is silently dropped back to the defaults.
    saved = sanitize_hparams(dict(config.get("hparams", {}))) if config else {}
    return {
        name: value
        for name, value in saved.items()
        if name in HPARAM_NAMES
    }


MODEL_SPEC = ModelSpec(
    name="dual_distillation",
    slug="dual_kd_gnn",
    uses_dual_features=True,
    builder=lambda num_classes, **model_kwargs: DualDistillationModel(
        num_classes=num_classes, **model_kwargs
    ),
    default_hparams={
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "num_epochs": 150,
        "patience": 10,
        "gcn_pretrain_epochs": 150,
        "head_epochs": 150,
        "pretrain_lr": 1e-3,
        "head_lr": 1e-3,
        "ema_decay": 0.99,
        "ema_decay_init": None,
        "distill_weight": 0.05,
        "cross_distill_weight": 0.05,
    },
    add_model_arguments=add_dual_model_arguments,
    collect_model_kwargs=collect_dual_model_kwargs,
    collect_hparam_overrides=collect_dual_hparam_overrides,
    notes="Dual-branch GNN (GCN/GINE/GAT) trained in two phases: phase 1 pretrains both view encoders with the task loss, EMA self-distillation MSE and graph-level cross-modal InfoNCE (CLIP-style, asymmetric predictor against the opposite-branch EMA teacher pool); phase 2 freezes them and fine-tunes the codebook interaction head (plus the optional fingerprint block). Optional cosine ramp on EMA momentum (ema_decay_init -> ema_decay) over phase 1.",
)


def main() -> None:
    run_from_cli(MODEL_SPEC, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
