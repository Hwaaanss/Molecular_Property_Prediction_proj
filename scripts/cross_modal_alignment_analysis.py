"""Cross-modal alignment trajectory analysis (Figure 6).

Trains Stage 1 only of the dual-branch model on Tox21 under two conditions —
full (cross_distill_weight as tuned) and A2 (cross_distill_weight=0) — and
logs the mean cosine similarity between chemical-branch and physical-branch
graph-pooled representations on the validation set after every epoch.

Outputs to results/artifacts/alignment/:
  alignment_log.csv     two conditions x epochs, with cosine and val AUC
  figure6_alignment.png plot of cosine similarity vs Stage-1 epoch

Run on the server where data + dependencies are available:
  python scripts/cross_modal_alignment_analysis.py
  python scripts/cross_modal_alignment_analysis.py --dataset bbbp --epochs 40

Default dataset is tox21 because (i) its 12 nuclear-receptor / stress-response
tasks make it the most chemically interesting multitask setting in the suite
and (ii) its full-model + A2 ablation entries already exist in
ablation/ablation_summary.csv, so the cosine trajectory connects to the
ablation table directly.

This is a separate Stage 1 run rather than a re-use of the ablation runs
because the existing ablation pipeline did not save model weights and did
not log per-epoch cosine similarity.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data import create_datasets  # noqa: E402
from common.datasets import DATASETS, resolve_target_columns  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402
from common.plotting import save_figure  # noqa: E402
from dual_kd_gnn.model import DualDistillationModel  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "artifacts" / "alignment"


def load_best_config(dataset: str) -> dict:
    path = PROJECT_ROOT / "dual_kd_gnn" / "optuna" / f"{dataset}_xkd" / "best_config.json"
    if not path.exists():
        raise FileNotFoundError(f"best_config.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


from torch_geometric.nn import global_mean_pool  # noqa: E402


@torch.no_grad()
def validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float]:
    """Return (mean cosine similarity, validation ROC-AUC).

    Cosine is computed on graph-pooled chemical and physical student
    representations after the phase-1 GNN forward, i.e. on exactly the vectors
    the cross-modal InfoNCE compares. Pooling matches forward_gcn_pretrain's.
    """
    model.eval()
    if hasattr(model, "set_teacher_eval"):
        model.set_teacher_eval()

    cosines: list[float] = []
    all_outputs, all_targets = [], []

    for data in loader:
        data = data.to(device, non_blocking=True)
        stage_out = model.forward_gcn_pretrain(data)

        # Per-graph cosine between paired (chem, phys) student graph pools.
        # student_chem_seq / student_phys_seq are node-level [N, H] tensors.
        student_c_graph = global_mean_pool(stage_out["student_chem_seq"], stage_out["batch"])
        student_p_graph = global_mean_pool(stage_out["student_phys_seq"], stage_out["batch"])
        cos = torch.nn.functional.cosine_similarity(student_c_graph, student_p_graph, dim=-1, eps=1e-8)
        cosines.append(cos.detach().cpu().float().numpy())

        all_outputs.append(stage_out["student_logits"].detach().float())
        targets = data.y
        if targets.dim() == 1:
            targets = targets.view(-1, num_classes)
        all_targets.append(targets.detach())

    mean_cos = float(np.concatenate(cosines).mean()) if cosines else float("nan")
    metrics = compute_metrics(torch.cat(all_outputs), torch.cat(all_targets), num_classes)
    return mean_cos, float(metrics["roc_auc"])


def train_one_condition(
    *,
    condition_name: str,
    dataset: str,
    cross_distill_weight: float,
    best_config: dict,
    seed: int,
    epochs: int,
    device: torch.device,
) -> pd.DataFrame:
    print(f"\n=== Condition: {condition_name} (cross_distill_weight={cross_distill_weight}) ===")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    spec = DATASETS[dataset]
    data_path = str(spec.data_path())
    target_columns = (
        list(spec.target_columns) if spec.target_columns else resolve_target_columns(spec, data_path)
    )
    num_classes = len(target_columns)

    train_ds, val_ds, _ = create_datasets(
        data_path=data_path,
        target_columns=target_columns,
        seed=seed,
        dual=True,
        smiles_column=spec.smiles_column,
    )

    model_kwargs = dict(best_config["model_kwargs"])
    hparams = dict(best_config["hparams"])
    batch_size = int(hparams.get("batch_size", 64))
    pretrain_lr = float(hparams.get("pretrain_lr", 1e-3))
    weight_decay = float(hparams.get("weight_decay", 1e-3))
    distill_weight = float(hparams.get("distill_weight", 0.05))
    ema_decay = float(hparams.get("ema_decay", 0.99))
    ema_decay_init = hparams.get("ema_decay_init", None)

    model = DualDistillationModel(num_classes=num_classes, **model_kwargs).to(device)
    model.sync_teachers()

    if hasattr(model, "get_gcn_pretrain_param_groups"):
        optimizer = torch.optim.Adam(
            model.get_gcn_pretrain_param_groups(weight_decay),
            lr=pretrain_lr,
        )
    else:
        optimizer = torch.optim.Adam(
            model.get_gcn_pretrain_parameters(),
            lr=pretrain_lr,
            weight_decay=weight_decay,
        )

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        model.set_teacher_eval()
        if ema_decay_init is None or ema_decay_init >= ema_decay or epochs <= 1:
            current_ema = ema_decay
        else:
            # cosine warmup, matching Trainer._compute_ema_decay
            import math
            progress = max(0.0, min(1.0, (epoch - 1) / max(1, epochs - 1)))
            current_ema = ema_decay - (ema_decay - ema_decay_init) * 0.5 * (1.0 + math.cos(math.pi * progress))

        epoch_loss = 0.0
        for data in train_loader:
            data = data.to(device, non_blocking=True)
            stage_out = model.forward_gcn_pretrain(data)

            targets = data.y
            if targets.dim() == 1:
                targets = targets.view(-1, num_classes)
            loss_matrix = criterion(stage_out["student_logits"], targets)
            mask = (targets != -1).float()
            cls_loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1.0)

            distill_loss = model.compute_distill_loss(stage_out)
            if cross_distill_weight > 0.0:
                cross_loss = model.compute_cross_distill_loss(stage_out)
            else:
                cross_loss = stage_out["student_logits"].new_zeros(())

            loss = cls_loss + distill_weight * distill_loss + cross_distill_weight * cross_loss
            if hasattr(model, "auxiliary_loss"):
                aux = model.auxiliary_loss()
                if isinstance(aux, torch.Tensor) and aux.requires_grad:
                    loss = loss + aux

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_teachers(current_ema)

            epoch_loss += float(loss.item())

        epoch_loss /= max(1, len(train_loader))
        val_cos, val_auc = validation_metrics(model, val_loader, device, num_classes)
        rows.append(
            {
                "condition": condition_name,
                "epoch": epoch,
                "train_loss": round(epoch_loss, 6),
                "val_cosine_chem_phys": round(val_cos, 6),
                "val_roc_auc": round(val_auc, 6),
                "ema_decay_used": round(current_ema, 6),
            }
        )
        print(
            f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss:.4f}  "
            f"val_cos={val_cos:+.4f}  val_auc={val_auc:.4f}  ema={current_ema:.5f}"
        )

    return pd.DataFrame(rows)


def plot_figure6(df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for condition, group in df.groupby("condition", sort=False):
        ax.plot(group["epoch"], group["val_cosine_chem_phys"], marker="o", markersize=3,
                linewidth=1.5, label=condition)
    ax.axhline(0.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Stage 1 epoch")
    ax.set_ylabel(r"mean $\cos(\mathbf{z}^{\mathrm{chem}}, \mathbf{z}^{\mathrm{phys}})$ on val")
    ax.set_title(f"Cross-modal representation alignment ({dataset})")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    save_figure(fig, out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tox21", choices=list(DATASETS.keys()))
    parser.add_argument("--epochs", type=int, default=50,
                        help="Stage-1 epochs per condition. 50 is enough to reveal the plateau.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda / cpu / mps; default auto")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    best_config = load_best_config(args.dataset)
    tuned_cross_weight = float(best_config["hparams"].get("cross_distill_weight", 0.05))
    print(f"Tuned cross_distill_weight for {args.dataset}: {tuned_cross_weight}")

    start = time.time()
    df_full = train_one_condition(
        condition_name="full",
        dataset=args.dataset,
        cross_distill_weight=tuned_cross_weight,
        best_config=best_config,
        seed=args.seed,
        epochs=args.epochs,
        device=device,
    )
    df_a2 = train_one_condition(
        condition_name="a2_no_infonce",
        dataset=args.dataset,
        cross_distill_weight=0.0,
        best_config=best_config,
        seed=args.seed,
        epochs=args.epochs,
        device=device,
    )
    df = pd.concat([df_full, df_a2], ignore_index=True)

    csv_path = OUT_DIR / "alignment_log.csv"
    fig_path = OUT_DIR / "figure6_alignment.png"
    df.to_csv(csv_path, index=False)
    plot_figure6(df, args.dataset, fig_path)

    elapsed = time.time() - start
    print(f"\nSaved CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved fig: {fig_path.relative_to(PROJECT_ROOT)}")
    print(f"Total time: {elapsed/60:.1f} min")

    print("\nFinal-epoch cosine summary:")
    print(df.groupby("condition").tail(1)[["condition", "epoch", "val_cosine_chem_phys", "val_roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
