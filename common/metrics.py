from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_roc_auc(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    num_classes: int,
) -> tuple[float, list[float]]:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    aucs: list[float] = []
    for class_idx in range(num_classes):
        valid_mask = y_true[:, class_idx] != -1
        if int(valid_mask.sum()) == 0:
            continue
        true_values = y_true[valid_mask, class_idx]
        pred_values = y_pred[valid_mask, class_idx]
        if len(np.unique(true_values)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(true_values, pred_values)))
        except ValueError:
            continue

    return (float(np.mean(aucs)) if aucs else 0.0, aucs)


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, float | list[float]]:
    probabilities = torch.sigmoid(outputs)
    mean_auc, per_task_aucs = compute_roc_auc(targets, probabilities, num_classes)
    return {
        "roc_auc": mean_auc,
        "roc_auc_per_task": per_task_aucs,
    }


def compute_regression_metrics(
    outputs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_tasks: int,
    mask: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float | list[float]]:
    """RMSE / MAE / R^2 averaged over tasks, ignoring masked-out entries.

    ``outputs`` and ``targets`` must already be in the dataset's original units;
    the trainer standardizes targets for the loss and un-standardizes before
    calling this, so the reported RMSE is directly comparable with published
    MoleculeNet numbers.

    Unlike the classification path there is no -1 sentinel: -1 is a perfectly
    ordinary regression target (ESOL log-solubility is mostly negative), so
    missing labels arrive as NaN or through the explicit ``mask``.
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    rmses: list[float] = []
    maes: list[float] = []
    r2s: list[float] = []
    for task_idx in range(num_tasks):
        true_values = targets[:, task_idx]
        pred_values = outputs[:, task_idx]
        valid = ~np.isnan(true_values)
        if mask is not None:
            valid &= mask[:, task_idx] > 0
        if int(valid.sum()) == 0:
            continue
        true_values = true_values[valid].astype(np.float64)
        pred_values = np.nan_to_num(pred_values[valid].astype(np.float64))
        errors = pred_values - true_values
        rmses.append(float(np.sqrt(np.mean(errors ** 2))))
        maes.append(float(np.mean(np.abs(errors))))
        variance = float(np.var(true_values))
        r2s.append(float(1.0 - np.mean(errors ** 2) / variance) if variance > 0 else float("nan"))

    return {
        "rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "r2": float(np.nanmean(r2s)) if r2s else float("nan"),
        "rmse_per_task": rmses,
        "mae_per_task": maes,
    }
