from __future__ import annotations

import math
import os
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from common.metrics import compute_metrics, compute_regression_metrics
from common.resources import loader_workers as default_loader_workers, worker_init


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        device: torch.device,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        num_epochs: int = 150,
        patience: int = 10,
        num_classes: int = 12,
        gcn_pretrain_epochs: int | None = None,
        head_epochs: int | None = None,
        pretrain_lr: float | None = None,
        head_lr: float | None = None,
        transformer_epochs: int | None = None,   # deprecated alias for head_epochs
        transformer_lr: float | None = None,     # deprecated alias for head_lr
        ema_decay: float = 0.99,
        ema_decay_init: float | None = None,
        distill_weight: float = 0.1,
        cross_distill_weight: float = 0.0,
        epoch_callback: Callable[[dict[str, object]], None] | None = None,
        task_type: str = "classification",
        target_mean=None,
        target_scale=None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.use_amp = device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.patience = patience
        self.epoch_callback = epoch_callback
        self.loader_num_workers = default_loader_workers()
        self.loader_kwargs = self._build_loader_kwargs()

        # Regression trains on standardized targets (per-task train mean/std) but
        # reports RMSE in the original units, so the number is comparable with
        # published MoleculeNet results.
        self.task_type = task_type
        self.is_regression = task_type == "regression"
        self.metric_name = "rmse" if self.is_regression else "roc_auc"
        self.greater_is_better = not self.is_regression
        self.target_mean = self._as_tensor(target_mean, device)
        self.target_scale = self._as_tensor(target_scale, device)
        # A trailing batch of exactly one sample crashes BatchNorm in train mode
        # ("Expected more than 1 value per channel"). Drop it only in that case,
        # so every other dataset keeps seeing all of its training data.
        drop_last = len(train_dataset) % batch_size == 1 and len(train_dataset) > batch_size
        if drop_last:
            print(f"  [loader] dropping the final training batch of 1 "
                  f"(n={len(train_dataset)}, batch_size={batch_size})")
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.loader_num_workers,
            drop_last=drop_last,
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.loader_num_workers,
            **self.loader_kwargs,
        )
        self.criterion = (
            nn.MSELoss(reduction="none") if self.is_regression
            else nn.BCEWithLogitsLoss(reduction="none")
        )

        self.use_stagewise_teacher = bool(getattr(self.model, "supports_stagewise_teacher", False))
        self.gcn_pretrain_epochs = gcn_pretrain_epochs if gcn_pretrain_epochs is not None else num_epochs
        # Phase 2 used to train transformer encoders; it now trains the head only.
        # The transformer_* names are still accepted so Optuna best_config.json
        # files written before the refactor keep loading without translation.
        if head_epochs is None:
            head_epochs = transformer_epochs
        if head_lr is None:
            head_lr = transformer_lr
        self.head_epochs = head_epochs if head_epochs is not None else num_epochs
        self.pretrain_lr = pretrain_lr if pretrain_lr is not None else lr
        self.head_lr = head_lr if head_lr is not None else lr
        # Read-only aliases for older callers.
        self.transformer_epochs = self.head_epochs
        self.transformer_lr = self.head_lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.ema_decay_init = ema_decay_init
        self._current_ema_decay = ema_decay
        self.distill_weight = distill_weight
        self.cross_distill_weight = cross_distill_weight
        self._pretrain_debug_logged = False

        self.optimizer = None
        self.scheduler = None
        self.pretrain_optimizer = None
        self.pretrain_scheduler = None

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_aucs: list[float] = []
        self.val_aucs: list[float] = []
        self.pretrain_train_losses: list[float] = []
        self.pretrain_val_losses: list[float] = []
        self.pretrain_train_aucs: list[float] = []
        self.pretrain_val_aucs: list[float] = []
        self.pretrain_cls_losses: list[float] = []
        self.pretrain_distill_losses: list[float] = []
        self.pretrain_cross_distill_losses: list[float] = []
        self.pretrain_val_cls_losses: list[float] = []
        self.pretrain_val_distill_losses: list[float] = []
        self.pretrain_val_cross_distill_losses: list[float] = []
        self.best_val_metric = self._worst_metric()
        self.best_epoch = 0
        self.best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}

        if not self.use_stagewise_teacher:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self._plateau_mode(),
                factor=0.5,
                patience=5,
            )

    @staticmethod
    def _as_tensor(values, device: torch.device):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return values.to(device=device, dtype=torch.float)
        return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)

    def _worst_metric(self) -> float:
        """Starting point for best-so-far tracking: -inf for AUC, +inf for RMSE."""
        return float("-inf") if self.greater_is_better else float("inf")

    def _plateau_mode(self) -> str:
        return "max" if self.greater_is_better else "min"

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        if math.isnan(candidate):
            return False
        return candidate > incumbent if self.greater_is_better else candidate < incumbent

    @property
    def best_val_auc(self) -> float:
        """Backwards-compatible alias for the selection metric.

        Reported as 0.0 rather than +/-inf when nothing has been recorded yet, so
        callers that serialize it straight into metrics.json stay JSON-clean.
        """
        if math.isinf(self.best_val_metric):
            return 0.0
        return self.best_val_metric

    @best_val_auc.setter
    def best_val_auc(self, value: float) -> None:
        self.best_val_metric = value

    def _standardize(self, targets: torch.Tensor) -> torch.Tensor:
        if self.target_mean is None or self.target_scale is None:
            return targets
        return (targets - self.target_mean) / self.target_scale

    def _unstandardize(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.target_mean is None or self.target_scale is None:
            return outputs
        return outputs * self.target_scale + self.target_mean

    def _build_loader_kwargs(self) -> dict[str, object]:
        loader_kwargs: dict[str, object] = {
            "pin_memory": self.device.type == "cuda",
        }
        if self.loader_num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            # RAM is plentiful relative to the CPU budget, so buy deeper overlap
            # between collation and the GPU step rather than more workers.
            loader_kwargs["prefetch_factor"] = 4
            # Keep each worker single-threaded; otherwise every worker opens its
            # own OpenMP pool and the job overruns its core allocation.
            loader_kwargs["worker_init_fn"] = worker_init
        return loader_kwargs

    def _forward(self, data):
        outputs = self.model(data)
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _set_model_mode(self, train: bool = True) -> None:
        self.model.train() if train else self.model.eval()
        if hasattr(self.model, "set_teacher_eval"):
            self.model.set_teacher_eval()

    def _reshape_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets.view(-1, self.num_classes) if targets.dim() == 1 else targets

    def _target_mask(self, targets: torch.Tensor, data=None) -> torch.Tensor:
        """1.0 where a label is present.

        Classification uses the -1 sentinel written by the dataset. Regression
        cannot -- -1 is a valid target -- so it relies on the explicit y_mask the
        dataset attaches; without one, every entry counts as present.
        """
        if not self.is_regression:
            return (targets != -1).float()
        mask = getattr(data, "y_mask", None) if data is not None else None
        if mask is None:
            return torch.ones_like(targets)
        return self._reshape_targets(mask).float()

    def _compute_task_loss(self, outputs: torch.Tensor, targets: torch.Tensor, data=None):
        targets = self._reshape_targets(targets)
        mask = self._target_mask(targets, data)
        loss_targets = self._standardize(targets) if self.is_regression else targets
        loss_matrix = self.criterion(outputs, loss_targets)
        loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, targets

    def _score(self, outputs: torch.Tensor, targets: torch.Tensor, masks=None) -> float:
        """Selection metric for a full pass: ROC-AUC, or RMSE in original units."""
        if not self.is_regression:
            return float(compute_metrics(outputs, targets, self.num_classes)["roc_auc"])
        predictions = self._unstandardize(outputs.to(self.device)).cpu()
        mask = torch.cat(masks).cpu() if masks else None
        metrics = compute_regression_metrics(predictions, targets, self.num_classes, mask)
        return float(metrics["rmse"])

    def _score_batches(self, outputs: list, targets: list, masks: list) -> float:
        """Selection metric over a whole pass, or NaN when the loader was empty.

        An empty split is not a crash: scaffold splitting a very small or very
        low-diversity set can leave a fold with no molecules, and torch.cat on
        an empty list raises a message that says nothing about which split was
        empty. NaN propagates instead, and _is_better already refuses NaN, so
        the run reports the problem rather than dying mid-sweep.
        """
        if not outputs:
            print("  [warn] evaluation split is empty — reporting NaN for this pass")
            return float("nan")
        return self._score(torch.cat(outputs), torch.cat(targets), masks)

    def _maybe_auxiliary_loss(self) -> torch.Tensor | None:
        if hasattr(self.model, "auxiliary_loss"):
            aux = self.model.auxiliary_loss()
            if isinstance(aux, torch.Tensor) and aux.requires_grad:
                return aux
            if isinstance(aux, torch.Tensor) and aux.numel() == 1 and float(aux.item()) != 0.0:
                return aux
        return None

    def _maybe_step_schedule(self, stage: str, epoch_idx: int, total_epochs: int) -> None:
        if hasattr(self.model, "step_classifier_schedule"):
            self.model.step_classifier_schedule(stage, epoch_idx, total_epochs)

    def _compute_ema_decay(self, epoch_idx: int, total_epochs: int) -> float:
        final_decay = self.ema_decay
        init_decay = self.ema_decay_init
        if init_decay is None or init_decay >= final_decay or total_epochs <= 1:
            return final_decay
        progress = max(0.0, min(1.0, float(epoch_idx) / float(total_epochs - 1)))
        return final_decay - (final_decay - init_decay) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _notify_epoch(self, event: dict[str, object]) -> None:
        if self.epoch_callback is not None:
            self.epoch_callback(event)

    def _run_epoch(self, loader, train: bool = True):
        self._set_model_mode(train)
        total_loss = torch.zeros((), device=self.device)
        all_outputs, all_targets, all_masks = [], [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for data in loader:
                data = data.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self._forward(data)
                    loss, targets = self._compute_task_loss(outputs, data.y, data)
                    if train:
                        aux = self._maybe_auxiliary_loss()
                        if aux is not None:
                            loss = loss + aux
                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                total_loss += loss.detach()
                all_outputs.append(outputs.detach().float())
                all_targets.append(targets.detach())
                if self.is_regression:
                    all_masks.append(self._target_mask(targets, data).detach())

        average_loss = float(total_loss.item()) / max(len(loader), 1)
        return average_loss, self._score_batches(all_outputs, all_targets, all_masks)

    def _run_pretrain_epoch(self, loader, train: bool = True):
        self._set_model_mode(train)
        total_cls = torch.zeros((), device=self.device)
        total_distill = torch.zeros((), device=self.device)
        total_cross = torch.zeros((), device=self.device)
        total_applied = torch.zeros((), device=self.device)
        use_cross = (
            self.cross_distill_weight > 0.0
            and hasattr(self.model, "compute_cross_distill_loss")
        )
        all_outputs, all_targets, all_masks = [], [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for data in loader:
                data = data.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    stage_out = self.model.forward_gcn_pretrain(data)
                    cls_loss, targets = self._compute_task_loss(
                        stage_out["student_logits"], data.y, data
                    )
                    distill_loss = self.model.compute_distill_loss(stage_out)
                    if use_cross:
                        cross_loss = self.model.compute_cross_distill_loss(stage_out)
                    else:
                        cross_loss = stage_out["student_logits"].new_zeros(())
                    loss = (
                        cls_loss
                        + (self.distill_weight * distill_loss)
                        + (self.cross_distill_weight * cross_loss)
                    )
                    if train:
                        aux = self._maybe_auxiliary_loss()
                        if aux is not None:
                            loss = loss + aux

                if train and not self._pretrain_debug_logged:
                    debug_info = stage_out.get("debug_info", {})
                    print("    [Stage 1 Debug - first step only]")
                    print(f"      teacher_edge_dropout: {debug_info.get('teacher_edge_dropout', 0.0):.1f}")
                    print(f"      student_edge_dropout: {debug_info.get('student_edge_dropout', 0.1):.1f}")
                    print(f"      teacher_edges: {debug_info.get('teacher_num_edges')}")
                    print(f"      student_edges: {debug_info.get('student_num_edges')}")
                    print(f"      task_loss_type: {type(self.criterion).__name__}")
                    print(f"      distill_weight: {self.distill_weight:.4f}")
                    print(f"      cross_distill_weight: {self.cross_distill_weight:.4f}")
                    print(f"      raw_cls_loss: {cls_loss.item():.6f}")
                    print(f"      raw_distill_loss: {distill_loss.item():.6f}")
                    print(f"      raw_cross_distill_loss: {float(cross_loss.item()):.6f}")
                    print(f"      total_loss: {loss.item():.6f}")
                    self._pretrain_debug_logged = True

                if train:
                    self.pretrain_optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.pretrain_optimizer)
                    nn.utils.clip_grad_norm_(self.model.get_gcn_pretrain_parameters(), 1.0)
                    self.scaler.step(self.pretrain_optimizer)
                    self.scaler.update()
                    self.model.update_teachers(self._current_ema_decay)

                total_cls += cls_loss.detach()
                total_distill += distill_loss.detach()
                total_cross += cross_loss.detach()
                total_applied += loss.detach()
                all_outputs.append(stage_out["student_logits"].detach().float())
                all_targets.append(targets.detach())
                if self.is_regression:
                    all_masks.append(self._target_mask(targets, data).detach())

        average_cls = float(total_cls.item()) / max(len(loader), 1)
        average_distill = float(total_distill.item()) / max(len(loader), 1)
        average_cross = float(total_cross.item()) / max(len(loader), 1)
        average_loss = float(total_applied.item()) / max(len(loader), 1)
        return (
            average_loss,
            self._score_batches(all_outputs, all_targets, all_masks),
            average_cls,
            average_distill,
            average_cross,
        )

    def _run_head_epoch(self, loader, train: bool = True):
        self._set_model_mode(train)
        total_loss = torch.zeros((), device=self.device)
        all_outputs, all_targets, all_masks = [], [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for data in loader:
                data = data.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self._forward(data)
                    loss, targets = self._compute_task_loss(outputs, data.y, data)
                    if train:
                        aux = self._maybe_auxiliary_loss()
                        if aux is not None:
                            loss = loss + aux
                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.get_head_parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                total_loss += loss.detach()
                all_outputs.append(outputs.detach().float())
                all_targets.append(targets.detach())
                if self.is_regression:
                    all_masks.append(self._target_mask(targets, data).detach())

        average_loss = float(total_loss.item()) / max(len(loader), 1)
        return average_loss, self._score_batches(all_outputs, all_targets, all_masks)

    def _train_standard(self):
        patience_count = 0
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_auc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_auc = self._run_epoch(self.val_loader, train=False)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_aucs.append(train_auc)
            self.val_aucs.append(val_auc)
            self.scheduler.step(val_auc)
            label = self.metric_name.upper()
            print(
                f"  Epoch {epoch}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train {label}: {train_auc:.4f} | Val {label}: {val_auc:.4f}"
            )
            self._notify_epoch(
                {
                    "phase": "train",
                    "epoch": epoch,
                    "global_epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metric": train_auc,
                    "val_metric": val_auc,
                }
            )

            if self._is_better(val_auc, self.best_val_metric):
                self.best_val_metric = val_auc
                self.best_epoch = epoch
                self.best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= self.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"  Best Val {self.metric_name.upper()}: {self.best_val_metric:.4f} "
              f"(epoch {self.best_epoch})")
        return self.best_val_metric

    def _train_stagewise_teacher(self):
        task_loss_label = "MSE" if self.is_regression else "BCE"
        print(f"  [Stage 1/2] Student GNN pretraining: task {task_loss_label} "
              "+ lambda * online EMA teacher KD + cross-modal InfoNCE")
        self.model.sync_teachers()
        if hasattr(self.model, "get_gcn_pretrain_param_groups"):
            self.pretrain_optimizer = torch.optim.Adam(
                self.model.get_gcn_pretrain_param_groups(self.weight_decay),
                lr=self.pretrain_lr,
            )
        else:
            self.pretrain_optimizer = torch.optim.Adam(
                self.model.get_gcn_pretrain_parameters(),
                lr=self.pretrain_lr,
                weight_decay=self.weight_decay,
            )
        self.pretrain_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.pretrain_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        for epoch in range(1, self.gcn_pretrain_epochs + 1):
            self._current_ema_decay = self._compute_ema_decay(epoch - 1, self.gcn_pretrain_epochs)
            train_loss, train_auc, train_cls, train_dist, train_cross = self._run_pretrain_epoch(
                self.train_loader, train=True
            )
            val_loss, val_auc, val_cls, val_dist, val_cross = self._run_pretrain_epoch(
                self.val_loader, train=False
            )
            self.pretrain_train_losses.append(train_loss)
            self.pretrain_val_losses.append(val_loss)
            self.pretrain_train_aucs.append(train_auc)
            self.pretrain_val_aucs.append(val_auc)
            self.pretrain_cls_losses.append(train_cls)
            self.pretrain_distill_losses.append(train_dist)
            self.pretrain_cross_distill_losses.append(train_cross)
            self.pretrain_val_cls_losses.append(val_cls)
            self.pretrain_val_distill_losses.append(val_dist)
            self.pretrain_val_cross_distill_losses.append(val_cross)
            self.pretrain_scheduler.step(val_loss)
            metric_label = self.metric_name.upper()
            print(
                f"    GCN Epoch {epoch}/{self.gcn_pretrain_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train {task_loss_label}: {train_cls:.4f} | Val {task_loss_label}: {val_cls:.4f} | "
                f"Train KD: {train_dist:.4f} | Val KD: {val_dist:.4f} | "
                f"Train XKD: {train_cross:.4f} | Val XKD: {val_cross:.4f} | "
                f"Train {metric_label}: {train_auc:.4f} | Val {metric_label}: {val_auc:.4f} | "
                f"EMA: {self._current_ema_decay:.5f}"
            )
            self._notify_epoch(
                {
                    "phase": "stage1_gcn_kd",
                    "epoch": epoch,
                    "global_epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metric": train_auc,
                    "val_metric": val_auc,
                    "train_distill_loss": train_dist,
                    "val_distill_loss": val_dist,
                    "train_cross_distill_loss": train_cross,
                    "val_cross_distill_loss": val_cross,
                }
            )

        print("  [Stage 2/2] Frozen student GNN -> codebook head fine-tuning")
        self.optimizer = torch.optim.Adam(
            self.model.get_head_parameters(),
            lr=self.head_lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self._plateau_mode(),
            factor=0.5,
            patience=5,
        )

        patience_count = 0
        self.best_val_metric = self._worst_metric()
        self.best_epoch = 0
        self.best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}

        for epoch in range(1, self.head_epochs + 1):
            self._maybe_step_schedule("head", epoch - 1, self.head_epochs)
            train_loss, train_auc = self._run_head_epoch(self.train_loader, train=True)
            val_loss, val_auc = self._run_head_epoch(self.val_loader, train=False)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_aucs.append(train_auc)
            self.val_aucs.append(val_auc)
            self.scheduler.step(val_auc)
            print(
                f"    Head Epoch {epoch}/{self.head_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train {metric_label}: {train_auc:.4f} | Val {metric_label}: {val_auc:.4f}"
            )
            self._notify_epoch(
                {
                    "phase": "stage2_head",
                    "epoch": epoch,
                    "global_epoch": self.gcn_pretrain_epochs + epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metric": train_auc,
                    "val_metric": val_auc,
                }
            )

            if self._is_better(val_auc, self.best_val_metric):
                self.best_val_metric = val_auc
                self.best_epoch = epoch
                self.best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= self.patience:
                print(f"  Early stopping at head epoch {epoch}")
                break

        print(f"  Best Final Val {self.metric_name.upper()}: {self.best_val_metric:.4f} "
              f"(head epoch {self.best_epoch})")
        return self.best_val_metric

    def train(self):
        if self.use_stagewise_teacher:
            return self._train_stagewise_teacher()
        return self._train_standard()

    def evaluate(self, test_dataset, batch_size: int = 64):
        """Score the best checkpoint on a held-out set.

        Returns the selection metric (ROC-AUC or RMSE) as a float; use
        :meth:`evaluate_full` when the secondary regression metrics are wanted.
        """
        self.model.load_state_dict(self.best_state)
        self.model.to(self.device)
        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.loader_num_workers,
            **self.loader_kwargs,
        )
        if self.use_stagewise_teacher:
            _, test_metric = self._run_head_epoch(loader, train=False)
        else:
            _, test_metric = self._run_epoch(loader, train=False)
        return test_metric

    def evaluate_full(self, test_dataset, batch_size: int = 64) -> dict[str, float]:
        """Every metric for the task type, e.g. {'rmse':..,'mae':..,'r2':..}."""
        self.model.load_state_dict(self.best_state)
        self.model.to(self.device)
        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.loader_num_workers,
            **self.loader_kwargs,
        )
        self._set_model_mode(False)
        all_outputs, all_targets, all_masks = [], [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    if self.use_stagewise_teacher:
                        outputs = self._forward(data)
                    else:
                        outputs = self._forward(data)
                targets = self._reshape_targets(data.y)
                all_outputs.append(outputs.detach().float())
                all_targets.append(targets.detach())
                if self.is_regression:
                    all_masks.append(self._target_mask(targets, data).detach())

        if not all_outputs:
            print("  [warn] test split is empty — reporting NaN metrics")
            nan = float("nan")
            return {"rmse": nan, "mae": nan, "r2": nan} if self.is_regression else {"roc_auc": nan}
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets)
        if not self.is_regression:
            metrics = compute_metrics(outputs, targets, self.num_classes)
            return {"roc_auc": float(metrics["roc_auc"])}
        predictions = self._unstandardize(outputs).cpu()
        mask = torch.cat(all_masks).cpu() if all_masks else None
        metrics = compute_regression_metrics(predictions, targets, self.num_classes, mask)
        return {key: float(metrics[key]) for key in ("rmse", "mae", "r2")}

    def build_history_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        global_epoch = 0

        if self.use_stagewise_teacher:
            for epoch, (
                train_loss,
                val_loss,
                train_auc,
                val_auc,
                train_dist,
                val_dist,
                train_cross,
                val_cross,
            ) in enumerate(
                zip(
                    self.pretrain_train_losses,
                    self.pretrain_val_losses,
                    self.pretrain_train_aucs,
                    self.pretrain_val_aucs,
                    self.pretrain_distill_losses,
                    self.pretrain_val_distill_losses,
                    self.pretrain_cross_distill_losses,
                    self.pretrain_val_cross_distill_losses,
                ),
                start=1,
            ):
                global_epoch += 1
                rows.append(
                    {
                        "phase": "stage1_gcn_kd",
                        "epoch": epoch,
                        "global_epoch": global_epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_metric": train_auc,
                        "val_metric": val_auc,
                        "train_distill_loss": train_dist,
                        "val_distill_loss": val_dist,
                        "train_cross_distill_loss": train_cross,
                        "val_cross_distill_loss": val_cross,
                    }
                )

        for epoch, (train_loss, val_loss, train_auc, val_auc) in enumerate(
            zip(self.train_losses, self.val_losses, self.train_aucs, self.val_aucs),
            start=1,
        ):
            global_epoch += 1
            rows.append(
                {
                    "phase": "stage2_head" if self.use_stagewise_teacher else "train",
                    "epoch": epoch,
                    "global_epoch": global_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metric": train_auc,
                    "val_metric": val_auc,
                    "train_distill_loss": math.nan,
                    "val_distill_loss": math.nan,
                    "train_cross_distill_loss": math.nan,
                    "val_cross_distill_loss": math.nan,
                }
            )

        return rows
