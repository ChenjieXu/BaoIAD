"""Strict official CFlow training loop."""

from __future__ import annotations

import math

from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.runner import EpochBasedTrainLoop

from baoiad.registry import LOOPS


@LOOPS.register_module()
@MMENGINE_LOOPS.register_module()
class CFlowOfficialTrainLoop(EpochBasedTrainLoop):
    """Mirror the official ``meta_epochs x sub_epochs`` training rhythm."""

    def __init__(
        self,
        runner,
        dataloader,
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals=None,
        sub_epochs: int = 8,
        lr_decay_rate: float = 0.1,
        lr_warm: bool = True,
        lr_warm_epochs: int = 2,
        lr_cosine: bool = True,
        warmup_ratio: float = 0.1,
    ) -> None:
        super().__init__(
            runner,
            dataloader,
            max_epochs=max_epochs,
            val_begin=val_begin,
            val_interval=val_interval,
            dynamic_intervals=dynamic_intervals,
        )
        self.sub_epochs = int(sub_epochs)
        self.lr_decay_rate = float(lr_decay_rate)
        self.lr_warm = bool(lr_warm)
        self.lr_warm_epochs = int(lr_warm_epochs)
        self.lr_cosine = bool(lr_cosine)
        self.warmup_ratio = float(warmup_ratio)
        self._base_lrs: list[float] | None = None

    def _ensure_base_lrs(self) -> list[float]:
        if self._base_lrs is None:
            optimizer = self.runner.optim_wrapper.optimizer
            self._base_lrs = [
                float(group.get('initial_lr', group['lr']))
                for group in optimizer.param_groups
            ]
        return self._base_lrs

    def _compute_epoch_lr(self, base_lr: float, epoch: int) -> float:
        if self.lr_cosine:
            eta_min = base_lr * (self.lr_decay_rate ** 3)
            return eta_min + (base_lr - eta_min) * (
                1 + math.cos(math.pi * epoch / self.max_epochs)
            ) / 2

        milestones = [
            int(self.max_epochs * 0.50),
            int(self.max_epochs * 0.75),
            int(self.max_epochs * 0.90),
        ]
        steps = sum(epoch >= milestone for milestone in milestones)
        return base_lr * (self.lr_decay_rate ** steps)

    def _compute_warmup_target(self, base_lr: float) -> float:
        if not self.lr_cosine:
            return base_lr
        eta_min = base_lr * (self.lr_decay_rate ** 3)
        return eta_min + (base_lr - eta_min) * (
            1 + math.cos(math.pi * self.lr_warm_epochs / self.max_epochs)
        ) / 2

    def _set_lrs(self, epoch: int, batch_id: int, total_batches: int) -> None:
        optimizer = self.runner.optim_wrapper.optimizer
        base_lrs = self._ensure_base_lrs()

        for base_lr, group in zip(base_lrs, optimizer.param_groups):
            lr = self._compute_epoch_lr(base_lr, epoch)
            if self.lr_warm and epoch < self.lr_warm_epochs and total_batches > 0:
                progress = (batch_id + epoch * total_batches) / (
                    self.lr_warm_epochs * total_batches
                )
                warmup_from = base_lr * self.warmup_ratio
                warmup_to = self._compute_warmup_target(base_lr)
                lr = warmup_from + progress * (warmup_to - warmup_from)
            group['lr'] = lr

    def run_epoch(self) -> None:
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        num_batches = len(self.dataloader)
        if num_batches == 0:
            self.runner.call_hook('after_train_epoch')
            self._epoch += 1
            return

        total_batches = num_batches * self.sub_epochs
        iterator = iter(self.dataloader)

        for sub_epoch in range(self.sub_epochs):
            for batch_idx in range(num_batches):
                iter_idx = sub_epoch * num_batches + batch_idx
                self._set_lrs(self._epoch, iter_idx, total_batches)
                try:
                    data_batch = next(iterator)
                except StopIteration:
                    iterator = iter(self.dataloader)
                    data_batch = next(iterator)
                self.run_iter(iter_idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
