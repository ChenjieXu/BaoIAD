"""Strict training hook for ViTAD alignment."""

from __future__ import annotations

from typing import Iterable, Sequence

from mmengine.hooks import Hook

from baoiad.registry import HOOKS


def _iter_decay_factor(current_iter: int, decay_iters: Sequence[int], gamma: float) -> float:
    factor = 1.0
    for decay_iter in decay_iters:
        if current_iter >= decay_iter:
            factor *= gamma
    return factor


@HOOKS.register_module()
class ViTADStrictTrainHook(Hook):
    """Mirror ADer's ViTAD training protocol where it matters.

    The reference trainer applies its step scheduler in iteration space before
    each optimization step. For ViTAD's default schedule this means keeping the
    base LR until epoch 81 starts, then decaying it by ``gamma``. The hook also
    reasserts that the teacher stays in eval mode during training.
    """

    priority = 'VERY_HIGH'

    def __init__(self, decay_epochs: Iterable[int] = (80,), gamma: float = 0.1):
        self.decay_epochs = tuple(int(epoch) for epoch in decay_epochs)
        self.gamma = float(gamma)
        self._base_lrs: list[float] = []
        self._decay_iters: tuple[int, ...] = ()

    def before_train(self, runner) -> None:
        optimizer = runner.optim_wrapper.optimizer
        self._base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        iters_per_epoch = len(runner.train_dataloader)
        self._decay_iters = tuple(epoch * iters_per_epoch for epoch in self.decay_epochs)

    def before_train_epoch(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if hasattr(model, 'net_t'):
            model.net_t.eval()

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        del batch_idx, data_batch
        if not self._base_lrs:
            return
        factor = _iter_decay_factor(runner.iter, self._decay_iters, self.gamma)
        for base_lr, group in zip(self._base_lrs, runner.optim_wrapper.optimizer.param_groups):
            group['lr'] = base_lr * factor
