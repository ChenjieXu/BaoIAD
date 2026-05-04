"""Strict training hook for UFlow official alignment."""

from __future__ import annotations

from mmengine.hooks import Hook

from baoiad.registry import HOOKS


@HOOKS.register_module(force=True)
class UFlowStrictTrainHook(Hook):
    """Mirror the official UFlow LinearLR budget in iteration space.

    The original trainer sets the scheduler `total_iters` to
    `len(train_dataloader) * epochs`. The strict config keeps a placeholder
    LinearLR entry and this hook rewrites it once the real dataloader length is
    known.
    """

    priority = 'VERY_HIGH'

    def before_train(self, runner) -> None:
        schedulers = _flatten_param_schedulers(getattr(runner, 'param_schedulers', None))
        if not schedulers:
            return

        epoch_length = len(runner.train_dataloader)
        max_epochs = getattr(runner, 'max_epochs', None)
        if max_epochs is None and hasattr(runner, 'train_loop'):
            max_epochs = getattr(runner.train_loop, 'max_epochs', None)
        if not epoch_length or not max_epochs:
            return

        total_iters = int(epoch_length) * int(max_epochs)
        updated = 0
        for scheduler in schedulers:
            if scheduler.__class__.__name__ != 'LinearLR':
                continue
            if getattr(scheduler, 'by_epoch', True):
                continue
            begin = int(getattr(scheduler, 'begin', 0))
            scheduler.total_iters = total_iters
            scheduler.end = begin + total_iters + 1
            updated += 1

        if updated and hasattr(runner, 'logger'):
            runner.logger.info(
                'Adjusted UFlow strict LinearLR total_iters to %d (epoch_length=%d, max_epochs=%d).',
                total_iters,
                epoch_length,
                max_epochs,
            )


def _flatten_param_schedulers(param_schedulers) -> list:
    if param_schedulers is None:
        return []
    if isinstance(param_schedulers, dict):
        flattened = []
        for value in param_schedulers.values():
            flattened.extend(_flatten_param_schedulers(value))
        return flattened
    if isinstance(param_schedulers, (list, tuple)):
        flattened = []
        for value in param_schedulers:
            flattened.extend(_flatten_param_schedulers(value))
        return flattened
    return [param_schedulers]
