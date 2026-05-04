"""Official Dinomaly warmup + cosine LR scheduler."""

import math

from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin
from mmengine.optim.scheduler.param_scheduler import _ParamScheduler

from baoiad.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module(force=True)
class WarmCosineLR(LRSchedulerMixin, _ParamScheduler):
    """Match Dinomaly's absolute warmup + cosine learning-rate curve.

    Official schedule:
    - linearly warm up from ``start_warmup_value`` to the optimizer base LR
    - then cosine decay from base LR to ``final_value``
    - stepped per iteration
    """

    def __init__(
        self,
        optimizer,
        total_iters,
        final_value,
        warmup_iters=0,
        start_warmup_value=0.0,
        begin=0,
        end=None,
        last_step=-1,
        by_epoch=False,
        verbose=False,
    ):
        self.total_iters = int(total_iters)
        self.final_value = float(final_value)
        self.warmup_iters = int(warmup_iters)
        self.start_warmup_value = float(start_warmup_value)
        if self.total_iters <= 0:
            raise ValueError(f'total_iters must be positive, got {total_iters}')
        if self.warmup_iters < 0:
            raise ValueError(f'warmup_iters must be non-negative, got {warmup_iters}')
        if self.warmup_iters > self.total_iters:
            raise ValueError(
                f'warmup_iters ({self.warmup_iters}) must not exceed '
                f'total_iters ({self.total_iters})'
            )
        if end is None:
            end = begin + self.total_iters

        super().__init__(
            optimizer=optimizer,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose,
        )

    def _get_value(self):
        step = self.last_step
        values = []

        for base_value in self.base_values:
            if step >= self.total_iters:
                values.append(self.final_value)
                continue

            if self.warmup_iters > 0 and step < self.warmup_iters:
                if self.warmup_iters == 1:
                    values.append(base_value)
                else:
                    progress = step / float(self.warmup_iters - 1)
                    values.append(
                        self.start_warmup_value
                        + (base_value - self.start_warmup_value) * progress
                    )
                continue

            cosine_total = max(1, self.total_iters - self.warmup_iters)
            cosine_step = max(0, step - self.warmup_iters)
            values.append(
                self.final_value
                + 0.5 * (base_value - self.final_value)
                * (1 + math.cos(math.pi * cosine_step / cosine_total))
            )

        return values
