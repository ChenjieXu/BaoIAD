"""Tests for the UFlow strict training hook."""

import torch
from mmengine.optim.scheduler import LinearLR

from baoiad.engine.hooks.uflow_strict_hook import UFlowStrictTrainHook


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs


class _Runner:
    def __init__(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=1e-3)
        self.param_schedulers = [
            LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.4,
                begin=0,
                end=2,
                by_epoch=False,
            )
        ]
        self.train_dataloader = [object()] * 7
        self.max_epochs = 200
        self.logger = _Logger()


def test_uflow_strict_hook_updates_linear_lr_total_iters():
    runner = _Runner()
    hook = UFlowStrictTrainHook()

    hook.before_train(runner)

    scheduler = runner.param_schedulers[0]
    assert scheduler.total_iters == 1400
    assert scheduler.end == 1401
