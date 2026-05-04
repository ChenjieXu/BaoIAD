"""Tests for ViTAD strict protocol hook."""

from types import SimpleNamespace

import torch

import baoiad  # noqa: F401
from baoiad.engine.hooks.vitad_strict_hook import ViTADStrictTrainHook


class TestViTADStrictTrainHook:
    def _make_runner(self, iter_count=0):
        net_t = torch.nn.Linear(2, 2)
        net_t.train()
        model = SimpleNamespace(net_t=net_t)
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
        return SimpleNamespace(
            model=model,
            optim_wrapper=SimpleNamespace(optimizer=optimizer),
            train_dataloader=list(range(5)),
            iter=iter_count,
        )

    def test_before_train_epoch_keeps_teacher_in_eval(self):
        hook = ViTADStrictTrainHook(decay_epochs=(80,), gamma=0.1)
        runner = self._make_runner()

        hook.before_train_epoch(runner)

        assert runner.model.net_t.training is False

    def test_before_train_iter_uses_iter_space_decay(self):
        hook = ViTADStrictTrainHook(decay_epochs=(2,), gamma=0.1)
        runner = self._make_runner(iter_count=0)
        hook.before_train(runner)

        hook.before_train_iter(runner, batch_idx=0)
        assert runner.optim_wrapper.optimizer.param_groups[0]['lr'] == 1e-4

        runner.iter = 10
        hook.before_train_iter(runner, batch_idx=0)
        assert runner.optim_wrapper.optimizer.param_groups[0]['lr'] == 1e-5

