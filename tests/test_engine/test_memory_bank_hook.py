"""Tests for MemoryBankHook."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import baoiad  # noqa: F401
import torch
import torch.nn as nn

from baoiad.engine.hooks.memory_bank_hook import MemoryBankHook


class TestMemoryBankHook:
    def _make_runner(self):
        runner = SimpleNamespace()
        runner.model = SimpleNamespace(build_memory_bank=MagicMock())
        runner.logger = MagicMock()
        runner.train_dataloader = MagicMock()
        runner.val_dataloader = MagicMock()
        runner.epoch = 0
        runner.max_epochs = 1
        return runner

    def test_hook_triggers_build(self):
        hook = MemoryBankHook()
        runner = self._make_runner()
        hook.after_train_epoch(runner)
        runner.model.build_memory_bank.assert_called_once()
        assert hook._built

    def test_hook_rebuilds_only_after_finalization(self):
        hook = MemoryBankHook()
        runner = self._make_runner()
        runner.max_epochs = 2
        hook.after_train_epoch(runner)
        runner.epoch = 1
        hook.after_train_epoch(runner)
        assert runner.model.build_memory_bank.call_count == 2
        hook.after_train(runner)
        assert runner.model.build_memory_bank.call_count == 2

    def test_before_val_triggers_build(self):
        hook = MemoryBankHook()
        runner = self._make_runner()
        if hasattr(hook, 'before_val_epoch'):
            hook.before_val_epoch(runner)
            runner.model.build_memory_bank.assert_called_once()

    def test_before_test_triggers_build(self):
        hook = MemoryBankHook()
        runner = self._make_runner()
        if hasattr(hook, 'before_test_epoch'):
            hook.before_test_epoch(runner)
            runner.model.build_memory_bank.assert_called_once()

    def test_after_train_triggers_build(self):
        hook = MemoryBankHook()
        runner = self._make_runner()
        if hasattr(hook, 'after_train'):
            hook.after_train(runner)
            runner.model.build_memory_bank.assert_called_once()

    def test_after_train_refits_when_model_requests_it(self):
        hook = MemoryBankHook()
        runner = self._make_runner()
        runner.model.refit_after_train = True
        hook.after_train_epoch(runner)
        hook.after_train(runner)
        assert runner.model.build_memory_bank.call_count == 2

    def test_no_memory_bank_model(self):
        """Model without build_memory_bank should not raise."""
        hook = MemoryBankHook()
        runner = SimpleNamespace(
            model=SimpleNamespace(),
            logger=MagicMock(),
            train_dataloader=MagicMock(),
            val_dataloader=MagicMock(),
            epoch=0,
            max_epochs=1,
        )
        hook.after_train_epoch(runner)
        # Should not raise, just skip

    def test_head_build_memory_bank(self):
        """Hook should try model.head.build_memory_bank as fallback."""
        hook = MemoryBankHook()
        runner = SimpleNamespace(
            model=SimpleNamespace(head=SimpleNamespace(build_memory_bank=MagicMock())),
            logger=MagicMock(),
            train_dataloader=MagicMock(),
            val_dataloader=MagicMock(),
            epoch=0,
            max_epochs=1,
        )
        hook.after_train_epoch(runner)
        runner.model.head.build_memory_bank.assert_called_once()

    def test_template_build_uses_train_dataloader_when_requested(self):
        class _TemplateModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
                self.template_dataloader_split = 'train'
                self.build_template_from_dataloader = MagicMock()

        hook = MemoryBankHook()
        model = _TemplateModel()
        train_loader = MagicMock(name='train_loader')
        val_loader = MagicMock(name='val_loader')
        runner = SimpleNamespace(
            model=model,
            logger=MagicMock(),
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epoch=0,
            max_epochs=1,
        )

        hook.after_train_epoch(runner)

        model.build_template_from_dataloader.assert_called_once()
        args, _ = model.build_template_from_dataloader.call_args
        assert args[0] is train_loader
        assert str(args[1]) == 'cpu'

    def test_pre_train_setup_can_mark_memory_bank_as_ready(self):
        hook = MemoryBankHook()
        model = SimpleNamespace(
            pre_train_setup=MagicMock(),
            pre_train_setup_builds_memory_bank=True,
            build_memory_bank=MagicMock(),
        )
        runner = SimpleNamespace(
            model=model,
            logger=MagicMock(),
            train_dataloader=MagicMock(),
            val_dataloader=MagicMock(),
            epoch=0,
            max_epochs=1,
        )

        hook.before_train(runner)
        assert hook._built is True

        hook.before_val_epoch(runner)
        model.build_memory_bank.assert_not_called()

    def test_last_epoch_does_not_force_refit_when_pretrain_setup_owns_bank(self):
        hook = MemoryBankHook()
        runner = self._make_runner()
        runner.model.pre_train_setup_builds_memory_bank = True
        hook._built = True

        hook.after_train_epoch(runner)

        runner.model.build_memory_bank.assert_not_called()
