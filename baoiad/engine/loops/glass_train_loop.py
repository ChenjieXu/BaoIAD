"""Strict GLASS training loop."""

from __future__ import annotations

from typing import Any

import torch
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.runner import EpochBasedTrainLoop

from baoiad.registry import LOOPS


def _batch_size_from_inputs(inputs: Any) -> int:
    if torch.is_tensor(inputs):
        return int(inputs.shape[0]) if inputs.ndim > 0 else 1
    if isinstance(inputs, (list, tuple)):
        return len(inputs)
    raise TypeError(f'Unsupported inputs container for GLASS loop: {type(inputs)!r}')


@LOOPS.register_module(force=True)
@MMENGINE_LOOPS.register_module(force=True)
class GLASSTrainLoop(EpochBasedTrainLoop):
    """Epoch-based loop that mirrors the official GLASS training rhythm.

    Each epoch first recomputes the feature-space center over the full
    training loader, then performs the actual optimization pass and stops the
    epoch once the configured sample budget has been consumed.
    """

    def run_epoch(self) -> None:
        self.runner.call_hook('before_train_epoch')

        model = self.runner.model
        inner_model = model.module if hasattr(model, 'module') else model
        if not hasattr(inner_model, 'prepare_strict_epoch'):
            raise AttributeError(
                f'{inner_model.__class__.__name__} does not implement prepare_strict_epoch() '
                'required by GLASSTrainLoop.'
            )

        inner_model.prepare_strict_epoch(self.dataloader)

        self.runner.model.train()
        sample_budget = getattr(inner_model, 'limit', None)
        sample_count = 0

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
            if sample_budget is None or sample_budget <= 0:
                continue
            sample_count += _batch_size_from_inputs(data_batch['inputs'])
            if sample_count > int(sample_budget):
                break

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
