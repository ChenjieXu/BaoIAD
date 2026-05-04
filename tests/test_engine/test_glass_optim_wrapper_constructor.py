"""Tests for the strict GLASS optim-wrapper constructor."""

import torch.nn as nn
from mmengine.optim import OptimWrapperDict

import baoiad  # noqa: F401
from baoiad.engine.optimizers.glass_optim_wrapper_constructor import GLASSOptimWrapperConstructor


class _ToyGLASSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)
        self.discriminator = nn.Linear(8, 1)


def test_glass_optim_wrapper_constructor_builds_split_wrappers():
    constructor = GLASSOptimWrapperConstructor(
        dict(
            projection=dict(optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-5)),
            discriminator=dict(optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-5)),
        )
    )

    wrappers = constructor(_ToyGLASSModel())

    assert isinstance(wrappers, OptimWrapperDict)
    assert wrappers['projection'].optimizer.__class__.__name__ == 'Adam'
    assert wrappers['discriminator'].optimizer.__class__.__name__ == 'AdamW'
