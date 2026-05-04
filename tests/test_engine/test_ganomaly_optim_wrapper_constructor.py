"""Tests for the strict GANomaly optim-wrapper constructor."""

import torch.nn as nn
from mmengine.optim import OptimWrapperDict

import baoiad  # noqa: F401
from baoiad.engine.optimizers.ganomaly_optim_wrapper_constructor import GanomalyOptimWrapperConstructor


class _ToyGanomalyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        self.discriminator = nn.Linear(8, 1)


def test_ganomaly_optim_wrapper_constructor_builds_split_wrappers():
    constructor = GanomalyOptimWrapperConstructor(
        dict(
            generator=dict(optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999), weight_decay=0)),
            discriminator=dict(optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999), weight_decay=0)),
        )
    )

    wrappers = constructor(_ToyGanomalyModel())

    assert isinstance(wrappers, OptimWrapperDict)
    assert wrappers['generator'].optimizer.__class__.__name__ == 'Adam'
    assert wrappers['discriminator'].optimizer.__class__.__name__ == 'Adam'
