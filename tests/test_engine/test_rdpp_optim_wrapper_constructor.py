"""Tests for the strict RD++ optim-wrapper constructor."""

import torch.nn as nn
from mmengine.optim import OptimWrapperDict

import baoiad  # noqa: F401
from baoiad.engine.optimizers.rdpp_optim_wrapper_constructor import RDPPOptimWrapperConstructor


class _ToyRDPPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_layer = nn.Linear(8, 8)
        self.ocbe = nn.Linear(8, 8)
        self.student = nn.Linear(8, 8)


def test_rdpp_optim_wrapper_constructor_builds_split_wrappers():
    constructor = RDPPOptimWrapperConstructor(
        dict(
            projection=dict(optimizer=dict(type='Adam', lr=1e-3, betas=(0.5, 0.999))),
            distillation=dict(optimizer=dict(type='Adam', lr=5e-3, betas=(0.5, 0.999))),
        )
    )

    wrappers = constructor(_ToyRDPPModel())

    assert isinstance(wrappers, OptimWrapperDict)
    assert wrappers['projection'].optimizer.__class__.__name__ == 'Adam'
    assert wrappers['distillation'].optimizer.__class__.__name__ == 'Adam'
