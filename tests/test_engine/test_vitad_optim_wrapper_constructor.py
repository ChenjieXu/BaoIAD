"""Tests for the strict ViTAD optim-wrapper constructor."""

import torch.nn as nn

import baoiad  # noqa: F401
from baoiad.engine.optimizers.vitad_optim_wrapper_constructor import ViTADOptimWrapperConstructor


class _ToyViTADModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)


def test_vitad_optim_wrapper_constructor_builds_decay_and_no_decay_groups():
    constructor = ViTADOptimWrapperConstructor(
        dict(
            optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
            clip_grad=dict(max_norm=5.0),
        )
    )

    wrapper = constructor(_ToyViTADModel())
    param_groups = wrapper.optimizer.param_groups

    assert len(param_groups) == 2
    assert param_groups[0]['weight_decay'] == 0.0
    assert param_groups[1]['weight_decay'] == 1e-4
    assert sum(param.numel() for param in param_groups[0]['params']) == 24
    assert sum(param.numel() for param in param_groups[1]['params']) == 64
