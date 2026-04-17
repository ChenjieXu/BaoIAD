"""Tests for the strict DeSTSeg optim-wrapper constructor."""

import torch.nn as nn
from mmengine.optim import OptimWrapperDict

import baoiad  # noqa: F401
from baoiad.engine.optimizers.destseg_optim_wrapper_constructor import DeSTSegOptimWrapperConstructor


class _ToySegmentationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = nn.Linear(8, 8)
        self.head = nn.Linear(8, 1)


class _ToyDeSTSegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_net = nn.Linear(8, 8)
        self.segmentation_net = _ToySegmentationNet()
        for param in self.segmentation_net.parameters():
            param.requires_grad = False


def test_destseg_optim_wrapper_constructor_builds_split_wrappers():
    model = _ToyDeSTSegModel()
    constructor = DeSTSegOptimWrapperConstructor(
        dict(
            student=dict(optimizer=dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=1e-4)),
            segmentation=dict(
                optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
                res_lr=0.1,
                head_lr=0.01,
            ),
        )
    )

    wrappers = constructor(model)

    assert isinstance(wrappers, OptimWrapperDict)
    assert wrappers['student'].optimizer.__class__.__name__ == 'SGD'
    assert wrappers['segmentation'].optimizer.__class__.__name__ == 'SGD'
    assert len(wrappers['segmentation'].optimizer.param_groups) == 2
    assert wrappers['segmentation'].optimizer.param_groups[0]['lr'] == 0.1
    assert wrappers['segmentation'].optimizer.param_groups[1]['lr'] == 0.01
    assert any(id(param) == id(model.student_net.weight) for param in wrappers['student'].optimizer.param_groups[0]['params'])
    assert any(
        id(param) == id(model.segmentation_net.res.weight)
        for param in wrappers['segmentation'].optimizer.param_groups[0]['params']
    )
    assert any(
        id(param) == id(model.segmentation_net.head.weight)
        for param in wrappers['segmentation'].optimizer.param_groups[1]['params']
    )
