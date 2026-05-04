"""Optimizer-wrapper constructor for strict RD++ training."""

from __future__ import annotations

import copy

import torch.nn as nn
from mmengine.optim import OptimWrapperDict

from baoiad.registry import OPTIMIZERS, OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class RDPPOptimWrapperConstructor:
    """Build the official RD++ split optimizers."""

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg: dict | None = None):
        if paramwise_cfg:
            raise ValueError('RDPPOptimWrapperConstructor does not support paramwise_cfg.')
        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError(f'optim_wrapper_cfg must be a dict, got {type(optim_wrapper_cfg)!r}')
        self.optim_wrapper_cfg = copy.deepcopy(optim_wrapper_cfg)

    def _build_wrapper(self, module: nn.Module, wrapper_cfg: dict):
        cfg = copy.deepcopy(wrapper_cfg)
        optimizer_cfg = copy.deepcopy(cfg.pop('optimizer'))
        cfg.setdefault('type', 'OptimWrapper')

        params = [param for param in module.parameters() if param.requires_grad]
        if not params:
            raise ValueError(f'No trainable parameters found for RD++ optimizer module {module.__class__.__name__}.')

        optimizer_cfg['params'] = params
        optimizer = OPTIMIZERS.build(optimizer_cfg)
        return OPTIM_WRAPPERS.build(cfg, default_args=dict(optimizer=optimizer))

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        if not hasattr(model, 'proj_layer'):
            raise AttributeError('RDPPOptimWrapperConstructor requires model.proj_layer.')
        if not hasattr(model, 'ocbe') or not hasattr(model, 'student'):
            raise AttributeError('RDPPOptimWrapperConstructor requires model.ocbe and model.student.')

        wrappers = {}
        if 'projection' in self.optim_wrapper_cfg:
            wrappers['projection'] = self._build_wrapper(
                model.proj_layer,
                self.optim_wrapper_cfg['projection'],
            )
        if 'distillation' in self.optim_wrapper_cfg:
            distillation_module = nn.ModuleList([model.ocbe, model.student])
            wrappers['distillation'] = self._build_wrapper(
                distillation_module,
                self.optim_wrapper_cfg['distillation'],
            )

        if not wrappers:
            raise ValueError(
                'RDPPOptimWrapperConstructor requires at least one of projection/distillation configs.'
            )
        return OptimWrapperDict(**wrappers)
