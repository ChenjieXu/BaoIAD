"""Optimizer-wrapper constructor for strict GLASS training."""

from __future__ import annotations

import copy

import torch.nn as nn
from mmengine.optim import OptimWrapperDict

from baoiad.registry import OPTIMIZERS, OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class GLASSOptimWrapperConstructor:
    """Build the official GLASS split optimizers.

    The strict GLASS path uses separate optimizers for the optional projection
    layer and the discriminator. MMEngine's default constructor only builds a
    single wrapper, so this constructor returns an ``OptimWrapperDict`` keyed
    by ``projection`` and ``discriminator``.
    """

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg: dict | None = None):
        if paramwise_cfg:
            raise ValueError('GLASSOptimWrapperConstructor does not support paramwise_cfg.')
        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError(f'optim_wrapper_cfg must be a dict, got {type(optim_wrapper_cfg)!r}')
        self.optim_wrapper_cfg = copy.deepcopy(optim_wrapper_cfg)

    def _build_wrapper(self, module: nn.Module, wrapper_cfg: dict):
        cfg = copy.deepcopy(wrapper_cfg)
        optimizer_cfg = copy.deepcopy(cfg.pop('optimizer'))
        cfg.setdefault('type', 'OptimWrapper')

        params = [param for param in module.parameters() if param.requires_grad]
        if not params:
            raise ValueError(f'No trainable parameters found for GLASS optimizer module {module.__class__.__name__}.')

        optimizer_cfg['params'] = params
        optimizer = OPTIMIZERS.build(optimizer_cfg)
        return OPTIM_WRAPPERS.build(cfg, default_args=dict(optimizer=optimizer))

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        wrappers = {}
        if 'projection' in self.optim_wrapper_cfg:
            if not hasattr(model, 'projection'):
                raise AttributeError('GLASS projection optimizer requested, but model has no projection module.')
            wrappers['projection'] = self._build_wrapper(model.projection, self.optim_wrapper_cfg['projection'])
        if 'discriminator' in self.optim_wrapper_cfg:
            if not hasattr(model, 'discriminator'):
                raise AttributeError('GLASS discriminator optimizer requested, but model has no discriminator module.')
            wrappers['discriminator'] = self._build_wrapper(model.discriminator, self.optim_wrapper_cfg['discriminator'])

        if not wrappers:
            raise ValueError('GLASSOptimWrapperConstructor requires at least one of projection/discriminator configs.')

        return OptimWrapperDict(**wrappers)
