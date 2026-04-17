"""Optimizer-wrapper constructor for strict GANomaly training."""

from __future__ import annotations

import copy

import torch.nn as nn
from mmengine.optim import OptimWrapperDict

from baoiad.registry import OPTIMIZERS, OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class GanomalyOptimWrapperConstructor:
    """Build the official split optimizers for GANomaly.

    GANomaly trains the generator and discriminator with separate Adam
    optimizers. MMEngine's default optimizer constructor only builds a single
    wrapper, so the strict path uses an ``OptimWrapperDict`` keyed by
    ``generator`` and ``discriminator``.
    """

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg: dict | None = None):
        if paramwise_cfg:
            raise ValueError('GanomalyOptimWrapperConstructor does not support paramwise_cfg.')
        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError(f'optim_wrapper_cfg must be a dict, got {type(optim_wrapper_cfg)!r}')
        self.optim_wrapper_cfg = copy.deepcopy(optim_wrapper_cfg)

    def _build_wrapper(self, module: nn.Module, wrapper_cfg: dict):
        cfg = copy.deepcopy(wrapper_cfg)
        optimizer_cfg = copy.deepcopy(cfg.pop('optimizer'))
        cfg.setdefault('type', 'OptimWrapper')

        params = [param for param in module.parameters() if param.requires_grad]
        if not params:
            raise ValueError(
                f'No trainable parameters found for GANomaly optimizer module {module.__class__.__name__}.'
            )

        optimizer_cfg['params'] = params
        optimizer = OPTIMIZERS.build(optimizer_cfg)
        return OPTIM_WRAPPERS.build(cfg, default_args=dict(optimizer=optimizer))

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        if not hasattr(model, 'generator'):
            raise AttributeError('GanomalyOptimWrapperConstructor requires model.generator.')
        if not hasattr(model, 'discriminator'):
            raise AttributeError('GanomalyOptimWrapperConstructor requires model.discriminator.')

        wrappers = {}
        if 'generator' in self.optim_wrapper_cfg:
            wrappers['generator'] = self._build_wrapper(
                model.generator,
                self.optim_wrapper_cfg['generator'],
            )
        if 'discriminator' in self.optim_wrapper_cfg:
            wrappers['discriminator'] = self._build_wrapper(
                model.discriminator,
                self.optim_wrapper_cfg['discriminator'],
            )

        if not wrappers:
            raise ValueError(
                'GanomalyOptimWrapperConstructor requires at least one of generator/discriminator configs.'
            )
        return OptimWrapperDict(**wrappers)
