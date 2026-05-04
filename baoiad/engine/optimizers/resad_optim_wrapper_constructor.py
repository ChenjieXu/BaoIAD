"""Optimizer-wrapper constructor for strict ResAD training.

ResAD official uses three separate optimizers:
1. optimizer_vq: only updates VQ module
2. optimizer0: only updates constraintor module
3. optimizer1: only updates flow modules

Each module is optimized independently with separate backward/step calls.
"""

from __future__ import annotations

import copy

import torch.nn as nn
from mmengine.optim import OptimWrapperDict

from baoiad.registry import OPTIMIZERS, OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class ResADOptimWrapperConstructor:
    """Build the official ResAD split optimizers.

    ResAD uses separate optimizers for VQ, constraintor, and flow modules,
    matching the upstream training loop with independent backward passes.
    """

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg: dict | None = None):
        if paramwise_cfg:
            raise ValueError('ResADOptimWrapperConstructor does not support paramwise_cfg.')
        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError(f'optim_wrapper_cfg must be a dict, got {type(optim_wrapper_cfg)!r}')
        self.optim_wrapper_cfg = copy.deepcopy(optim_wrapper_cfg)

    def _build_wrapper(self, module: nn.Module, wrapper_cfg: dict, name: str):
        cfg = copy.deepcopy(wrapper_cfg)
        optimizer_cfg = copy.deepcopy(cfg.pop('optimizer'))
        cfg.setdefault('type', 'OptimWrapper')

        params = [param for param in module.parameters() if param.requires_grad]
        if not params:
            raise ValueError(
                f'No trainable parameters found for ResAD optimizer module {name}.'
            )

        optimizer_cfg['params'] = params
        optimizer = OPTIMIZERS.build(optimizer_cfg)
        return OPTIM_WRAPPERS.build(cfg, default_args=dict(optimizer=optimizer))

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        wrappers = {}
        cfg = self.optim_wrapper_cfg

        # VQ optimizer
        if 'vq' in cfg:
            if not hasattr(model, 'vq'):
                raise AttributeError(
                    'ResAD VQ optimizer requested, but model has no vq module.'
                )
            wrappers['vq'] = self._build_wrapper(model.vq, cfg['vq'], 'vq')

        # Constraintor optimizer
        if 'constraintor' in cfg:
            if not hasattr(model, 'constraintor'):
                raise AttributeError(
                    'ResAD constraintor optimizer requested, but model has no constraintor module.'
                )
            wrappers['constraintor'] = self._build_wrapper(
                model.constraintor, cfg['constraintor'], 'constraintor'
            )

        # Flow optimizer
        if 'flow' in cfg:
            if not hasattr(model, 'flows'):
                raise AttributeError(
                    'ResAD flow optimizer requested, but model has no flows module.'
                )
            # Combine all flow module parameters
            flow_params = []
            for flow in model.flows:
                flow_params.extend([p for p in flow.parameters() if p.requires_grad])
            if not flow_params:
                raise ValueError('No trainable parameters found in ResAD flow modules.')

            flow_cfg = copy.deepcopy(cfg['flow'])
            optimizer_cfg = copy.deepcopy(flow_cfg.pop('optimizer'))
            flow_cfg.setdefault('type', 'OptimWrapper')
            optimizer_cfg['params'] = flow_params
            optimizer = OPTIMIZERS.build(optimizer_cfg)
            wrappers['flow'] = OPTIM_WRAPPERS.build(
                flow_cfg, default_args=dict(optimizer=optimizer)
            )

        if not wrappers:
            raise ValueError(
                'ResADOptimWrapperConstructor requires at least one of vq/constraintor/flow configs.'
            )

        return OptimWrapperDict(**wrappers)