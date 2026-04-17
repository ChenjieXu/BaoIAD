"""Optimizer-wrapper constructor for strict ViTAD training."""

from __future__ import annotations

import copy

import torch.nn as nn

from baoiad.registry import OPTIMIZERS, OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS


def _check_keywords_in_name(name: str, keywords=()) -> bool:
    return any(keyword in name for keyword in keywords)


def _vitad_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    skip = set(model.no_weight_decay()) if hasattr(model, 'no_weight_decay') else set()
    skip_keywords = set(model.no_weight_decay_keywords()) if hasattr(model, 'no_weight_decay_keywords') else set()

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith('.bias')
            or name in skip
            or _check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = []
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0})
    if decay:
        groups.append({'params': decay, 'weight_decay': float(weight_decay)})
    return groups


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ViTADOptimWrapperConstructor:
    """Build ADer-style AdamW param groups for ViTAD.

    ADer splits trainable parameters into ``no_decay`` and ``decay`` groups
    before constructing AdamW. The ``no_decay`` group contains 1D tensors and
    biases, mirroring upstream `add_weight_decay`.
    """

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg: dict | None = None):
        if paramwise_cfg:
            raise ValueError('ViTADOptimWrapperConstructor does not support paramwise_cfg.')
        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError(f'optim_wrapper_cfg must be a dict, got {type(optim_wrapper_cfg)!r}')
        self.optim_wrapper_cfg = copy.deepcopy(optim_wrapper_cfg)

    def __call__(self, model: nn.Module):
        if hasattr(model, 'module'):
            model = model.module

        cfg = copy.deepcopy(self.optim_wrapper_cfg)
        optimizer_cfg = copy.deepcopy(cfg.pop('optimizer'))
        cfg.setdefault('type', 'OptimWrapper')

        weight_decay = float(optimizer_cfg.pop('weight_decay', 0.0))
        optimizer_cfg['params'] = _vitad_param_groups(model, weight_decay)
        optimizer = OPTIMIZERS.build(optimizer_cfg)
        return OPTIM_WRAPPERS.build(cfg, default_args=dict(optimizer=optimizer))
