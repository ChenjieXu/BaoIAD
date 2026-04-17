"""Optimizer-wrapper constructor for strict DeSTSeg training."""

from __future__ import annotations

import copy

from mmengine.optim import OptimWrapperDict

from baoiad.registry import OPTIMIZERS, OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class DeSTSegOptimWrapperConstructor:
    """Build the official split optimizers for DeSTSeg.

    The official training loop uses:
    - one optimizer for ``student_net`` with ``lr=0.4``
    - one optimizer for ``segmentation_net`` with two param groups:
      ``res`` at ``lr=0.1`` and ``head`` at ``lr=0.01``

    MMEngine's default constructor builds only one wrapper, so strict DeSTSeg
    needs a custom constructor that returns an ``OptimWrapperDict``.
    """

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg: dict | None = None):
        if paramwise_cfg:
            raise ValueError('DeSTSegOptimWrapperConstructor does not support paramwise_cfg.')
        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError(f'optim_wrapper_cfg must be a dict, got {type(optim_wrapper_cfg)!r}')
        self.optim_wrapper_cfg = copy.deepcopy(optim_wrapper_cfg)

    @staticmethod
    def _copy_optimizer_cfg_preserve_params(optimizer_cfg: dict) -> dict:
        """Deep-copy optimizer config while keeping live Parameter references."""
        copied = copy.deepcopy({key: value for key, value in optimizer_cfg.items() if key != 'params'})
        params = optimizer_cfg.get('params')
        if params is None:
            return copied
        if isinstance(params, list):
            copied_params = []
            for item in params:
                if isinstance(item, dict):
                    item_copy = copy.deepcopy({key: value for key, value in item.items() if key != 'params'})
                    item_copy['params'] = item['params']
                    copied_params.append(item_copy)
                else:
                    copied_params.append(item)
            copied['params'] = copied_params
        else:
            copied['params'] = params
        return copied

    def _build_wrapper(self, optimizer_cfg: dict, wrapper_cfg: dict):
        cfg = copy.deepcopy(wrapper_cfg)
        cfg.setdefault('type', 'OptimWrapper')
        optimizer = OPTIMIZERS.build(self._copy_optimizer_cfg_preserve_params(optimizer_cfg))
        return OPTIM_WRAPPERS.build(cfg, default_args=dict(optimizer=optimizer))

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        if not hasattr(model, 'student_net') or not hasattr(model, 'segmentation_net'):
            raise AttributeError(
                'DeSTSegOptimWrapperConstructor requires model.student_net and model.segmentation_net.'
            )

        wrappers = {}

        if 'student' in self.optim_wrapper_cfg:
            student_cfg = copy.deepcopy(self.optim_wrapper_cfg['student'])
            optimizer_cfg = copy.deepcopy(student_cfg.pop('optimizer'))
            optimizer_cfg['params'] = list(model.student_net.parameters())
            wrappers['student'] = self._build_wrapper(optimizer_cfg, student_cfg)

        if 'segmentation' in self.optim_wrapper_cfg:
            segmentation_cfg = copy.deepcopy(self.optim_wrapper_cfg['segmentation'])
            optimizer_cfg = copy.deepcopy(segmentation_cfg.pop('optimizer'))
            res_lr = float(segmentation_cfg.pop('res_lr'))
            head_lr = float(segmentation_cfg.pop('head_lr'))
            optimizer_cfg['params'] = [
                dict(params=list(model.segmentation_net.res.parameters()), lr=res_lr),
                dict(params=list(model.segmentation_net.head.parameters()), lr=head_lr),
            ]
            wrappers['segmentation'] = self._build_wrapper(optimizer_cfg, segmentation_cfg)

        if not wrappers:
            raise ValueError(
                'DeSTSegOptimWrapperConstructor requires at least one of student/segmentation configs.'
            )

        return OptimWrapperDict(**wrappers)
