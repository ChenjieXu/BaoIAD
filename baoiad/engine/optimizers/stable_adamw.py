"""Official Dinomaly StableAdamW optimizer."""

import math

import torch
from torch.optim import Optimizer

from baoiad.registry import OPTIMIZERS


@OPTIMIZERS.register_module(force=True)
class StableAdamW(Optimizer):
    """AdamW with RMS-based adaptive step clipping.

    This matches the optimizer shipped in the official Dinomaly repository.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        clip_threshold=1.0,
    ):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if eps < 0.0:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if clip_threshold <= 0.0:
            raise ValueError(f'clip_threshold must be positive, got {clip_threshold}')

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            clip_threshold=clip_threshold,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('clip_threshold', 1.0)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> float:
        return float(tensor.norm(2) / (tensor.numel() ** 0.5))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            amsgrad = group['amsgrad']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'StableAdamW does not support sparse gradients; '
                        'consider SparseAdam instead.'
                    )

                param.data.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param)
                    state['exp_avg_sq'] = torch.zeros_like(param)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(param)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (
                        max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    ).add_(group['eps'])
                else:
                    denom = (
                        exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    ).add_(group['eps'])

                lr_scale = grad / denom
                lr_scale = max(
                    1.0,
                    self._rms(lr_scale) / group['clip_threshold'],
                )
                step_size = group['lr'] / bias_correction1 / lr_scale
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
