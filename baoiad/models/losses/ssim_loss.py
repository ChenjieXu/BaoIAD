"""SSIM-based loss module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from baoiad.registry import MODELS


@MODELS.register_module(force=True)
class SSIMLoss(BaseModule):
    """1 - SSIM loss for reconstruction-based methods.

    Args:
        window_size (int): Size of Gaussian window. Default 11.
        loss_weight (float): Weight factor.
    """

    def __init__(self, window_size=11, loss_weight=1.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.window_size = window_size
        self.loss_weight = loss_weight

    def _gaussian_window(self, channels, device):
        from math import exp
        sigma = 1.5
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device) - self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()

    def forward(self, pred, target, **kwargs):
        C = pred.shape[1]
        window = self._gaussian_window(C, pred.device)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=C)
        mu2 = F.conv2d(target, window, padding=pad, groups=C)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu12

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return self.loss_weight * (1.0 - ssim.mean())
