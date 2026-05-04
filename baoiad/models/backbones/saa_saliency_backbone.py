"""Official-style saliency backbone used by SAA+.

This mirrors the public Segment-Any-Anomaly ``ModelINet`` behavior:
longest-side resize, square padding, multi-scale feature concatenation, and
feature normalization before the self-similarity calculation.
"""

import glob
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torchvision.transforms.functional import resize, to_pil_image

from baoiad.registry import MODELS


class ResizeLongestSide:
    """Resize an image so its longest side matches a target length."""

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    @staticmethod
    def get_preprocess_shape(
        oldh: int,
        oldw: int,
        long_side_length: int,
    ) -> Tuple[int, int]:
        scale = float(long_side_length) / float(max(oldh, oldw))
        newh = int(oldh * scale + 0.5)
        neww = int(oldw * scale + 0.5)
        return newh, neww

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self.get_preprocess_shape(
            image.shape[0],
            image.shape[1],
            self.target_length,
        )
        return np.array(resize(to_pil_image(image), target_size))


@MODELS.register_module(force=True)
class SAASaliencyBackbone(BaseModule):
    """Feature extractor aligned with the official SAA+ ``ModelINet``."""

    def __init__(
        self,
        model_name: str = 'wide_resnet50_2',
        out_indices=(1, 2, 3),
        pretrained: bool = True,
        checkpoint_path: str = '',
        pool_last: bool = False,
        image_size: int = 1024,
        frozen: bool = True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        import timm
        import torch.hub

        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs['out_indices'] = list(out_indices)

        resolved_checkpoint = checkpoint_path
        resolved_pretrained = pretrained
        if resolved_checkpoint:
            resolved_pretrained = False
        if not resolved_checkpoint:
            hub_dir = torch.hub.get_dir()
            checkpoint_glob = os.path.join(
                hub_dir,
                'checkpoints',
                f'{model_name}*.pth',
            )
            matches = sorted(glob.glob(checkpoint_glob))
            if matches:
                resolved_checkpoint = matches[0]
                resolved_pretrained = False

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=resolved_pretrained,
            checkpoint_path='',
            **kwargs,
        )
        if resolved_checkpoint and os.path.exists(resolved_checkpoint):
            checkpoint = torch.load(resolved_checkpoint, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            else:
                state_dict = checkpoint
            if isinstance(state_dict, dict) and next(iter(state_dict)).startswith('module.'):
                state_dict = {key[7:]: value for key, value in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None

        self.pixel_mean = torch.tensor(
            [0.485 * 255, 0.456 * 255, 0.406 * 255],
            dtype=torch.float32,
        ).view(-1, 1, 1)
        self.pixel_std = torch.tensor(
            [0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=torch.float32,
        ).view(-1, 1, 1)

        self.image_size = image_size
        self.transform = ResizeLongestSide(self.image_size)

        if frozen:
            self.eval()
            for parameter in self.parameters():
                parameter.requires_grad = False

    def set_img_size(self, image_size: int) -> None:
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, float, float]:
        input_image = self.transform.apply_image(image)
        x = torch.as_tensor(input_image, dtype=torch.float32, device=self.pixel_mean.device)
        x = x.permute(2, 0, 1).contiguous().unsqueeze(0)
        x = (x - self.pixel_mean) / self.pixel_std

        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))

        ratio_h = float(h) / float(self.image_size)
        ratio_w = float(w) / float(self.image_size)
        return x, ratio_h, ratio_w

    @torch.no_grad()
    def forward(self, image: np.ndarray) -> Tuple[torch.Tensor, float, float]:
        x, ratio_h, ratio_w = self.preprocess(image)
        x = x.to(next(self.backbone.parameters()).device)

        features = list(self.backbone(x))

        if self.avg_pool is not None:
            pooled = self.avg_pool(features[-1])
            pooled = torch.flatten(pooled, 1)
            features.append(pooled)

        size_0 = features[0].shape[2:]
        for index in range(1, len(features)):
            if features[index].ndim == 4:
                features[index] = F.interpolate(features[index], size=size_0)
            else:
                features[index] = features[index][:, :, None, None].expand(
                    -1, -1, size_0[0], size_0[1]
                )

        features = torch.cat(features, dim=1)
        features = F.normalize(features, dim=1)
        return features, ratio_h, ratio_w

    def train(self, mode: bool = True):
        if mode and not any(parameter.requires_grad for parameter in self.parameters()):
            return super().train(False)
        return super().train(mode)
