"""MemSeg: Memory-based Segmentation for Industrial Anomaly Detection (AAAI 2023).

Faithful reimplementation of https://github.com/TooTouch/MemSeg

Key components:
- Memory Bank: Stores features from normal samples
- MSFF: Multi-scale Feature Fusion with Coordinate Attention
- Decoder: UNet-style decoder with skip connections
- Synthetic anomaly generation during training

Reference results on MVTec AD:
- Image AUROC: 99.56%
- Pixel AUROC: 98.84%
"""
import glob
import logging
import math
import os
import tarfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageEnhance, ImageOps

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import ReconstructionADModel
from baoiad.utils.dtd import download_dtd as _download_dtd

logger = logging.getLogger(__name__)

# ==================== Perlin Noise ====================

def _lerp_np(x, y, w):
    return (y - x) * w + x


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    """Generate 2D Perlin noise."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_grads(s1, s2):
        return np.repeat(
            np.repeat(gradients[s1[0]:s1[1], s2[0]:s2[1]], d[0], axis=0),
            d[1],
            axis=1,
        )

    def dot(grad, shift):
        return (
            np.stack(
                (
                    grid[:shape[0], :shape[1], 0] + shift[0],
                    grid[:shape[0], :shape[1], 1] + shift[1],
                ),
                axis=-1,
            ) * grad[:shape[0], :shape[1]]
        ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * _lerp_np(_lerp_np(n00, n10, t[..., 0]), _lerp_np(n01, n11, t[..., 0]), t[..., 1])


# ==================== Coordinate Attention ====================

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention from https://github.com/houqb/CoordAttention."""

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


# ==================== MSFF Module ====================

class MSFFBlock(nn.Module):
    """Multi-scale Feature Fusion Block with Coordinate Attention."""

    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.attn = CoordAtt(in_channel, in_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x_conv = self.conv1(x)
        x_att = self.attn(x)
        x = x_conv * x_att
        x = self.conv2(x)
        return x


class MSFF(nn.Module):
    """Multi-scale Feature Fusion module."""

    def __init__(self):
        super().__init__()
        # Input channels after memory concatenation: 128, 256, 512 (doubled from original 64, 128, 256)
        self.blk1 = MSFFBlock(128)
        self.blk2 = MSFFBlock(256)
        self.blk3 = MSFFBlock(512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upconv32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.upconv21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3] concatenated features (input + memory diff)
                      with channels [128, 256, 512]
        Returns:
            [f1_out, f2_out, f3_out] fused features with channels [64, 128, 256]
        """
        f1, f2, f3 = features

        # MSFF Module
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)

        f2_f = f2_k + self.upconv32(f3_k)
        f1_f = f1_k + self.upconv21(f2_f)

        # Spatial attention using second half of channels (diff features)
        m3 = f3[:, 256:, ...].mean(dim=1, keepdim=True)
        m2 = f2[:, 128:, ...].mean(dim=1, keepdim=True) * self.upsample(m3)
        m1 = f1[:, 64:, ...].mean(dim=1, keepdim=True) * self.upsample(m2)

        f1_out = f1_f * m1
        f2_out = f2_f * m2
        f3_out = f3_k * m3

        return [f1_out, f2_out, f3_out]


# ==================== Decoder ====================

class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.blk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blk(x)


class Decoder(nn.Module):
    """UNet-style decoder for anomaly mask prediction."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1)

        self.upconv3 = UpConvBlock(512, 256)
        self.upconv2 = UpConvBlock(512, 128)
        self.upconv1 = UpConvBlock(256, 64)
        self.upconv0 = UpConvBlock(128, 48)
        self.upconv2mask = UpConvBlock(96, 48)

        self.final_conv = nn.Conv2d(48, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, encoder_output, concat_features):
        """
        Args:
            encoder_output: f_out (B, 512, H/32, W/32)
            concat_features: [f_in, f1_f, f2_f, f3_f] with channels [64, 64, 128, 256]
        """
        f0, f1, f2, f3 = concat_features

        # 512 x 8 x 8 -> 512 x 16 x 16
        x_up3 = self.upconv3(encoder_output)
        x_up3 = torch.cat([x_up3, f3], dim=1)

        # 512 x 16 x 16 -> 256 x 32 x 32
        x_up2 = self.upconv2(x_up3)
        x_up2 = torch.cat([x_up2, f2], dim=1)

        # 256 x 32 x 32 -> 128 x 64 x 64
        x_up1 = self.upconv1(x_up2)
        x_up1 = torch.cat([x_up1, f1], dim=1)

        # 128 x 64 x 64 -> 96 x 128 x 128
        x_up0 = self.upconv0(x_up1)
        f0 = self.conv(f0)
        x_up2mask = torch.cat([x_up0, f0], dim=1)

        # 96 x 128 x 128 -> 48 x 256 x 256
        x_mask = self.upconv2mask(x_up2mask)

        # 48 x 256 x 256 -> 2 x 256 x 256
        x_mask = self.final_conv(x_mask)

        return x_mask


# ==================== Focal Loss ====================

class FocalLoss(nn.Module):
    """Focal Loss with label smoothing.

    Expects softmax probabilities as input (not raw logits).
    """

    def __init__(self, gamma=4, alpha=None, smooth=1e-5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, input, target):
        """
        Args:
            input: (B, C, H, W) softmax probabilities
            target: (B, H, W) long class indices
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N, C, H*W
            input = input.transpose(1, 2)  # N, H*W, C
            input = input.contiguous().view(-1, input.size(2))  # N*H*W, C
        target = target.view(-1, 1)

        pt = input
        logpt = (pt + 1e-5).log()

        # Label smoothing
        num_class = input.shape[1]
        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != input.device:
            one_hot_key = one_hot_key.to(input.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth, 1.0 - self.smooth)
            logpt = logpt * one_hot_key

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = (-1 * (1 - pt) ** self.gamma * logpt).sum(1)
        return loss.mean()


# ==================== Memory Bank ====================

class MemoryBank:
    """Memory bank for storing normal sample features."""

    def __init__(self, nb_memory_sample: int = 30, device='cpu'):
        self.device = device
        self.memory_information = {}
        self.nb_memory_sample = nb_memory_sample

    def update(self, features_list: List[torch.Tensor]):
        """Update memory bank with features from normal samples.

        Args:
            features_list: List of feature tensors from normal samples
                           each tensor is (N, C, H, W) for each level
        """
        # features_list: list of [f1, f2, f3] tensors, each (N, C, H, W)
        for level_idx, features_l in enumerate(features_list):
            key = f'level{level_idx}'
            if key not in self.memory_information:
                self.memory_information[key] = features_l
            else:
                self.memory_information[key] = torch.cat(
                    [self.memory_information[key], features_l], dim=0
                )

    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Calculate MSE difference between input features and memory bank.

        Args:
            features: [f1, f2, f3] input features

        Returns:
            diff_bank: (B, nb_memory_sample) total MSE for each sample
        """
        diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)

        for level_idx, level in enumerate(self.memory_information.keys()):
            for b_idx, features_b in enumerate(features[level_idx]):
                diff = F.mse_loss(
                    input=torch.repeat_interleave(features_b.unsqueeze(0), repeats=self.nb_memory_sample, dim=0),
                    target=self.memory_information[level],
                    reduction='none'
                ).mean(dim=[1, 2, 3])
                diff_bank[b_idx] += diff

        return diff_bank

    def select(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Select most similar memory features and concatenate with diff.

        Args:
            features: [f1, f2, f3] input features with channels [64, 128, 256]

        Returns:
            concat_features: [f1⊕diff1, f2⊕diff2, f3⊕diff3] with doubled channels
        """
        diff_bank = self._calc_diff(features)

        for level_idx, level in enumerate(self.memory_information.keys()):
            selected_features = torch.index_select(
                self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1)
            )
            diff_features = F.mse_loss(selected_features, features[level_idx], reduction='none')
            features[level_idx] = torch.cat([features[level_idx], diff_features], dim=1)

        return features

    def to(self, device):
        self.device = device
        for key in self.memory_information:
            self.memory_information[key] = self.memory_information[key].to(device)
        return self


# ==================== Anomaly Generation ====================

# Category-specific settings for foreground mask generation
ANOMALY_MASK_SETTINGS = {
    "bottle": {"use_mask": True, "bg_threshold": 250, "bg_reverse": False},
    "cable": {"use_mask": False, "bg_threshold": 150, "bg_reverse": True},
    "capsule": {"use_mask": True, "bg_threshold": 120, "bg_reverse": False},
    "carpet": {"use_mask": False, "bg_threshold": None, "bg_reverse": None},
    "grid": {"use_mask": False, "bg_threshold": None, "bg_reverse": None},
    "hazelnut": {"use_mask": True, "bg_threshold": 50, "bg_reverse": True},
    "leather": {"use_mask": False, "bg_threshold": None, "bg_reverse": None},
    "metal_nut": {"use_mask": True, "bg_threshold": 40, "bg_reverse": True},
    "pill": {"use_mask": True, "bg_threshold": 100, "bg_reverse": True},
    "screw": {"use_mask": True, "bg_threshold": 110, "bg_reverse": False},
    "tile": {"use_mask": False, "bg_threshold": None, "bg_reverse": None},
    "toothbrush": {"use_mask": True, "bg_threshold": 30, "bg_reverse": True},
    "transistor": {"use_mask": True, "bg_threshold": 90, "bg_reverse": False},
    "wood": {"use_mask": False, "bg_threshold": None, "bg_reverse": None},
    "zipper": {"use_mask": True, "bg_threshold": 100, "bg_reverse": False},
}


def rand_augment(force_torchvision: bool = False):
    """Random augmentations for structure anomaly generation.

    The original implementation used imgaug. BaoIAD keeps a local equivalent
    augmenter set so the training path no longer depends on imgaug at runtime.
    """
    from PIL import Image  # noqa: PLC0415

    del force_torchvision

    def _as_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    class _CallableAugmenter:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, image=None, **kwargs):
            if image is None:
                image = kwargs['image']
            return self.fn(np.asarray(image))

    def _gamma_contrast(image: np.ndarray) -> np.ndarray:
        image = _as_uint8(image).astype(np.float32) / 255.0
        gamma = float(np.random.uniform(0.5, 2.0))
        adjusted = np.power(np.clip(image, 0.0, 1.0), gamma)
        return np.clip(adjusted * 255.0, 0.0, 255.0).astype(np.uint8)

    def _multiply_add_brightness(image: np.ndarray) -> np.ndarray:
        image = _as_uint8(image).astype(np.float32)
        mul = float(np.random.uniform(0.8, 1.2))
        add = float(np.random.uniform(-30.0, 30.0))
        return np.clip(image * mul + add, 0.0, 255.0).astype(np.uint8)

    def _enhance_sharpness(image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(_as_uint8(image))
        factor = float(np.random.uniform(0.0, 2.0))
        return np.asarray(Image.ImageEnhance.Sharpness(pil_img).enhance(factor), dtype=np.uint8)

    def _add_hue_saturation(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(_as_uint8(image), cv2.COLOR_RGB2HSV).astype(np.int16)
        hue_delta = int(np.random.randint(-25, 26))
        sat_delta = int(np.random.randint(-50, 51))
        hsv[..., 0] = np.mod(hsv[..., 0] + hue_delta, 180)
        hsv[..., 1] = np.clip(hsv[..., 1] + sat_delta, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _solarize(image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(_as_uint8(image))
        threshold = int(np.random.randint(32, 129))
        return np.asarray(Image.eval(pil_img, lambda value: value if value < threshold else 255 - value), dtype=np.uint8)

    def _posterize(image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(_as_uint8(image))
        bits = int(np.random.randint(4, 9))
        return np.asarray(ImageOps.posterize(pil_img, bits=bits), dtype=np.uint8)

    def _invert(image: np.ndarray) -> np.ndarray:
        return 255 - _as_uint8(image)

    def _autocontrast(image: np.ndarray) -> np.ndarray:
        return np.asarray(ImageOps.autocontrast(Image.fromarray(_as_uint8(image))), dtype=np.uint8)

    def _equalize(image: np.ndarray) -> np.ndarray:
        return np.asarray(ImageOps.equalize(Image.fromarray(_as_uint8(image))), dtype=np.uint8)

    def _affine_rotate(image: np.ndarray) -> np.ndarray:
        image = _as_uint8(image)
        angle = float(np.random.uniform(-45.0, 45.0))
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    augmenters = [
        _CallableAugmenter(_gamma_contrast),
        _CallableAugmenter(_multiply_add_brightness),
        _CallableAugmenter(_enhance_sharpness),
        _CallableAugmenter(_add_hue_saturation),
        _CallableAugmenter(_solarize),
        _CallableAugmenter(_posterize),
        _CallableAugmenter(_invert),
        _CallableAugmenter(_autocontrast),
        _CallableAugmenter(_equalize),
        _CallableAugmenter(_affine_rotate),
    ]

    class _SequentialAugment:
        def __call__(self, image=None, **kwargs):
            if image is None:
                image = kwargs['image']
            selected = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
            output = np.asarray(image)
            for index in selected:
                output = augmenters[int(index)](image=output)
            return output

    return _SequentialAugment()


class AnomalyGenerator:
    """Synthetic anomaly generator for training."""

    def __init__(
        self,
        texture_source_dir: str = None,
        structure_grid_size: int = 8,
        transparency_range: Tuple[float, float] = (0.15, 1.0),
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        perlin_noise_threshold: float = 0.5,
        use_mask: bool = True,
        bg_threshold: float = 100,
        bg_reverse: bool = False,
        require_texture_source: bool = False,
        use_imgaug: bool = True,
    ):
        self.structure_grid_size = structure_grid_size
        self.transparency_range = transparency_range
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold
        self.use_mask = use_mask
        self.bg_threshold = bg_threshold
        self.bg_reverse = bg_reverse
        self.use_imgaug = use_imgaug

        # Load texture images
        self.texture_source_paths = []
        if texture_source_dir:
            if not os.path.isdir(texture_source_dir):
                if require_texture_source:
                    raise FileNotFoundError(
                        f"MemSeg strict mode requires texture_source_dir, got missing path: "
                        f"{texture_source_dir}"
                    )
            else:
                self.texture_source_paths = sorted(
                    glob.glob(os.path.join(texture_source_dir, '*', '*.jpg')) +
                    glob.glob(os.path.join(texture_source_dir, '*', '*.png')) +
                    glob.glob(os.path.join(texture_source_dir, '*.jpg')) +
                    glob.glob(os.path.join(texture_source_dir, '*.png'))
                )
                if self.texture_source_paths:
                    logger.info(f"MemSeg: Loaded {len(self.texture_source_paths)} DTD texture images")

        if require_texture_source and not self.texture_source_paths:
            raise FileNotFoundError(
                'MemSeg strict mode requires DTD texture assets, but no texture images were found.'
            )

        # Anomaly switch for alternating between normal and anomaly
        self.anomaly_switch = False

    def generate_target_foreground_mask(self, img: np.ndarray) -> np.ndarray:
        """Generate foreground mask from image."""
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, target_background_mask = cv2.threshold(
            img_gray, self.bg_threshold, 255, cv2.THRESH_BINARY
        )
        target_background_mask = target_background_mask.astype(bool).astype(int)

        if self.bg_reverse:
            target_foreground_mask = target_background_mask
        else:
            target_foreground_mask = -(target_background_mask - 1)

        return target_foreground_mask

    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        """Generate Perlin noise mask."""
        perlin_scalex = 2 ** int(torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).item())
        perlin_scaley = 2 ** int(torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).item())

        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))
        angle = np.random.uniform(-90, 90)
        h, w = img_size
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        perlin_noise = cv2.warpAffine(
            perlin_noise.astype(np.float32),
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )

        return mask_noise

    def _texture_source(self, img_size: tuple) -> np.ndarray:
        """Get texture anomaly source from DTD."""
        if self.texture_source_paths:
            idx = np.random.randint(len(self.texture_source_paths))
            texture_source_img = cv2.imread(self.texture_source_paths[idx])
            texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
            texture_source_img = cv2.resize(texture_source_img, dsize=img_size).astype(np.float32)
        else:
            texture_source_img = np.random.rand(*img_size, 3).astype(np.float32) * 255
        return texture_source_img

    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        """Generate structure anomaly by shuffling image patches."""
        structure_source_img = rand_augment(force_torchvision=not self.use_imgaug)(image=img)

        img_size = img.shape[:-1]
        grid_w = img_size[1] // self.structure_grid_size
        grid_h = img_size[0] // self.structure_grid_size

        structure_source_img = rearrange(
            tensor=structure_source_img,
            pattern='(h gh) (w gw) c -> (h w) gw gh c',
            gw=grid_w,
            gh=grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor=structure_source_img[disordered_idx],
            pattern='(h w) gw gh c -> (h gh) (w gw) c',
            h=self.structure_grid_size,
            w=self.structure_grid_size
        ).astype(np.float32)

        return structure_source_img

    def anomaly_source(self, img: np.ndarray) -> np.ndarray:
        """Randomly select texture or structure anomaly source."""
        p = np.random.uniform() if self.texture_source_paths else 1.0
        if p < 0.5:
            return self._texture_source(img.shape[:-1])
        else:
            return self._structure_source(img)

    def generate_anomaly(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic anomaly image and mask.

        Args:
            img: (H, W, 3) uint8 image

        Returns:
            anomaly_img: (H, W, 3) uint8 anomaly image
            mask: (H, W) float32 binary mask
        """
        img_size = img.shape[:-1]

        # Generate foreground mask
        if self.use_mask:
            target_foreground_mask = self.generate_target_foreground_mask(img)
        else:
            target_foreground_mask = np.ones(img_size)

        # Generate Perlin noise mask
        perlin_noise_mask = self.generate_perlin_noise_mask(img_size)

        # Combine masks
        mask = perlin_noise_mask * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)

        # Generate anomaly source
        anomaly_source_img = self.anomaly_source(img)

        # Blend anomaly with original
        factor = np.random.uniform(*self.transparency_range)
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + \
                            (1 - factor) * (mask_expanded * img)

        # Blend with original image
        anomaly_img = ((-mask_expanded + 1) * img) + anomaly_source_img

        return anomaly_img.astype(np.uint8), mask.astype(np.float32)


# ==================== MemSeg Detector ====================

@MODELS.register_module()
class MemSegDetector(ReconstructionADModel):
    """MemSeg: Memory-based Segmentation for Anomaly Detection.

    Args:
        backbone: Backbone config dict (ResNet18 with 5 feature levels).
        nb_memory_sample: Number of normal samples to store in memory bank.
        dtd_path: Path to DTD texture dataset for anomaly generation.
        structure_grid_size: Grid size for structure anomaly.
        transparency_range: Range for anomaly blending factor.
        perlin_scale: Maximum scale for Perlin noise.
        min_perlin_scale: Minimum scale for Perlin noise.
        perlin_noise_threshold: Threshold for Perlin noise mask.
        l1_weight: Weight for L1 loss.
        focal_weight: Weight for Focal loss.
        focal_gamma: Gamma parameter for Focal loss.
        anomaly_ratio: Ratio of anomaly samples during training.
    """

    def __init__(
        self,
        backbone: dict,
        nb_memory_sample: int = 30,
        dtd_path: str = 'auto',
        structure_grid_size: int = 8,
        transparency_range: Tuple[float, float] = (0.15, 1.0),
        perlin_scale: int = 6,
        min_perlin_scale: int = 0,
        perlin_noise_threshold: float = 0.5,
        l1_weight: float = 0.6,
        focal_weight: float = 0.4,
        focal_gamma: int = 4,
        anomaly_ratio: float = 0.5,
        alternate_anomaly_sampling: bool = False,
        require_texture_source: bool = False,
        memory_bank_seed: int = 42,
        use_imgaug: bool = True,
        anomaly_source_resize: Optional[Union[int, Tuple[int, int]]] = None,
        anomaly_source_crop: Optional[Union[int, Tuple[int, int]]] = None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build backbone (ResNet18 with 5 feature levels)
        self.backbone = MODELS.build(backbone)
        self.freeze_backbone = not any(p.requires_grad for p in self.backbone.parameters())
        if self.freeze_backbone:
            self.backbone.eval()

        # Memory bank
        self.nb_memory_sample = nb_memory_sample
        self.memory_bank = None

        # MSFF and Decoder
        self.msff = MSFF()
        self.decoder = Decoder()

        # Loss
        self.l1_loss = nn.L1Loss()
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.l1_weight = l1_weight
        self.focal_weight = focal_weight

        # Anomaly generation settings
        self.anomaly_ratio = anomaly_ratio
        self.alternate_anomaly_sampling = alternate_anomaly_sampling
        self.require_texture_source = require_texture_source
        self.memory_bank_seed = memory_bank_seed
        self.use_imgaug = use_imgaug
        self.anomaly_source_resize = anomaly_source_resize
        self.anomaly_source_crop = anomaly_source_crop

        # DTD texture path
        self.dtd_path = dtd_path
        self._dtd_dir = None

        # Category-specific settings (will be set during training)
        self._category_settings = None
        self._anomaly_generator = None
        self._anomaly_switch = False
        self._memory_bank_built = False
        self.pre_train_setup_builds_memory_bank = True

    def _get_dtd_dir(self):
        """Get DTD directory, downloading if necessary."""
        if self._dtd_dir is None:
            if self.dtd_path == 'auto':
                try:
                    self._dtd_dir = _download_dtd()
                except Exception as e:
                    if self.require_texture_source:
                        raise
                    logger.warning(f"Failed to auto-download DTD: {e}. Using random textures.")
                    self._dtd_dir = None
            else:
                if os.path.isdir(self.dtd_path):
                    self._dtd_dir = self.dtd_path
                elif self.require_texture_source:
                    raise FileNotFoundError(
                        f"MemSeg strict mode requires an existing DTD path, got: {self.dtd_path}"
                    )
                else:
                    logger.warning(
                        'MemSeg: DTD path %s does not exist. Falling back to structure-only anomalies.',
                        self.dtd_path,
                    )
                    self._dtd_dir = None
        return self._dtd_dir

    def _get_anomaly_generator(self, category: str = None):
        """Get anomaly generator for specific category."""
        if self._anomaly_generator is None or (category and self._category_settings != category):
            settings = ANOMALY_MASK_SETTINGS.get(category, {
                "use_mask": True, "bg_threshold": 100, "bg_reverse": False
            })
            self._anomaly_generator = AnomalyGenerator(
                texture_source_dir=self._get_dtd_dir(),
                structure_grid_size=8,
                transparency_range=(0.15, 1.0),
                perlin_scale=6,
                min_perlin_scale=0,
                perlin_noise_threshold=0.5,
                use_mask=settings["use_mask"],
                bg_threshold=settings.get("bg_threshold", 100) or 100,
                bg_reverse=settings.get("bg_reverse", False) or False,
                require_texture_source=self.require_texture_source,
                use_imgaug=self.use_imgaug,
            )
            self._category_settings = category
        return self._anomaly_generator

    @staticmethod
    def _normalize_hw(size: Optional[Union[int, Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
        """Convert spatial size to ``(height, width)``."""
        if size is None:
            return None
        if isinstance(size, int):
            return (size, size)
        if len(size) != 2:
            raise ValueError(f'Expected spatial size as int or pair, got {size!r}')
        return (int(size[0]), int(size[1]))

    @staticmethod
    def _center_crop_array(array: np.ndarray, crop_size: Union[int, Tuple[int, int]]) -> np.ndarray:
        """Center crop a 2D or 3D numpy array."""
        crop_h, crop_w = MemSegDetector._normalize_hw(crop_size)
        height, width = array.shape[:2]
        if crop_h > height or crop_w > width:
            raise ValueError(
                f'Crop size {(crop_h, crop_w)} exceeds array shape {(height, width)}.'
            )
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        if array.ndim == 2:
            return array[top:top + crop_h, left:left + crop_w]
        return array[top:top + crop_h, left:left + crop_w, ...]

    def _load_training_source_image(self, img_path: str) -> np.ndarray:
        """Load the pre-crop training image used by the frozen reference pipeline."""
        with Image.open(img_path) as _img:
            image = _img.convert('RGB')
        resize_hw = self._normalize_hw(self.anomaly_source_resize)
        if resize_hw is not None:
            resize_h, resize_w = resize_hw
            image = image.resize((resize_w, resize_h))
        return np.array(image, dtype=np.uint8)

    def _get_training_source_image(
        self,
        idx: int,
        data_samples: Optional[List],
        fallback_image: np.ndarray,
    ) -> np.ndarray:
        """Resolve the raw image used for anomaly synthesis during training."""
        if not data_samples or idx >= len(data_samples):
            return fallback_image
        img_path = getattr(data_samples[idx], 'img_path', None)
        if not img_path or not os.path.isfile(img_path):
            return fallback_image
        return self._load_training_source_image(img_path)

    def extract_feat(
        self,
        x: torch.Tensor,
        requires_grad: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Extract multi-scale features from backbone.

        Returns:
            features: (f_in, f1, f2, f3, f_out) with channels [64, 64, 128, 256, 512]
        """
        if requires_grad is None:
            requires_grad = not self.freeze_backbone
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            feats = self.backbone(x)
        if isinstance(feats, torch.Tensor):
            feats = (feats,)
        return feats

    @staticmethod
    def _unwrap_dataset(dataset: Any) -> Any:
        """Unwrap dataset wrappers such as RepeatDataset."""
        visited = set()
        current = dataset
        while hasattr(current, 'dataset') and id(current) not in visited:
            visited.add(id(current))
            current = current.dataset
        return current

    @staticmethod
    def _extract_inputs_from_batch(batch: Any) -> torch.Tensor:
        """Extract image tensor(s) from dataset items or loader batches."""
        if isinstance(batch, dict):
            inputs = batch.get('inputs', batch.get('img'))
        elif isinstance(batch, (list, tuple)):
            inputs = batch[0]
        else:
            inputs = batch

        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(list(inputs))
        if torch.is_tensor(inputs) and inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        return inputs

    def _build_memory_bank_from_dataset(self, dataset: Any, device: torch.device) -> bool:
        """Build the memory bank using official-style random dataset sampling."""
        dataset = self._unwrap_dataset(dataset)
        if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
            return False

        num_items = len(dataset)
        if num_items <= 0:
            raise RuntimeError('MemSeg: cannot build memory bank from an empty dataset.')

        num_samples = min(self.nb_memory_sample, num_items)
        rng = np.random.RandomState(self.memory_bank_seed)
        sample_indices = np.arange(num_items)
        rng.shuffle(sample_indices)
        sample_indices = sample_indices[:num_samples]

        collected = [[], [], []]
        for sample_idx in sample_indices:
            item = dataset[int(sample_idx)]
            inputs = self._extract_inputs_from_batch(item)
            if not torch.is_tensor(inputs):
                raise TypeError(f'MemSeg: unsupported dataset item type for memory bank: {type(inputs)!r}')
            inputs = inputs.to(device)
            features = self.extract_feat(inputs, requires_grad=False)
            for level, feat in enumerate(features[1:4]):
                collected[level].append(feat)

        self.memory_bank = MemoryBank(nb_memory_sample=num_samples, device=device)
        self.memory_bank.update([
            torch.cat(level_features, dim=0) for level_features in collected
        ])
        logger.info('MemSeg: Built memory bank with %d samples (dataset random sampling)', num_samples)
        return True

    def _build_memory_bank_from_iterable(self, dataloader: Iterable[Any], device: torch.device) -> None:
        """Fallback memory-bank build path for synthetic/unit-test loaders."""
        all_features = [[], [], []]

        with torch.no_grad():
            for data in dataloader:
                if len(all_features[0]) >= self.nb_memory_sample:
                    break

                inputs = self._extract_inputs_from_batch(data)
                if isinstance(inputs, (list, tuple)):
                    inputs = torch.stack(list(inputs))
                if not torch.is_tensor(inputs):
                    raise TypeError(f'MemSeg: unsupported dataloader batch type: {type(inputs)!r}')
                inputs = inputs.to(device)

                features = self.extract_feat(inputs, requires_grad=False)
                f1, f2, f3 = features[1], features[2], features[3]

                for b in range(f1.size(0)):
                    if len(all_features[0]) >= self.nb_memory_sample:
                        break
                    all_features[0].append(f1[b:b + 1])
                    all_features[1].append(f2[b:b + 1])
                    all_features[2].append(f3[b:b + 1])

        if not all_features[0]:
            raise RuntimeError('MemSeg: failed to collect any samples for the memory bank.')

        self.memory_bank = MemoryBank(nb_memory_sample=len(all_features[0]), device=device)
        self.memory_bank.update([
            torch.cat(all_features[0], dim=0),
            torch.cat(all_features[1], dim=0),
            torch.cat(all_features[2], dim=0),
        ])
        logger.info('MemSeg: Built memory bank with %d samples (iterable fallback)', len(all_features[0]))

    def build_memory_bank(self, dataloader=None) -> None:
        """Build memory bank from training data.

        Called by MemoryBankHook after training starts.

        Args:
            dataloader: Training dataloader to extract features from.
        """
        if self._memory_bank_built and self.memory_bank is not None:
            return

        if dataloader is None:
            logger.warning("MemSeg: No dataloader provided for memory bank building")
            return

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            dataset = getattr(dataloader, 'dataset', None)
            built_from_dataset = False
            if dataset is not None:
                built_from_dataset = self._build_memory_bank_from_dataset(dataset, device)
            if not built_from_dataset:
                self._build_memory_bank_from_iterable(dataloader, device)
            self._memory_bank_built = True
        finally:
            if was_training:
                self.train(True)

    def pre_train_setup(self, dataloader=None) -> None:
        """Build memory bank before training starts.

        Called by MemoryBankHook at before_train.
        This ensures memory bank is available during training.
        """
        self.build_memory_bank(dataloader)

    def forward(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        data_samples: Optional[List] = None,
        mode: str = 'tensor',
    ) -> Union[Dict[str, torch.Tensor], List, Tuple[torch.Tensor, ...]]:
        """Forward pass with three modes.

        Args:
            inputs: Input images (B, C, H, W)
            data_samples: List of ADDataSample
            mode: 'loss', 'predict', or 'tensor'

        Returns:
            - 'loss': dict of loss tensors
            - 'predict': list of ADDataSample with predictions
            - 'tensor': raw output logits
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        B, C, H, W = inputs.shape
        device = inputs.device

        if mode == 'loss':
            # Training: generate synthetic anomalies
            # Get category from data_samples
            category = None
            if data_samples and len(data_samples) > 0:
                category = getattr(data_samples[0], 'cls_name', None)

            anomaly_gen = self._get_anomaly_generator(category)

            augmented_list = []
            mask_list = []

            # Denormalize for anomaly generation (ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            inputs_denorm = inputs * std + mean
            inputs_denorm = (inputs_denorm * 255).clamp(0, 255).cpu().numpy()
            crop_hw = self._normalize_hw(self.anomaly_source_crop)
            if crop_hw is None:
                crop_hw = (H, W)

            for i in range(B):
                fallback_img = inputs_denorm[i].transpose(1, 2, 0).astype(np.uint8)
                source_img = self._get_training_source_image(i, data_samples, fallback_img)

                # Decide if generating anomaly
                if self.alternate_anomaly_sampling:
                    should_generate = self._anomaly_switch
                    self._anomaly_switch = not self._anomaly_switch
                else:
                    should_generate = np.random.random() < self.anomaly_ratio

                if should_generate:
                    aug_img, mask = anomaly_gen.generate_anomaly(source_img)
                else:
                    aug_img = source_img
                    mask = np.zeros(source_img.shape[:2], dtype=np.float32)

                if aug_img.shape[:2] != crop_hw:
                    aug_img = self._center_crop_array(aug_img, crop_hw)
                if mask.shape[:2] != crop_hw:
                    mask = self._center_crop_array(mask, crop_hw)

                augmented_list.append(aug_img.transpose(2, 0, 1))
                mask_list.append(mask.astype(np.float32))

            # Convert back to tensor and normalize
            augmented = torch.from_numpy(np.stack(augmented_list)).float().to(device) / 255.0
            augmented = (augmented - mean) / std
            masks = torch.from_numpy(np.stack(mask_list)).float().to(device)

            # Forward pass
            features = self.extract_feat(augmented)
            f_in, f1, f2, f3, f_out = features

            # Memory bank selection (if available)
            if self.memory_bank is not None:
                concat_features = self.memory_bank.select([f1, f2, f3])
            else:
                # During first epoch, use zero diff
                concat_features = [
                    torch.cat([f1, torch.zeros_like(f1)], dim=1),
                    torch.cat([f2, torch.zeros_like(f2)], dim=1),
                    torch.cat([f3, torch.zeros_like(f3)], dim=1),
                ]

            # MSFF
            msff_outputs = self.msff(concat_features)

            # Decoder
            pred = self.decoder(f_out, [f_in] + msff_outputs)

            # Compute loss
            pred_softmax = F.softmax(pred, dim=1)
            l1_loss = self.l1_loss(pred_softmax[:, 1], masks)
            focal_loss = self.focal_loss(pred_softmax, masks.long())

            return {
                'loss': self.l1_weight * l1_loss + self.focal_weight * focal_loss,
                'l1_loss': l1_loss,
                'focal_loss': focal_loss,
            }

        elif mode == 'predict':
            # Inference
            if self.memory_bank is None:
                raise RuntimeError(
                    'MemSeg memory bank is not built. Call build_memory_bank(dataloader) first.'
                )

            features = self.extract_feat(inputs)
            f_in, f1, f2, f3, f_out = features

            # Memory bank selection
            concat_features = self.memory_bank.select([f1, f2, f3])

            # MSFF
            msff_outputs = self.msff(concat_features)

            # Decoder
            pred = self.decoder(f_out, [f_in] + msff_outputs)

            # Get anomaly probability
            prob = F.softmax(pred, dim=1)[:, 1]  # (B, H, W)

            # Image score: top-k mean (matching original implementation)
            # Original uses topk(100) mean
            B, H, W = prob.shape
            img_scores = torch.topk(prob.view(B, -1), min(100, H * W), dim=1)[0].mean(dim=1)

            return build_predict_results(data_samples, img_scores, prob)

        else:
            # Tensor mode
            features = self.extract_feat(inputs)
            f_in, f1, f2, f3, f_out = features

            if self.memory_bank is not None:
                concat_features = self.memory_bank.select([f1, f2, f3])
            else:
                concat_features = [
                    torch.cat([f1, torch.zeros_like(f1)], dim=1),
                    torch.cat([f2, torch.zeros_like(f2)], dim=1),
                    torch.cat([f3, torch.zeros_like(f3)], dim=1),
                ]

            msff_outputs = self.msff(concat_features)
            pred = self.decoder(f_out, [f_in] + msff_outputs)

            return pred

    def train(self, mode: bool = True):
        """Override to keep backbone frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
