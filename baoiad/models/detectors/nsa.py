"""NSA: Natural Synthetic Anomalies for Self-Supervised Anomaly Detection.

Reference: Schluter et al., "Natural Synthetic Anomalies for Self-Supervised
Anomaly Detection and Localization", ECCV 2022.

Aligned with the official MVTec "NSA (logistic)" setting:
- ResNet-18 encoder with U-Net decoder
- logistic-intensity labels trained with sigmoid + BCE
- Poisson blending with class-dependent source/mixed gradients
- ImageNet-normalized features for backbone, [0,1] images for blending
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import median as sk_median
from skimage.morphology import disk as sk_disk
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import BaseADModel

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

logger = logging.getLogger(__name__)

CV2_NORMAL_CLONE = cv2.NORMAL_CLONE if HAS_CV2 else 1
CV2_MIXED_CLONE = cv2.MIXED_CLONE if HAS_CV2 else 2

# MVTec AD texture categories use MIXED_CLONE for Poisson blending
TEXTURE_CATEGORIES = {'carpet', 'grid', 'leather', 'tile', 'wood'}
OBJECT_CATEGORIES = {
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
}

# Per-category width bounds for patch generation (from original NSA)
# Format: ((height_min_pct, height_max_pct), (width_min_pct, width_max_pct))
WIDTH_BOUNDS_PCT = {
    'bottle': ((0.03, 0.4), (0.03, 0.4)),
    'cable': ((0.05, 0.4), (0.05, 0.4)),
    'capsule': ((0.03, 0.15), (0.03, 0.4)),
    'hazelnut': ((0.03, 0.35), (0.03, 0.35)),
    'metal_nut': ((0.03, 0.4), (0.03, 0.4)),
    'pill': ((0.03, 0.2), (0.03, 0.4)),
    'screw': ((0.03, 0.12), (0.03, 0.12)),
    'toothbrush': ((0.03, 0.4), (0.03, 0.2)),
    'transistor': ((0.03, 0.4), (0.03, 0.4)),
    'zipper': ((0.03, 0.4), (0.03, 0.2)),
    'carpet': ((0.03, 0.4), (0.03, 0.4)),
    'grid': ((0.03, 0.4), (0.03, 0.4)),
    'leather': ((0.03, 0.4), (0.03, 0.4)),
    'tile': ((0.03, 0.4), (0.03, 0.4)),
    'wood': ((0.03, 0.4), (0.03, 0.4)),
}

# Per-category logistic transform parameters (k, x0) for label generation
# From original NSA self_sup_tasks.py
INTENSITY_LOGISTIC_PARAMS = {
    'bottle': (1/12, 24),
    'cable': (1/12, 24),
    'capsule': (1/2, 4),
    'hazelnut': (1/12, 24),
    'metal_nut': (1/3, 7),
    'pill': (1/3, 7),
    'screw': (1, 3),
    'toothbrush': (1/6, 15),
    'transistor': (1/6, 15),
    'zipper': (1/6, 15),
    'carpet': (1/3, 7),
    'grid': (1/3, 7),
    'leather': (1/3, 7),
    'tile': (1/3, 7),
    'wood': (1/6, 15),
}

# Per-category number of patches to generate per image
NUM_PATCHES = {
    'bottle': 3,
    'cable': 3,
    'capsule': 3,
    'hazelnut': 3,
    'metal_nut': 3,
    'pill': 3,
    'screw': 4,
    'toothbrush': 3,
    'transistor': 3,
    'zipper': 4,
    'carpet': 4,
    'grid': 4,
    'leather': 4,
    'tile': 4,
    'wood': 4,
}

# Default parameters for unknown categories
DEFAULT_WIDTH_BOUNDS = ((0.03, 0.4), (0.03, 0.4))
DEFAULT_LOGISTIC_PARAMS = (1/3, 7)
DEFAULT_NUM_PATCHES = 3

# Background detection parameters for skip_background logic
# Format: (brightness_threshold, binary_threshold)
# From original NSA train_mvtec.py: BACKGROUND = {'bottle':(200, 60), ...}
# For bright backgrounds (mean > 127): object is darker, use < threshold
# For dark backgrounds (mean <= 127): object is brighter, use > threshold
BACKGROUND_PARAMS = {
    'bottle': (200, 60),    # bright background
    'screw': (200, 60),
    'capsule': (200, 60),
    'zipper': (200, 60),
    'hazelnut': (20, 20),   # dark background
    'pill': (20, 20),
    'toothbrush': (20, 20),
    'metal_nut': (20, 20),
}

# Minimum object percentage for patch generation
# Patches must cover at least this much of the object (not background)
MIN_OBJECT_PCT = {
    'bottle': 0.7, 'capsule': 0.7,
    'hazelnut': 0.7, 'metal_nut': 0.5, 'pill': 0.7,
    'screw': 0.5, 'toothbrush': 0.25,
    'zipper': 0.7,
}

# Minimum overlap between pasted source object area and destination object area
MIN_OVERLAP_PCT = {
    'bottle': 0.25, 'capsule': 0.25,
    'hazelnut': 0.25, 'metal_nut': 0.25, 'pill': 0.25,
    'screw': 0.25, 'toothbrush': 0.25, 'zipper': 0.25,
}

DEFAULT_GAMMA_PARAMS = (2.0, 0.05, 0.03)
DEFAULT_RESIZE_BOUNDS = (0.7, 1.3)
TEXTURE_RESIZE_BOUNDS = (0.5, 2.0)


class UNetSegHead(nn.Module):
    """Lightweight U-Net segmentation head for anomaly detection."""

    def __init__(self, in_channels_list, base_width=64):
        super().__init__()
        b = base_width
        self.up3 = nn.Sequential(
            nn.Conv2d(in_channels_list[2], b * 4, 1),
            nn.ReLU(True),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(b * 4 + in_channels_list[1], b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4), nn.ReLU(True),
            nn.Conv2d(b * 4, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(b * 2 + in_channels_list[0], b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(True),
            nn.Conv2d(b * 2, b, 3, padding=1),
            nn.BatchNorm2d(b), nn.ReLU(True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b), nn.ReLU(True),
            nn.Conv2d(b, 1, 1),
        )

    def forward(self, f1, f2, f3):
        """f1, f2, f3: multi-scale features (large to small spatial)."""
        x = self.up3(f3)
        x = F.interpolate(x, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, f1], dim=1))
        return self.final(x)


def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride,
        padding=1, bias=False,
    )


def _conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def _conv3x3_transpose(in_channels, out_channels, stride=1):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=3, stride=stride,
        padding=1, output_padding=1 if stride > 1 else 0, bias=False,
    )


def _conv1x1_transpose(in_channels, out_channels, stride=1):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=1, stride=stride,
        output_padding=1 if stride > 1 else 0, bias=False,
    )


class _EncoderBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class _DecoderBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, upsample=None):
        super().__init__()
        self.conv1 = _conv3x3_transpose(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = upsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out = self.relu(out + identity)
        return out


class NSAResNetEncDec(nn.Module):
    """Official NSA ResNet-18 encoder-decoder head."""

    def __init__(self, out_channels=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_encoder_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_encoder_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_encoder_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_encoder_layer(512, blocks=2, stride=2)

        self.nin = nn.Sequential(
            _conv1x1(512, 256),
            nn.ReLU(inplace=True),
            _conv1x1(256, 128),
            nn.ReLU(inplace=True),
        )

        self.inplanes = 128
        self.uplayer1 = self._make_decoder_layer(64, blocks=1, stride=2)
        self.uplayer2 = self._make_decoder_layer(32, blocks=1, stride=2)
        self.uplayer3 = self._make_decoder_layer(16, blocks=1, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convtranspose1 = nn.ConvTranspose2d(
            16, out_channels, kernel_size=7, stride=2, padding=3,
            output_padding=1, bias=False,
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_encoder_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = [_EncoderBlock(self.inplanes, out_channels, stride=stride, downsample=downsample)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(_EncoderBlock(self.inplanes, out_channels))
        return nn.Sequential(*layers)

    def _make_decoder_layer(self, out_channels, blocks, stride):
        upsample = None
        if stride != 1 or self.inplanes != out_channels:
            upsample = nn.Sequential(
                _conv1x1_transpose(self.inplanes, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = [_DecoderBlock(self.inplanes, out_channels, stride=stride, upsample=upsample)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(_DecoderBlock(self.inplanes, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.nin(x)
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.upsample(x)
        x = self.convtranspose1(x)
        return x


# Pre-computed Gaussian blur kernel for smooth mask boundaries (constant)
_SMOOTH_KSIZE = 15
_SMOOTH_SIGMA = 4.0
_x_coord = torch.arange(_SMOOTH_KSIZE).float() - _SMOOTH_KSIZE // 2
_kernel_1d = torch.exp(-_x_coord ** 2 / (2 * _SMOOTH_SIGMA ** 2))
_kernel_1d = _kernel_1d / _kernel_1d.sum()
_SMOOTH_KERNEL = (_kernel_1d.unsqueeze(1) * _kernel_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)


def _generate_smooth_mask(H, W, width_bounds_pct=None, device='cpu',
                          center_bounds=None):
    """Generate a smooth mask for blending using random ellipses/blobs.

    Uses gamma distribution for patch sizes following the original NSA paper.

    Args:
        H, W: Spatial dimensions
        width_bounds_pct: ((h_min_pct, h_max_pct), (w_min_pct, w_max_pct))
            Per-category bounds for patch sizes
        device: torch device
        center_bounds: Optional (cy_min, cy_max, cx_min, cx_max) to constrain
            the ellipse center for guided sampling within the object region.

    Returns:
        (1, H, W) tensor with smooth mask
    """
    if width_bounds_pct is None:
        width_bounds_pct = DEFAULT_WIDTH_BOUNDS

    mask = np.zeros((H, W), dtype=np.float32)

    gamma_shape, gamma_scale = 2.0, 0.05

    min_h = int(width_bounds_pct[0][0] * H)
    max_h = int(width_bounds_pct[0][1] * H)
    min_w = int(width_bounds_pct[1][0] * W)
    max_w = int(width_bounds_pct[1][1] * W)

    min_h = max(min_h, H // 32)
    min_w = max(min_w, W // 32)

    rx = int(np.clip(min_w + np.random.gamma(gamma_shape, gamma_scale) * W, min_w, max_w))
    ry = int(np.clip(min_h + np.random.gamma(gamma_shape, gamma_scale) * H, min_h, max_h))

    # Determine center sampling range
    if center_bounds is not None:
        cy_min, cy_max, cx_min, cx_max = center_bounds
        # Clamp to ensure ellipse stays within image
        cx_lo = max(rx, cx_min)
        cx_hi = min(W - rx, cx_max)
        cy_lo = max(ry, cy_min)
        cy_hi = min(H - ry, cy_max)
        cx = np.random.randint(cx_lo, max(cx_lo + 1, cx_hi)) if cx_hi > cx_lo else (cx_lo + cx_hi) // 2
        cy = np.random.randint(cy_lo, max(cy_lo + 1, cy_hi)) if cy_hi > cy_lo else (cy_lo + cy_hi) // 2
    else:
        cx = np.random.randint(rx, W - rx) if W > 2 * rx else W // 2
        cy = np.random.randint(ry, H - ry) if H > 2 * ry else H // 2

    y, x = np.ogrid[-cy:H - cy, -cx:W - cx]
    ellipse = (x * x) / (rx * rx + 1e-6) + (y * y) / (ry * ry + 1e-6) <= 1
    mask[ellipse] = 1.0

    # Gaussian blur with pre-computed kernel
    mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    mask_t = F.conv2d(mask_t, _SMOOTH_KERNEL, padding=_SMOOTH_KSIZE // 2)
    mask_t = torch.clamp(mask_t, 0, 1)
    return mask_t.squeeze(0).to(device)  # (1, H, W)


def _median_filter_2d(tensor, radius=5):
    """Apply 2D median filter with disk-shaped kernel.

    Args:
        tensor: (1, H, W) tensor in [0, 1] range
        radius: Disk radius for median filter

    Returns:
        Filtered tensor of same shape
    """
    # Convert to numpy for scipy median_filter
    arr = tensor.squeeze(0).cpu().numpy()  # (H, W)

    # Apply median filter with disk footprint
    # scipy's footprint is a boolean array, create disk-shaped
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    footprint = (x * x + y * y) <= radius * radius

    filtered = median_filter(arr, footprint=footprint)

    # Convert back to tensor
    return torch.from_numpy(filtered).float().unsqueeze(0).to(tensor.device)


def _median_blur_binary(mask, kernel_size=5):
    """Apply median blur to binary mask using cv2.medianBlur.

    This is used for label_mask computation in NSA to remove grain
    from the threshold operation.

    Args:
        mask: (1, H, W) tensor, expected to be binary (0 or 1)
        kernel_size: Kernel size for median blur (must be odd)

    Returns:
        Blurred binary mask as (1, H, W) tensor
    """
    if not HAS_CV2:
        # Fallback to scipy if cv2 not available
        return _median_filter_2d(mask, radius=kernel_size // 2)

    mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    blurred = cv2.medianBlur(mask_np, kernel_size)
    return torch.from_numpy(blurred.astype(np.float32) / 255.0).unsqueeze(0).to(mask.device)


def _make_disk_footprint(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def _median_filter_numpy(arr: np.ndarray, radius: int = 5) -> np.ndarray:
    return median_filter(arr, footprint=_make_disk_footprint(radius))


def _tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.uint8(np.round(arr * 255.0))


def _uint8_hwc_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor.to(device)


def _label_numpy_to_tensor(label: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(label.astype(np.float32)).permute(2, 0, 1).to(device)


def _official_patch_ex(
    ima_dest: np.ndarray,
    ima_src: np.ndarray,
    mode: int,
    width_bounds_pct,
    min_object_pct: Optional[float],
    min_overlap_pct: Optional[float],
    label_mode: str,
    skip_background,
    intensity_logistic_params,
    num_patches: int,
    resize_bounds,
):
    """Reference-aligned copy of NSA `patch_ex` for HWC uint8 RGB images."""
    same = False
    shift = True
    resize = True
    tol = 1
    gamma_params = DEFAULT_GAMMA_PARAMS
    verbose = False
    num_ellipses = None
    cutpaste_patch_generation = False

    if cutpaste_patch_generation:
        width_bounds_pct = None
        resize = False
        skip_background = None
        min_overlap_pct = None
        min_object_pct = None
        gamma_params = None
        num_patches = 1

    ima_src = ima_dest.copy() if same or (ima_src is None) else ima_src

    if skip_background is not None:
        if isinstance(skip_background, tuple):
            skip_background = [skip_background]
        src_object_mask = np.ones_like(ima_src[..., 0:1])
        dest_object_mask = np.ones_like(ima_dest[..., 0:1])
        for background, threshold in skip_background:
            src_object_mask &= np.uint8(np.abs(ima_src.mean(axis=-1, keepdims=True) - background) > threshold)
            dest_object_mask &= np.uint8(np.abs(ima_dest.mean(axis=-1, keepdims=True) - background) > threshold)
        src_object_mask[..., 0] = cv2.medianBlur(src_object_mask[..., 0], 7)
        dest_object_mask[..., 0] = cv2.medianBlur(dest_object_mask[..., 0], 7)
    else:
        src_object_mask = None
        dest_object_mask = None

    mask = np.zeros_like(ima_dest[..., 0:1])
    patchex = ima_dest.copy()
    coor_min_dim1, coor_max_dim1 = mask.shape[0] - 1, 0
    coor_min_dim2, coor_max_dim2 = mask.shape[1] - 1, 0

    factor = 1
    for i in range(num_patches):
        if i == 0 or np.random.randint(2) > 0:
            patchex, ((_coor_min_dim1, _coor_max_dim1), (_coor_min_dim2, _coor_max_dim2)), patch_mask = _official_single_patch(
                patchex,
                ima_src,
                dest_object_mask,
                src_object_mask,
                mode,
                label_mode,
                shift,
                resize,
                width_bounds_pct,
                gamma_params,
                min_object_pct,
                min_overlap_pct,
                factor,
                resize_bounds,
                num_ellipses,
                verbose,
                cutpaste_patch_generation,
            )
            if patch_mask is not None:
                mask[_coor_min_dim1:_coor_max_dim1, _coor_min_dim2:_coor_max_dim2] = patch_mask
                coor_min_dim1 = min(coor_min_dim1, _coor_min_dim1)
                coor_max_dim1 = max(coor_max_dim1, _coor_max_dim1)
                coor_min_dim2 = min(coor_min_dim2, _coor_min_dim2)
                coor_max_dim2 = max(coor_max_dim2, _coor_max_dim2)

    label_mask = np.uint8(
        np.mean(np.abs(1.0 * mask * ima_dest - 1.0 * mask * patchex), axis=-1, keepdims=True) > tol
    )
    label_mask[..., 0] = cv2.medianBlur(label_mask[..., 0], 5)

    if label_mode in {'logistic-intensity', 'intensity'}:
        k, x0 = intensity_logistic_params
        label = np.mean(
            np.abs(label_mask * ima_dest * 1.0 - label_mask * patchex * 1.0),
            axis=-1,
            keepdims=True,
        )
        label[..., 0] = sk_median(label[..., 0], sk_disk(5))
        if label_mode == 'logistic-intensity':
            label = label_mask / (1 + np.exp(-k * (label - x0)))
    elif label_mode == 'binary':
        label = label_mask
    else:
        raise ValueError(f'Unsupported label_mode: {label_mode}')

    return patchex, label.astype(np.float32)


def _official_single_patch(
    ima_dest: np.ndarray,
    ima_src: np.ndarray,
    dest_object_mask: Optional[np.ndarray],
    src_object_mask: Optional[np.ndarray],
    mode: int,
    label_mode: str,
    shift: bool,
    resize: bool,
    width_bounds_pct,
    gamma_params,
    min_object_pct: Optional[float],
    min_overlap_pct: Optional[float],
    factor: float,
    resize_bounds,
    num_ellipses,
    verbose: bool,
    cutpaste_patch_generation: bool,
):
    if cutpaste_patch_generation:
        raise NotImplementedError('cutpaste_patch_generation is not used for NSA alignment.')

    skip_background = (src_object_mask is not None) and (dest_object_mask is not None)
    dims = np.array(ima_dest.shape)
    min_width_dim1 = (width_bounds_pct[0][0] * dims[0]).round().astype(int)
    max_width_dim1 = (width_bounds_pct[0][1] * dims[0]).round().astype(int)
    min_width_dim2 = (width_bounds_pct[1][0] * dims[1]).round().astype(int)
    max_width_dim2 = (width_bounds_pct[1][1] * dims[1]).round().astype(int)

    if gamma_params is not None:
        shape, scale, lower_bound = gamma_params
        patch_width_dim1 = int(np.clip((lower_bound + np.random.gamma(shape, scale)) * dims[0], min_width_dim1, max_width_dim1))
        patch_width_dim2 = int(np.clip((lower_bound + np.random.gamma(shape, scale)) * dims[1], min_width_dim2, max_width_dim2))
    else:
        patch_width_dim1 = np.random.randint(min_width_dim1, max_width_dim1)
        patch_width_dim2 = np.random.randint(min_width_dim2, max_width_dim2)

    found_patch = False
    attempts = 0
    while not found_patch:
        center_dim1 = np.random.randint(min_width_dim1, dims[0] - min_width_dim1)
        center_dim2 = np.random.randint(min_width_dim2, dims[1] - min_width_dim2)

        coor_min_dim1 = np.clip(center_dim1 - patch_width_dim1, 0, dims[0])
        coor_min_dim2 = np.clip(center_dim2 - patch_width_dim2, 0, dims[1])
        coor_max_dim1 = np.clip(center_dim1 + patch_width_dim1, 0, dims[0])
        coor_max_dim2 = np.clip(center_dim2 + patch_width_dim2, 0, dims[1])

        if num_ellipses is not None:
            ellipse_min_dim1 = min_width_dim1
            ellipse_min_dim2 = min_width_dim2
            ellipse_max_dim1 = max(min_width_dim1 + 1, patch_width_dim1 // 2)
            ellipse_max_dim2 = max(min_width_dim2 + 1, patch_width_dim2 // 2)
            patch_mask = np.zeros((coor_max_dim1 - coor_min_dim1, coor_max_dim2 - coor_min_dim2), dtype=np.uint8)
            x = np.arange(patch_mask.shape[0]).reshape(-1, 1)
            y = np.arange(patch_mask.shape[1]).reshape(1, -1)
            for _ in range(num_ellipses):
                theta = np.random.uniform(0, np.pi)
                x0 = np.random.randint(0, patch_mask.shape[0])
                y0 = np.random.randint(0, patch_mask.shape[1])
                a = np.random.randint(ellipse_min_dim1, ellipse_max_dim1)
                b = np.random.randint(ellipse_min_dim2, ellipse_max_dim2)
                ellipse = (((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) / a) ** 2 + (((x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)) / b) ** 2 <= 1
                patch_mask |= ellipse
            patch_mask = patch_mask[..., None]
        else:
            patch_mask = np.ones((coor_max_dim1 - coor_min_dim1, coor_max_dim2 - coor_min_dim2, 1), dtype=np.uint8)

        if skip_background:
            background_area = np.sum(patch_mask & src_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2])
            patch_area = np.sum(patch_mask) if num_ellipses is not None else patch_mask.shape[0] * patch_mask.shape[1]
            found_patch = (background_area / max(patch_area, 1) > min_object_pct)
        else:
            found_patch = True
        attempts += 1
        if attempts == 200:
            if verbose:
                logger.warning('No suitable patch found.')
            return ima_dest.copy(), ((0, 0), (0, 0)), None

    src = ima_src[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]
    height, width, _ = src.shape
    if resize:
        lb, ub = resize_bounds
        scale = np.clip(np.random.normal(1, 0.5), lb, ub)
        new_height = np.clip(scale * height, min_width_dim1, max_width_dim1)
        new_width = np.clip(int(new_height / height * width), min_width_dim2, max_width_dim2)
        new_height = np.clip(int(new_width / width * height), min_width_dim1, max_width_dim1)
        src = cv2.resize(src, (new_width, new_height))
        height, width, _ = src.shape
        patch_mask = cv2.resize(patch_mask[..., 0], (width, height))[..., None]

    if skip_background:
        src_object_mask = cv2.resize(
            src_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2, 0],
            (width, height),
        )[..., None]

    if shift:
        found_center = False
        attempts = 0
        while not found_center:
            center_dim1 = np.random.randint(height // 2 + 1, ima_dest.shape[0] - height // 2 - 1)
            center_dim2 = np.random.randint(width // 2 + 1, ima_dest.shape[1] - width // 2 - 1)
            coor_min_dim1, coor_max_dim1 = center_dim1 - height // 2, center_dim1 + (height + 1) // 2
            coor_min_dim2, coor_max_dim2 = center_dim2 - width // 2, center_dim2 + (width + 1) // 2

            if skip_background:
                src_and_dest = dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] & src_object_mask & patch_mask
                src_mask_sum = np.sum(src_object_mask)
                found_center = (
                    src_mask_sum / max(patch_mask.shape[0] * patch_mask.shape[1], 1) > min_object_pct
                    and np.sum(src_and_dest) / max(src_mask_sum, 1) > min_overlap_pct
                )
            else:
                found_center = True
            attempts += 1
            if attempts == 200:
                if verbose:
                    logger.warning('No suitable center found. Dims were: %d %d', width, height)
                return ima_dest.copy(), ((0, 0), (0, 0)), None

    if skip_background:
        patch_mask &= src_object_mask | dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]

    if mode in [CV2_NORMAL_CLONE, CV2_MIXED_CLONE]:
        int_factor = np.uint8(np.ceil(factor * 255))
        if skip_background:
            patch_mask_scaled = int_factor * (
                patch_mask | ((1 - src_object_mask) & (1 - dest_object_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2]))
            )
        else:
            patch_mask_scaled = int_factor * patch_mask
        patch_mask_scaled[0], patch_mask_scaled[-1], patch_mask_scaled[:, 0], patch_mask_scaled[:, -1] = 0, 0, 0, 0
        center = (
            coor_max_dim2 - (coor_max_dim2 - coor_min_dim2) // 2,
            coor_min_dim1 + (coor_max_dim1 - coor_min_dim1) // 2,
        )
        if np.sum(patch_mask_scaled > 0) < 50:
            return ima_dest.copy(), ((0, 0), (0, 0)), None
        try:
            patchex = cv2.seamlessClone(src, ima_dest, patch_mask_scaled, center, mode)
        except cv2.error:
            if verbose:
                logger.warning('Tried bad interpolation mask.')
            return ima_dest.copy(), ((0, 0), (0, 0)), None
    else:
        raise ValueError(f'mode not supported {mode}')

    return patchex, ((coor_min_dim1, coor_max_dim1), (coor_min_dim2, coor_max_dim2)), patch_mask


def _poisson_blend(source, target, mask, use_mixed=False):
    """Poisson blending via cv2.seamlessClone, with alpha blending fallback.

    Args:
        source: (C, H, W) tensor in [0, 1].
        target: (C, H, W) tensor in [0, 1].
        mask: (1, H, W) tensor, binary-ish mask.
        use_mixed: Use MIXED_CLONE (for textures) vs NORMAL_CLONE (for objects).
    Returns:
        (C, H, W) blended tensor in [0, 1].
    """
    if HAS_CV2:
        try:
            src_np = (source.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            tgt_np = (target.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)

            if mask_np.sum() == 0:
                return target

            ys, xs = np.where(mask_np > 0)
            center = (int(xs.mean()), int(ys.mean()))

            clone_mode = cv2.MIXED_CLONE if use_mixed else cv2.NORMAL_CLONE
            result_np = cv2.seamlessClone(
                src_np, tgt_np, mask_np, center, clone_mode)
            result = torch.from_numpy(result_np.astype(np.float32) / 255.0)
            result = result.permute(2, 0, 1).to(source.device)
            return result.clamp(0, 1)
        except cv2.error:
            pass

    # Fallback: alpha blending with smooth mask
    beta = np.random.uniform(0.5, 0.9)
    result = target * (1 - mask * beta) + source * mask * beta
    return result.clamp(0, 1)


@MODELS.register_module()
class NSADetector(BaseADModel):
    """NSA: Natural Synthetic Anomalies anomaly detector.

    Implements the official NSA (logistic) variant from Schluter et al.,
    ECCV 2022:
    - logistic-intensity labels with BCE-style supervision
    - Cross-batch image buffer for diverse source images
    - Class-dependent Poisson blending (mixed gradients for textures only)
    - Images in [0,1] for blending, ImageNet-normalized for backbone

    Args:
        backbone: Backbone config or name for feature extraction.
        anomaly_ratio: Probability of generating anomaly per sample.
        seg_base_width: Base channel width for segmentation head.
        buffer_size: Size of the image reservoir when `source_sampling="reservoir"`.
        gaussian_sigma: Sigma for Gaussian smoothing of anomaly maps.
        use_logistic_labels: Whether to use logistic-intensity labels.
        source_sampling: `previous` for the official NSA source-image policy,
            `reservoir` to use the older cross-batch sampling ablation.
        median_filter_radius: Radius for median filter in label generation.
    """

    def __init__(self, backbone='resnet18', anomaly_ratio=1.0,
                 seg_base_width=64,
                 buffer_size=1000,
                 gaussian_sigma=0.0,
                 use_logistic_labels=True,
                 source_sampling='previous',
                 median_filter_radius=5,
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # Official MVTec logistic variant uses sigmoid + BCELoss.
        # We keep logits in the model and use BCEWithLogitsLoss for stability.
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.backbone_name = self._resolve_backbone_name(backbone)
        self.use_reference_arch = (self.backbone_name == 'resnet18')
        if self.use_reference_arch:
            self.model = NSAResNetEncDec(out_channels=1)
            self.layer0 = None
            self.layer1 = None
            self.layer2 = None
            self.layer3 = None
            self.seg_head = None
        else:
            if isinstance(backbone, dict):
                net = MODELS.build(backbone)
            else:
                net = MODELS.build(dict(type='RawBackbone', backbone_name=backbone))
            self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            ch = net.channel_dims
            self.seg_head = UNetSegHead(
                in_channels_list=[ch[0], ch[1], ch[2]],
                base_width=seg_base_width,
            )
            self.model = None

        self.anomaly_ratio = anomaly_ratio
        self.gaussian_sigma = gaussian_sigma
        self.use_logistic_labels = use_logistic_labels
        self.source_sampling = source_sampling
        self.median_filter_radius = median_filter_radius

        # ImageNet normalization constants for backbone feature extraction
        # Images arrive in [0,1] (using ScaleNormalizeAD), need ImageNet norm for backbone
        self.register_buffer('_img_mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('_img_std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                             persistent=False)

        self.buffer_size = buffer_size
        self.register_buffer('_image_buffer_count', torch.tensor(0))
        self._image_buffer = None
        self._previous_source = None

        # Current category name (set via data_samples for MIXED_CLONE detection)
        self._current_cls_name = None

        if self.source_sampling not in {'previous', 'reservoir'}:
            raise ValueError(f'Unsupported source_sampling: {source_sampling}')

        if not self.use_reference_arch:
            for module in [self.layer0, self.layer1, self.layer2, self.layer3]:
                for p in module.parameters():
                    p.requires_grad = False

    @staticmethod
    def _resolve_backbone_name(backbone) -> str:
        if isinstance(backbone, str):
            return backbone
        if isinstance(backbone, dict):
            if backbone.get('type') == 'RawBackbone':
                return str(backbone.get('backbone_name'))
            return str(backbone.get('backbone_name') or backbone.get('name') or backbone.get('type'))
        return str(backbone)

    def _get_width_bounds(self, cls_name):
        """Get per-category width bounds for patch generation."""
        if cls_name and cls_name in WIDTH_BOUNDS_PCT:
            return WIDTH_BOUNDS_PCT[cls_name]
        return DEFAULT_WIDTH_BOUNDS

    def _get_logistic_params(self, cls_name):
        """Get per-category logistic transform parameters (k, x0)."""
        if cls_name and cls_name in INTENSITY_LOGISTIC_PARAMS:
            return INTENSITY_LOGISTIC_PARAMS[cls_name]
        return DEFAULT_LOGISTIC_PARAMS

    def _get_num_patches(self, cls_name):
        """Get per-category number of patches."""
        if cls_name and cls_name in NUM_PATCHES:
            return NUM_PATCHES[cls_name]
        return DEFAULT_NUM_PATCHES

    def _get_background_params(self, cls_name):
        """Get background detection parameters (brightness_thresh, binary_thresh)."""
        if cls_name and cls_name in BACKGROUND_PARAMS:
            return BACKGROUND_PARAMS[cls_name]
        return None

    def _get_min_object_pct(self, cls_name):
        """Get minimum object percentage for patch generation."""
        if cls_name and cls_name in MIN_OBJECT_PCT:
            return MIN_OBJECT_PCT[cls_name]
        return None

    def _get_min_overlap_pct(self, cls_name):
        """Get minimum overlap percentage for source/destination objects."""
        if cls_name and cls_name in MIN_OVERLAP_PCT:
            return MIN_OVERLAP_PCT[cls_name]
        return None

    def _detect_object_mask(self, img, cls_name):
        """Detect object mask based on background brightness.

        Args:
            img: (C, H, W) tensor in [0, 1] range
            cls_name: Category name

        Returns:
            (1, H, W) binary mask where 1=object, 0=background, or None if no background detection
        """
        bg_params = self._get_background_params(cls_name)
        if bg_params is None:
            return None

        brightness_thresh, binary_thresh = bg_params

        # Convert to grayscale and scale to [0, 255]
        gray = torch.mean(img, dim=0) * 255.0  # (H, W)

        # Determine if bright or dark background
        mean_brightness = gray.mean().item()

        if mean_brightness > 127:  # bright background
            # Object is darker than background
            object_mask = (gray < brightness_thresh).float()
        else:  # dark background
            # Object is brighter than background
            object_mask = (gray > brightness_thresh).float()

        # Apply binary threshold to clean up mask
        object_mask = (object_mask * 255 > binary_thresh).float()

        return object_mask.unsqueeze(0)  # (1, H, W)

    def _normalize_for_backbone(self, x):
        """Apply ImageNet normalization for backbone feature extraction."""
        return (x - self._img_mean) / self._img_std

    def _init_buffer(self, sample_img):
        """Initialize image buffer."""
        C, H, W = sample_img.shape[1:]
        self._image_buffer = torch.zeros(self.buffer_size, C, H, W,
                                          device=sample_img.device, dtype=sample_img.dtype)

    def _update_buffer(self, batch):
        """Update buffer with new images using reservoir sampling."""
        if self._image_buffer is None:
            self._init_buffer(batch)

        B = batch.shape[0]
        count = self._image_buffer_count.item()

        for i in range(B):
            if count < self.buffer_size:
                self._image_buffer[count] = batch[i]
                count += 1
            else:
                j = np.random.randint(0, count)
                if j < self.buffer_size:
                    self._image_buffer[j] = batch[i]
                count += 1

        self._image_buffer_count.fill_(count)

    def _get_source_image(self, batch, idx):
        """Get source image using the configured source sampling policy."""
        if self.source_sampling == 'previous':
            if idx > 0:
                return batch[idx - 1]
            if self._previous_source is not None:
                return self._previous_source.to(batch.device, dtype=batch.dtype)
            if batch.shape[0] > 1:
                return batch[1]
            return batch[idx]

        B = batch.shape[0]
        count = self._image_buffer_count.item()
        if count >= B * 2:
            j = np.random.randint(0, min(count, self.buffer_size))
            return self._image_buffer[j]
        j = np.random.randint(0, B)
        while j == idx and B > 1:
            j = np.random.randint(0, B)
        return batch[j]

    @torch.no_grad()
    def extract_features(self, x):
        """Extract multi-scale features. x should be ImageNet-normalized."""
        x = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3

    def _generate_nsa(self, batch):
        """Generate Natural Synthetic Anomalies for a batch.

        Port of the official NSA `patch_ex` augmentation and label generation.

        Args:
            batch: (B, C, H, W) tensor in [0, 1] range.

        Returns:
            augmented: Augmented images in [0, 1]
            labels: Soft logistic-intensity labels in [0, 1]
        """
        if self.source_sampling == 'reservoir':
            self._update_buffer(batch)

        B, C, H, W = batch.shape
        device = batch.device
        augmented = batch.clone()
        labels = torch.zeros(B, 1, H, W, device=device)

        # Get per-category parameters
        cls_name = self._current_cls_name
        width_bounds = self._get_width_bounds(cls_name)
        logistic_k, logistic_x0 = self._get_logistic_params(cls_name)
        n_patches = self._get_num_patches(cls_name)
        min_overlap_pct = self._get_min_overlap_pct(cls_name)

        skip_background = self._get_background_params(cls_name)
        min_obj_pct = self._get_min_object_pct(cls_name)
        use_mixed = (cls_name is not None and cls_name in TEXTURE_CATEGORIES)
        blend_mode = CV2_MIXED_CLONE if use_mixed else CV2_NORMAL_CLONE
        resize_bounds = TEXTURE_RESIZE_BOUNDS if use_mixed else DEFAULT_RESIZE_BOUNDS
        label_mode = 'logistic-intensity' if self.use_logistic_labels else 'intensity'

        for i in range(B):
            if np.random.random() > self.anomaly_ratio:
                continue

            patched, label = _official_patch_ex(
                ima_dest=_tensor_to_uint8_hwc(batch[i]),
                ima_src=_tensor_to_uint8_hwc(self._get_source_image(batch, i)),
                mode=blend_mode,
                width_bounds_pct=width_bounds,
                min_object_pct=min_obj_pct,
                min_overlap_pct=min_overlap_pct,
                label_mode=label_mode,
                skip_background=skip_background,
                intensity_logistic_params=(logistic_k, logistic_x0),
                num_patches=n_patches,
                resize_bounds=resize_bounds,
            )
            augmented[i] = _uint8_hwc_to_tensor(patched, device=device)
            labels[i] = _label_numpy_to_tensor(label, device=device)

        if self.source_sampling == 'previous':
            self._previous_source = batch[-1].detach().cpu()

        return augmented, labels

    @staticmethod
    def _get_object_bbox(obj_mask):
        """Get bounding box of object mask as (y_min, y_max, x_min, x_max)."""
        mask_2d = obj_mask.squeeze(0)  # (H, W)
        ys = torch.where(mask_2d.sum(dim=1) > 0)[0]
        xs = torch.where(mask_2d.sum(dim=0) > 0)[0]
        if len(ys) == 0 or len(xs) == 0:
            return None
        return (ys[0].item(), ys[-1].item() + 1, xs[0].item(), xs[-1].item() + 1)

    def _generate_patch_mask_with_skip_bg(self, H, W, width_bounds, dest_obj_mask, src_obj_mask,
                                           min_obj_pct, device, max_attempts=50):
        """Generate patch mask with skip_background logic.

        Uses guided center sampling within the object bounding box to increase
        acceptance rate and avoid timeouts for strict overlap requirements.

        Args:
            H, W: Spatial dimensions
            width_bounds: Per-category bounds for patch sizes ((h_min, h_max), (w_min, w_max))
            dest_obj_mask: Object mask for destination image (1, H, W) or None
            src_obj_mask: Object mask for source image (1, H, W) or None
            min_obj_pct: Minimum percentage of object required in patch
            device: torch device
            max_attempts: Maximum attempts to find valid patch

        Returns:
            (1, H, W) mask tensor (always returns a valid mask)
        """
        if dest_obj_mask is None:
            return _generate_smooth_mask(H, W, width_bounds_pct=width_bounds, device=device)

        # Compute object bounding box for guided center sampling
        bbox = self._get_object_bbox(dest_obj_mask)
        center_bounds = (bbox[0], bbox[1], bbox[2], bbox[3]) if bbox is not None else None

        for _ in range(max_attempts):
            mask = _generate_smooth_mask(H, W, width_bounds_pct=width_bounds, device=device,
                                         center_bounds=center_bounds)

            mask_area = mask.sum()
            if mask_area < 1:
                continue

            object_in_patch = (mask * dest_obj_mask).sum()
            obj_pct = object_in_patch / mask_area

            if obj_pct >= min_obj_pct:
                return mask

        # Fallback: intersect mask with object region to guarantee object overlap
        mask = _generate_smooth_mask(H, W, width_bounds_pct=width_bounds, device=device,
                                     center_bounds=center_bounds)
        return mask * dest_obj_mask

    def _smooth_anomaly_map(self, score_map):
        """Apply Gaussian smoothing to anomaly score map."""
        if self.gaussian_sigma <= 0:
            return score_map
        ksize = int(4 * self.gaussian_sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        x_coord = torch.arange(ksize, device=score_map.device).float() - ksize // 2
        kernel_1d = torch.exp(-x_coord ** 2 / (2 * self.gaussian_sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        # Separable 2D Gaussian
        score_map = F.conv2d(score_map, kernel_1d.view(1, 1, -1, 1),
                             padding=(ksize // 2, 0))
        score_map = F.conv2d(score_map, kernel_1d.view(1, 1, 1, -1),
                             padding=(0, ksize // 2))
        return score_map

    @staticmethod
    def _center_crop_to_size(inputs: torch.Tensor, crop_size: int) -> torch.Tensor:
        h, w = inputs.shape[-2:]
        top = max((h - crop_size) // 2, 0)
        left = max((w - crop_size) // 2, 0)
        return inputs[..., top:top + crop_size, left:left + crop_size]

    @staticmethod
    def _pad_back_to_size(score_map: torch.Tensor, target_hw) -> torch.Tensor:
        target_h, target_w = target_hw
        pad_h = max(target_h - score_map.shape[-2], 0)
        pad_w = max(target_w - score_map.shape[-1], 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(score_map, (pad_left, pad_right, pad_top, pad_bottom))

    def _forward_logits(self, inputs: torch.Tensor, cls_name: Optional[str], mode: str) -> torch.Tensor:
        inputs_norm = self._normalize_for_backbone(inputs)
        if self.use_reference_arch:
            if mode == 'predict' and cls_name in OBJECT_CATEGORIES and tuple(inputs.shape[-2:]) == (256, 256):
                cropped = self._center_crop_to_size(inputs_norm, 224)
                logits = self.model(cropped)
                return self._pad_back_to_size(logits, inputs.shape[-2:])
            return self.model(inputs_norm)

        f1, f2, f3 = self.extract_features(inputs_norm)
        logits = self.seg_head(f1, f2, f3)
        if logits.shape[-2:] != inputs.shape[-2:]:
            logits = F.interpolate(logits, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def _predict_score_map(self, inputs: torch.Tensor, cls_name: Optional[str]):
        """Return pixel map and image score using the official NSA eval path."""
        inputs_norm = self._normalize_for_backbone(inputs)
        if self.use_reference_arch and cls_name in OBJECT_CATEGORIES and tuple(inputs.shape[-2:]) == (256, 256):
            cropped = self._center_crop_to_size(inputs_norm, 224)
            logits_crop = self.model(cropped)
            score_map_crop = torch.sigmoid(logits_crop)
            img_scores = self._compute_image_scores_from_map(
                score_map_crop,
                cls_name=cls_name,
                input_hw=tuple(cropped.shape[-2:]),
                mode='reference_mean',
            )
            score_map = self._pad_back_to_size(score_map_crop, inputs.shape[-2:])
            return score_map, img_scores

        logits = self.model(inputs_norm) if self.use_reference_arch else self._forward_logits(inputs, cls_name, mode='predict')
        score_map = torch.sigmoid(logits)
        img_scores = self._compute_image_scores_from_map(
            score_map,
            cls_name=cls_name,
            input_hw=tuple(inputs.shape[-2:]),
            mode='reference_mean',
        )
        return score_map, img_scores

    @classmethod
    def _extract_image_score_region(
        cls,
        score_map: torch.Tensor,
        cls_name: Optional[str],
        input_hw: Optional[tuple[int, int]],
    ) -> torch.Tensor:
        """Return the region used for image-level scoring."""
        if cls_name in OBJECT_CATEGORIES and input_hw == (256, 256):
            return cls._center_crop_to_size(score_map, 224)
        return score_map

    @classmethod
    def _compute_image_scores_from_map(
        cls,
        score_map: torch.Tensor,
        cls_name: Optional[str],
        input_hw: Optional[tuple[int, int]],
        mode: str = 'reference_mean',
        topk_ratio: float = 0.01,
    ) -> torch.Tensor:
        """Compute image-level scores from an anomaly map under different aggregations."""
        if score_map.ndim != 4:
            raise ValueError(f'Expected score_map of shape (B, 1, H, W), got {tuple(score_map.shape)}')

        if mode == 'full_mean':
            region = score_map
        else:
            region = cls._extract_image_score_region(score_map, cls_name=cls_name, input_hw=input_hw)

        flat = region.flatten(1)
        if mode in {'reference_mean', 'full_mean'}:
            return flat.mean(dim=1)
        if mode == 'max':
            return flat.max(dim=1).values
        if mode == 'topk_mean':
            k = max(1, min(flat.shape[1], int(round(flat.shape[1] * topk_ratio))))
            return flat.topk(k, dim=1).values.mean(dim=1)
        raise ValueError(f'Unsupported NSA image score mode: {mode}')

    @staticmethod
    def _collect_target_masks(data_samples, device: torch.device) -> Optional[torch.Tensor]:
        if not data_samples:
            return None
        masks = []
        for sample in data_samples:
            if not hasattr(sample, 'gt_mask'):
                return None
            mask = sample.gt_mask
            if mask is None:
                return None
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            masks.append(mask.float().to(device))
        if not masks:
            return None
        return torch.stack(masks, dim=0)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        cls_name = None

        # Extract category name for MIXED_CLONE detection
        if data_samples is not None and len(data_samples) > 0:
            cls_name = getattr(data_samples[0], 'cls_name', None)
            if cls_name is not None:
                self._current_cls_name = cls_name

        if mode == 'loss':
            target_masks = self._collect_target_masks(data_samples, inputs.device)
            if target_masks is None:
                augmented, target_masks = self._generate_nsa(inputs)
                loss_inputs = augmented
            else:
                loss_inputs = inputs
            pred = self._forward_logits(loss_inputs, cls_name, mode='loss')
            target_labels = F.interpolate(target_masks, size=pred.shape[-2:],
                                           mode='bilinear', align_corners=False)
            loss = self.loss_fn(pred, target_labels)
            return {'loss': loss}

        elif mode == 'predict':
            score_map, img_scores = self._predict_score_map(inputs, cls_name)
            # Gaussian smoothing
            score_map = self._smooth_anomaly_map(score_map)

            return build_predict_results(data_samples, img_scores, score_map)

        return self._forward_logits(inputs, cls_name, mode='tensor')

    def train(self, mode=True):
        super().train(mode)
        if not self.use_reference_arch:
            self.layer0.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
        return self
