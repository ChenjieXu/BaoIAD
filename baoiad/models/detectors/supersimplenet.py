"""SuperSimpleNet: Simple and Reliable Anomaly Detection (CVPR 2024).

Reimplementation with:
- WideResNet-50-2 feature extractor with multi-scale upscaling
- 1x1 conv feature adapter
- Segmentation-Detection dual-head module (anomaly map + classification score)
- Perlin noise based synthetic anomaly generation during training
- Focal loss + truncated L1 loss for segmentation, focal loss for classification
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.models.predict_utils import build_predict_results
from baoiad.models.losses.focal_loss import sigmoid_focal_loss
from baoiad.models.base_ad_model import DiscriminatorADModel

from baoiad.registry import MODELS


# ─── Loss ───────────────────────────────────────────────────────────────────

class SSNLoss(nn.Module):
    """SuperSimpleNet loss: focal + truncated L1 for segmentation, focal for classification."""

    def __init__(self, truncation_term: float = 0.5):
        super().__init__()
        self.focal_loss = partial(sigmoid_focal_loss, alpha=-1, gamma=4.0, reduction="mean")
        self.th = truncation_term

    def trunc_l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        normal_scores = pred[target == 0]
        anomalous_scores = pred[target > 0]
        true_loss = torch.clip(normal_scores + self.th, min=0)
        fake_loss = torch.clip(-anomalous_scores + self.th, min=0)
        true_loss = true_loss.mean() if len(true_loss) else torch.tensor(0.0, device=pred.device)
        fake_loss = fake_loss.mean() if len(fake_loss) else torch.tensor(0.0, device=pred.device)
        return true_loss + fake_loss

    def forward(self, pred_map, pred_score, target_mask, target_label):
        map_focal = self.focal_loss(pred_map, target_mask)
        map_trunc_l1 = self.trunc_l1_loss(pred_map, target_mask)
        score_focal = self.focal_loss(pred_score, target_label)
        return map_focal + map_trunc_l1 + score_focal


# ─── Gaussian blur ──────────────────────────────────────────────────────────

class GaussianBlur2d(nn.Module):
    def __init__(self, sigma: float, channels: int = 1):
        super().__init__()
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.padding = kernel_size // 2
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-0.5 * x ** 2 / sigma ** 2)
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.expand(channels, 1, -1, -1).contiguous()
        self.register_buffer('kernel', kernel_2d)
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)


# ─── Feature Extractor ──────────────────────────────────────────────────────

class UpscalingFeatureExtractor(nn.Module):
    """WideResNet-50-2 feature extractor with multi-scale upscaling."""

    def __init__(self, backbone='wide_resnet50_2', layers=('layer2', 'layer3'), patch_size: int = 3):
        super().__init__()
        # backbone built via MODELS.build(dict(type='RawBackbone', ...))
        if isinstance(backbone, dict):
            wrn = MODELS.build(backbone)
        else:
            wrn = MODELS.build(dict(type='RawBackbone', backbone_name=backbone))
        self.stem = nn.Sequential(wrn.conv1, wrn.bn1, wrn.relu, wrn.maxpool)
        self.layer1 = wrn.layer1
        self.layer2 = wrn.layer2
        self.layer3 = wrn.layer3
        self.layer4 = wrn.layer4
        self._layers = layers
        ch = wrn.channel_dims
        self._layer_map = {
            'layer1': (self.layer1, ch[0]),
            'layer2': (self.layer2, ch[1]),
            'layer3': (self.layer3, ch[2]),
            'layer4': (self.layer4, ch[3]),
        }
        self.channels = sum(self._layer_map[layer_name][1] for layer_name in layers)
        self.pooler = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)

        # Freeze
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        out = self.stem(x)
        layer_outputs = {}
        prev = out
        for name in ('layer1', 'layer2', 'layer3', 'layer4'):
            module = self._layer_map[name][0]
            prev = module(prev)
            if name in self._layers:
                layer_outputs[name] = prev

        features = list(layer_outputs.values())
        _, _, h, w = features[0].shape
        feature_map = []
        for feat in features:
            resized = F.interpolate(feat, size=(h * 2, w * 2), mode='bilinear', align_corners=False)
            feature_map.append(resized)
        feature_map = torch.cat(feature_map, dim=1)
        return self.pooler(feature_map)


# ─── Feature Adapter ────────────────────────────────────────────────────────

class FeatureAdapter(nn.Module):
    def __init__(self, channel_dim: int):
        super().__init__()
        self.projection = nn.Conv2d(channel_dim, channel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.projection.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


# ─── Segmentation-Detection Module ──────────────────────────────────────────

class SegmentationDetectionModule(nn.Module):
    """Dual-head: segmentation anomaly map + classification anomaly score."""

    def __init__(self, channel_dim: int, stop_grad: bool = True):
        super().__init__()
        self.stop_grad = stop_grad

        self.seg_head = nn.Sequential(
            nn.Conv2d(channel_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1, bias=False),
        )

        self.map_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dec_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # cls_conv takes cls_features (channel_dim) + anomaly_map (1) = channel_dim + 1
        self.cls_conv = nn.Sequential(
            nn.Conv2d(channel_dim + 1, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.cls_fc = nn.Linear(128 * 2 + 2, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)

    def forward(self, seg_features: torch.Tensor, cls_features: torch.Tensor):
        """Forward pass with separate seg and cls features.

        Args:
            seg_features: Features for segmentation head (perturbed adapted features during training)
            cls_features: Features for classification head (perturbed raw features during training)
        """
        ano_map = self.seg_head(seg_features)

        map_dec_copy = ano_map.detach() if self.stop_grad else ano_map
        mask_cat = torch.cat((cls_features, map_dec_copy), dim=1)
        dec_out = self.cls_conv(mask_cat)

        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)

        map_max = self.map_max_pool(ano_map)
        if self.stop_grad:
            map_max = map_max.detach()
        map_avg = self.map_avg_pool(ano_map)
        if self.stop_grad:
            map_avg = map_avg.detach()

        dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze(-1).squeeze(-1)
        ano_score = self.cls_fc(dec_cat).squeeze(-1)
        return ano_map, ano_score


# ─── Anomaly Generator (official-style Perlin noise) ────────────────────────


def _generate_perlin_noise(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Generate Perlin noise matching anomalib's torch implementation."""
    min_scale, max_scale = 0, 6
    scalex = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()
    scaley = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()

    pad_h = AnomalyGenerator._next_power_2(height)
    pad_w = AnomalyGenerator._next_power_2(width)
    scalex = min(scalex, pad_h)
    scaley = min(scaley, pad_w)
    delta = (scalex / pad_h, scaley / pad_w)
    repeats = (pad_h // scalex, pad_w // scaley)
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, scalex, delta[0], device=device),
                torch.arange(0, scaley, delta[1], device=device),
                indexing='ij',
            ),
            dim=-1,
        )
        % 1
    )

    angles = 2 * torch.pi * torch.rand(int(scalex) + 1, int(scaley) + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1, slice2) -> torch.Tensor:
        return (
            gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]]
            .repeat_interleave(int(repeats[0]), 0)
            .repeat_interleave(int(repeats[1]), 1)
        )

    def dot(grad: torch.Tensor, shift) -> torch.Tensor:
        return (
            torch.stack(
                (grid[:pad_h, :pad_w, 0] + shift[0], grid[:pad_h, :pad_w, 1] + shift[1]),
                dim=-1,
            )
            * grad[:pad_h, :pad_w]
        ).sum(dim=-1)

    n00 = dot(tile_grads((0, -1), (0, -1)), (0, 0))
    n10 = dot(tile_grads((1, None), (0, -1)), (-1, 0))
    n01 = dot(tile_grads((0, -1), (1, None)), (0, -1))
    n11 = dot(tile_grads((1, None), (1, None)), (-1, -1))

    def fade(t: torch.Tensor) -> torch.Tensor:
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    t = fade(grid[:pad_h, :pad_w])
    noise = torch.sqrt(torch.tensor(2.0, device=device)) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]),
        torch.lerp(n01, n11, t[..., 0]),
        t[..., 1],
    )
    return noise[:height, :width]

class AnomalyGenerator(nn.Module):
    """Generate synthetic anomalies using anomalib-style Perlin noise on features."""

    def __init__(self, noise_mean: float = 0.0, noise_std: float = 0.015, threshold: float = 0.2):
        super().__init__()
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.threshold = threshold

    @staticmethod
    def _next_power_2(n: int) -> int:
        return 1 << (n - 1).bit_length()

    def _generate_perlin_mask(self, b: int, h: int, w: int, device) -> torch.Tensor:
        """Generate binarized random masks matching anomalib's logic."""
        masks = []
        for _ in range(b):
            ph, pw = self._next_power_2(h), self._next_power_2(w)
            noise = _generate_perlin_noise(ph, pw, device=device)
            if not (noise > self.threshold).any():
                noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
                noise = (noise * 2) - 1
            noise = F.interpolate(
                noise.reshape(1, 1, ph, pw),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )
            mask = torch.where(noise > self.threshold, torch.ones_like(noise), torch.zeros_like(noise))
            if torch.rand(1).item() > 0.5:
                mask = torch.zeros_like(mask)
            masks.append(mask)
        return torch.cat(masks, dim=0)

    def forward(self, input_features, adapted_features, mask, labels):
        """Generate anomaly on both raw and adapted features with the same noise.

        Args:
            input_features: Raw features from backbone (can be None if adapt_cls_features=True)
            adapted_features: Adapted features after feature adapter
            mask: GT masks
            labels: GT labels

        Returns:
            perturbed_feat: Perturbed raw features (or None if input_features is None)
            perturbed_adapt: Perturbed adapted features
            mask: Updated GT masks
            labels: Updated GT labels
        """
        b, _, h, w = adapted_features.shape

        # Duplicate batch
        adapted_features = torch.cat((adapted_features, adapted_features))
        mask = torch.cat((mask, mask))
        labels = torch.cat((labels, labels))
        if input_features is not None:
            input_features = torch.cat((input_features, input_features))

        noise = torch.normal(
            mean=self.noise_mean, std=self.noise_std,
            size=adapted_features.shape, device=adapted_features.device, requires_grad=False,
        )

        noise_mask = torch.ones(b * 2, 1, h, w, device=adapted_features.device, requires_grad=False)
        noise_mask = noise_mask * (1 - mask)
        perlin_mask = self._generate_perlin_mask(b * 2, h, w, adapted_features.device)
        noise_mask = noise_mask * perlin_mask

        mask = torch.where((mask + noise_mask) > 0, torch.ones_like(mask), torch.zeros_like(mask))

        new_anomalous = noise_mask.reshape(b * 2, -1).any(dim=1).float()
        labels = torch.where((labels + new_anomalous) > 0, torch.ones_like(labels), torch.zeros_like(labels))

        # Apply same noise to both raw and adapted features
        perturbed_adapt = adapted_features + noise * noise_mask
        perturbed_feat = input_features + noise * noise_mask if input_features is not None else None

        return perturbed_feat, perturbed_adapt, mask, labels


# ─── Anomaly Map Generator ──────────────────────────────────────────────────

class AnomalyMapGenerator(nn.Module):
    def __init__(self, sigma: float = 4.0):
        super().__init__()
        self.smoothing = GaussianBlur2d(sigma=sigma)

    def forward(self, out_map: torch.Tensor, final_size: tuple) -> torch.Tensor:
        anomaly_map = F.interpolate(out_map, size=final_size, mode='bilinear', align_corners=False)
        return self.smoothing(anomaly_map)


# ─── SuperSimpleNet Detector ────────────────────────────────────────────────

@MODELS.register_module(force=True)
class SuperSimpleNetDetector(DiscriminatorADModel):
    """SuperSimpleNet anomaly detector.

    Args:
        backbone (str): Backbone name (unused, always WRN-50-2).
        layers (list[str]): Feature extraction layers.
        perlin_threshold (float): Perlin noise binarization threshold.
        stop_grad (bool): Stop gradient from classification to segmentation head.
        sigma (float): Gaussian blur sigma for anomaly map smoothing.
        adapt_cls_features (bool): Whether to adapt classification features.
            False (default, JIMS extension): cls uses raw features.
            True (ICPR original): cls uses adapted features.
    """

    def __init__(
        self,
        backbone: str = 'wide_resnet50_2',
        layers: list = None,
        perlin_threshold: float = 0.2,
        stop_grad: bool = True,
        sigma: float = 4.0,
        adapt_cls_features: bool = False,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if layers is None:
            layers = ['layer2', 'layer3']

        self.adapt_cls_features = adapt_cls_features
        self.feature_extractor = UpscalingFeatureExtractor(backbone=backbone, layers=layers)
        channels = self.feature_extractor.channels
        self.adaptor = FeatureAdapter(channels)
        self.segdec = SegmentationDetectionModule(channel_dim=channels, stop_grad=stop_grad)
        self.anomaly_generator = AnomalyGenerator(threshold=perlin_threshold)
        self.anomaly_map_generator = AnomalyMapGenerator(sigma=sigma)
        self.loss_fn = SSNLoss()

    @staticmethod
    def _downsample_mask(masks: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        masks = masks.float()
        masks = F.interpolate(masks.unsqueeze(1), size=(feat_h, feat_w), mode='bilinear', align_corners=False)
        return torch.where(masks < 0.5, torch.zeros_like(masks), torch.ones_like(masks))

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        output_size = inputs.shape[-2:]
        features = self.feature_extractor(inputs)
        adapted = self.adaptor(features)

        if mode == 'loss':
            # Get masks and labels from data_samples
            B = inputs.shape[0]
            device = inputs.device
            if data_samples is not None:
                masks = torch.stack([
                    getattr(ds, 'gt_mask', torch.zeros(1, *output_size, device=device)).squeeze()
                    for ds in data_samples
                ]).to(device)
                labels = torch.tensor([
                    getattr(ds, 'gt_label', 0)
                    for ds in data_samples
                ], dtype=torch.float32, device=device)
            else:
                masks = torch.zeros(B, *output_size, device=device)
                labels = torch.zeros(B, dtype=torch.float32, device=device)

            masks = self._downsample_mask(masks, *features.shape[-2:])
            labels = labels.float()

            # Pass both raw and adapted features to anomaly generator
            # When adapt_cls_features=True, only adapted features get noise (input_features=None)
            input_feats = None if self.adapt_cls_features else features
            perturbed_feat, perturbed_adapt, masks, labels = self.anomaly_generator(
                input_feats, adapted, masks, labels
            )

            # seg always uses perturbed adapted features
            seg_feats = perturbed_adapt
            # cls uses perturbed adapted (ICPR) or perturbed raw (JIMS extension)
            cls_feats = perturbed_adapt if self.adapt_cls_features else perturbed_feat

            anomaly_map, anomaly_score = self.segdec(seg_feats, cls_feats)
            loss = self.loss_fn(anomaly_map, anomaly_score, masks, labels)
            return {'loss': loss}

        elif mode == 'predict':
            # seg uses adapted features
            seg_feats = adapted
            # cls uses adapted (ICPR) or raw features (JIMS extension, default)
            cls_feats = adapted if self.adapt_cls_features else features

            anomaly_map, anomaly_score = self.segdec(seg_feats, cls_feats)
            anomaly_map = self.anomaly_map_generator(anomaly_map, final_size=output_size)

            # Apply sigmoid activation (matching anomalib reference)
            anomaly_score = anomaly_score.sigmoid()
            anomaly_map = anomaly_map.sigmoid()
            flat_maps = anomaly_map.view(anomaly_map.shape[0], -1)
            anomaly_score_mean = flat_maps.mean(dim=1)
            anomaly_score_max = flat_maps.max(dim=1).values

            return build_predict_results(
                data_samples,
                anomaly_score,
                anomaly_map,
                extra_scores={
                    'pred_score_mean': anomaly_score_mean,
                    'pred_score_max': anomaly_score_max,
                },
            )

        else:  # tensor
            return adapted

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self
