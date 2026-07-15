"""DifferNet: Semi-Supervised Defect Detection via Normalizing Flow (WACV 2021).

Paper: Rudolph et al., "Same Same But DifferNet: Semi-Supervised Defect
Detection with Normalizing Flows", WACV 2021.

Uses AlexNet backbone (paper's choice) with multi-scale GAP features fed to
a fully-connected normalizing flow. **Image-level detection only** — the
original paper does not produce pixel-level anomaly maps. A uniform placeholder
map is provided for API compatibility.

Key features from original paper:
1. Multi-scale input: 448x448, 224x224, 112x112
2. Multi-transform testing: 64 rotations + contrast/brightness augmentations
3. Only the final feature map of AlexNet is used (256-dim after GAP per scale)
4. Multi-transform training: n_transforms augmented copies per image per batch
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import FlowBasedADModel


class SubnetFC(nn.Module):
    """4-layer FC subnet matching original DifferNet's F_fully_connected.

    Original has 3 hidden layers: fc1 -> fc2 -> fc2b -> fc3
    """

    def __init__(self, size_in, size, internal_size=None, dropout=0.0):
        super().__init__()
        if internal_size is None:
            internal_size = 2048  # Match official config.py: fc_internal = 2048
        self.bn = nn.BatchNorm1d(size_in)
        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc2b = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, size)

    def forward(self, x):
        # Note: bn is defined but NOT used in forward (matches reference implementation)
        # Reference: ref/differnet/freia_funcs.py F_fully_connected.forward()
        x = F.relu(self.d1(self.fc1(x)))
        x = F.relu(self.d2(self.fc2(x)))
        x = F.relu(self.d2b(self.fc2b(x)))
        return self.fc3(x)


class PermuteLayer(nn.Module):
    """Fixed random permutation layer (matches reference implementation)."""

    def __init__(self, dims_in, seed):
        super().__init__()
        self.in_channels = dims_in[0][0]

        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.in_channels)

        perm_inv = np.zeros_like(perm)
        for i, p in enumerate(perm):
            perm_inv[p] = i

        self.register_buffer('perm', torch.LongTensor(perm))
        self.register_buffer('perm_inv', torch.LongTensor(perm_inv))

    def forward(self, x, rev=False):
        if not rev:
            return [x[0][:, self.perm]]
        else:
            return [x[0][:, self.perm_inv]]

    def jacobian(self, x, rev=False):
        return 0.0

    def output_dims(self, input_dims):
        return input_dims


class GlowCouplingLayer(nn.Module):
    """Glow-style coupling layer with atan clamping (matches reference).

    The key difference from FrEIA's AllInOneBlock is the clamping function:
    - Reference: clamp * 0.636 * atan(s / clamp)  (0.636 ≈ 2/π)
    - FrEIA: tanh-based clamping

    The atan clamping provides smoother saturation behavior.
    """

    def __init__(self, dims_in, F_class=SubnetFC, F_args={}, clamp=5.0):
        super().__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp

        # s1 predicts scale/translation for x2 from x1
        # s2 predicts scale/translation for x1 from x2
        self.s1 = F_class(self.split_len1, self.split_len2 * 2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1 * 2, **F_args)

    def e(self, s):
        return torch.exp(self.log_e(s))

    def log_e(self, s):
        # 0.636 ≈ 2/π, scales atan output from [-π/2, π/2] to ~[-1, 1]
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 = self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
        else:
            # Reverse pass: names of x and y are swapped
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)

        y = torch.cat((y1, y2), 1)
        y = torch.clamp(y, -1e6, 1e6)
        return [y]

    def jacobian(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 = self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
        else:
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]

        jac = (torch.sum(self.log_e(s1), dim=1) + torch.sum(self.log_e(s2), dim=1))
        for i in range(self.ndims - 1):
            jac = torch.sum(jac, dim=1)

        return jac

    def output_dims(self, input_dims):
        return input_dims


class FlowSequence(nn.Module):
    """Sequence of coupling and permutation layers matching reference implementation.

    Returns (z, log_jac_det) tuple to match FrEIA's SequenceINN interface.
    """

    def __init__(self, total_dim, n_coupling_blocks, clamp, fc_internal_size=2048):
        super().__init__()
        self.layers = nn.ModuleList()
        self.total_dim = total_dim

        dims_in = [(total_dim,)]

        for k in range(n_coupling_blocks):
            # Fixed random permutation seeded by block index
            self.layers.append(PermuteLayer(dims_in, seed=k))
            # Coupling layer with atan clamping
            # SubnetFC uses default internal_size = 2 * size (matching reference)
            self.layers.append(GlowCouplingLayer(
                dims_in,
                F_class=SubnetFC,
                F_args=dict(internal_size=fc_internal_size),
                clamp=clamp
            ))

    def forward(self, x, rev=False):
        """Forward pass returning (z, log_jac_det) to match FrEIA interface."""
        # Input is a tensor, wrap in list for layer interface
        current = [x]
        log_jac = 0.0

        if rev:
            for layer in reversed(self.layers):
                log_jac = log_jac + layer.jacobian(current, rev=True)
                current = layer(current, rev=True)
        else:
            for layer in self.layers:
                log_jac = log_jac + layer.jacobian(current, rev=False)
                current = layer(current, rev=False)

        # current is a list with one tensor
        return current[0], log_jac


@MODELS.register_module(force=True)
class DifferNetDetector(FlowBasedADModel):
    """DifferNet anomaly detector using normalizing flows on AlexNet features.

    NOTE: This is an image-level only method. The anomaly map output is a
    uniform placeholder (constant value = image score) for API compatibility.

    Args:
        backbone: Backbone name (default: 'alexnet' per paper).
        n_coupling_blocks: Number of coupling blocks in the flow.
        clamp: Clamping value for affine coupling.
        multi_scale: Whether to use multi-scale input (448, 224, 112).
        n_transforms: Number of test-time transforms (rotations). Set to 0 to disable.
        scales: List of (H, W) tuples for multi-scale input.
    """

    def __init__(self, backbone='alexnet', n_coupling_blocks=8,
                 clamp=3.0, pretrained=True,
                 multi_scale=True, n_transforms=64,
                 n_train_transforms=4,
                 colorjitter_brightness=0.0, colorjitter_contrast=0.0,
                 colorjitter_saturation=0.0,
                 fc_internal_size=2048,
                 loss_normalize_by_dim=True,
                 test_rotation_mode='random',
                 scales=((448, 448), (224, 224), (112, 112)),
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build AlexNet backbone via torchvision — use full features module
        # Original DifferNet: alexnet.features(x) → GAP → 256-dim per scale
        self.backbone_name = backbone if isinstance(backbone, str) else backbone.get('backbone_name', 'alexnet')
        weights = tv_models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        from baoiad.runtime import require_torchvision_weights

        require_torchvision_weights(weights, action='load AlexNet pretrained weights')
        alexnet = tv_models.alexnet(weights=weights)
        self.backbone_features = alexnet.features
        self.backbone_features.eval()
        for p in self.backbone_features.parameters():
            p.requires_grad = False

        # Multi-scale configuration
        self.multi_scale = multi_scale
        self.scales = scales
        self.n_transforms = n_transforms
        self.n_train_transforms = n_train_transforms
        self.colorjitter_brightness = colorjitter_brightness
        self.colorjitter_contrast = colorjitter_contrast
        self.colorjitter_saturation = colorjitter_saturation
        self.fc_internal_size = fc_internal_size
        self.loss_normalize_by_dim = loss_normalize_by_dim
        self.test_rotation_mode = test_rotation_mode
        if self.test_rotation_mode not in {'random', 'fixed'}:
            raise ValueError(
                f'Unsupported test_rotation_mode={test_rotation_mode!r}; '
                "expected 'random' or 'fixed'."
            )
        self._rotation_fill = tuple(
            -mean / std for mean, std in zip((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
        )

        # Determine feature dimension: run full backbone → GAP → 256-dim per scale
        with torch.no_grad():
            dummy = torch.randn(1, 3, 448, 448)
            feat = self.backbone_features(dummy)
            single_dim = feat.shape[1]  # 256 for AlexNet
        n_scales = len(scales) if multi_scale else 1
        total_dim = single_dim * n_scales  # 768 for 3 scales

        # Normalizing flow with custom coupling layers (atan clamping)
        self.flow = FlowSequence(
            total_dim,
            n_coupling_blocks,
            clamp,
            fc_internal_size=fc_internal_size,
        )

    @torch.no_grad()
    def _extract_single_scale(self, x_scaled):
        """Run full AlexNet features → GAP → flat vector (256-dim)."""
        feat = self.backbone_features(x_scaled)  # (B, 256, H', W')
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 256)

    @torch.no_grad()
    def extract_features(self, x):
        """Extract multi-scale features from backbone, GAP + concat."""
        if self.multi_scale:
            pooled = []
            for scale in self.scales:
                x_scaled = F.interpolate(x, size=scale, mode='bilinear',
                                         align_corners=False)
                pooled.append(self._extract_single_scale(x_scaled))
            return torch.cat(pooled, dim=1)  # (B, 768)
        else:
            return self._extract_single_scale(x)  # (B, 256)

    def _compute_score(self, feats):
        """Compute anomaly score using z² term only.

        For unconditional normalizing flows, the Jacobian is constant across
        samples and doesn't help with relative anomaly ranking. This matches
        FastFlow's prediction approach (discarding Jacobian in predict mode).
        """
        z, _ = self.flow(feats)
        return 0.5 * torch.mean(z ** 2, dim=1)

    def _rotate_tensor(self, x, angle):
        """Rotate tensor by given angle using torchvision's RandomRotation defaults."""
        return rotate(
            x,
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=list(self._rotation_fill),
        )

    def _apply_random_rotations(self, x):
        """Apply independent random rotations to each image in a batch."""
        rotated = []
        for i in range(x.shape[0]):
            angle = random.uniform(-180, 180)
            rotated.append(self._rotate_tensor(x[i:i + 1], angle))
        return torch.cat(rotated, dim=0)

    def _fixed_test_angles(self):
        """Return evenly spaced test angles over [0, 360)."""
        return [i * 360.0 / self.n_transforms for i in range(self.n_transforms)]

    def _apply_color_jitter(self, x):
        """Apply random color jitter (brightness + contrast + saturation) to batch."""
        # Brightness
        factor = 1.0 + (random.random() * 2 - 1) * self.colorjitter_brightness
        x = x * factor
        # Contrast
        factor = 1.0 + (random.random() * 2 - 1) * self.colorjitter_contrast
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * factor + mean
        # Saturation
        factor = 1.0 + (random.random() * 2 - 1) * self.colorjitter_saturation
        gray = x.mean(dim=1, keepdim=True)
        x = (x - gray) * factor + gray
        return x

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            # Multi-transform training: apply n_train_transforms augmented copies
            # per image, matching original DifferNet behavior.
            # Each image gets its own random rotation (per-image, not per-batch),
            # matching reference DataLoader with RandomRotation(180).
            all_feats = []
            for _ in range(self.n_train_transforms):
                augmented = self._apply_color_jitter(inputs)
                augmented = self._apply_random_rotations(augmented)
                feats = self.extract_features(augmented)
                all_feats.append(feats)
            feats = torch.cat(all_feats, dim=0)
            z, log_jac_det = self.flow(feats)
            loss = torch.mean(0.5 * torch.sum(z ** 2, dim=1) - log_jac_det)
            if self.loss_normalize_by_dim:
                loss = loss / z.shape[1]
            return {'loss': loss}

        elif mode == 'predict':
            if self.n_transforms > 1:
                return self._predict_with_transforms(inputs, data_samples)
            else:
                return self._predict_single(inputs, data_samples)

        # tensor mode
        feats = self.extract_features(inputs)
        return feats

    def _predict_single(self, inputs, data_samples):
        """Single-scale prediction without transforms."""
        feats = self.extract_features(inputs)
        img_scores = self._compute_score(feats)

        B = inputs.shape[0]
        H, W = inputs.shape[-2], inputs.shape[-1]
        score_map = img_scores.view(B, 1, 1, 1).expand(B, 1, H, W)

        return build_predict_results(data_samples, img_scores, score_map)

    def _predict_with_transforms(self, inputs, data_samples):
        """Multi-transform prediction (original DifferNet behavior).

        Applies N random transforms per sample and takes the MEAN score across
        all transforms, matching the original paper's test loader:
        matching the original paper:
        z_grouped = torch.cat(test_z, dim=0).view(-1, n_transforms_test, n_feat)
        anomaly_score = torch.mean(z_grouped ** 2, dim=(-2, -1))
        """
        B = inputs.shape[0]
        H, W = inputs.shape[-2], inputs.shape[-1]

        all_z = []
        if self.test_rotation_mode == 'fixed':
            angle_iter = self._fixed_test_angles()
        else:
            angle_iter = [None] * self.n_transforms

        for angle in angle_iter:
            rotated = self._apply_color_jitter(inputs)
            if angle is None:
                rotated = self._apply_random_rotations(rotated)
            else:
                rotated = self._rotate_tensor(rotated, angle)
            feats = self.extract_features(rotated)
            z, _ = self.flow(feats)
            all_z.append(z)

        # Original DifferNet: mean(z²) across all transforms and features
        # z_grouped shape: (B, n_transforms, n_feat) -> mean over transforms and features
        z_grouped = torch.stack(all_z, dim=1)  # (B, n_transforms, n_feat)
        img_scores = torch.mean(z_grouped ** 2, dim=(-2, -1))  # (B,)

        score_map = img_scores.view(B, 1, 1, 1).expand(B, 1, H, W)

        return build_predict_results(data_samples, img_scores, score_map)

    def train(self, mode=True):
        super().train(mode)
        self.backbone_features.eval()
        return self
