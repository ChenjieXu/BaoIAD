"""PaDiM anomaly detector."""
import random
from typing import Any, Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import MemoryBankADModel


@MODELS.register_module()
class PaDiMDetector(MemoryBankADModel):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection."""

    # Default n_features per backbone (from anomalib / original paper)
    _N_FEATURES_DEFAULTS = {
        'resnet18': 100,
        'wide_resnet50_2': 550,
    }

    def __init__(self, backbone: Union[str, dict] = 'wide_resnet50_2',
                 d_reduced=None, eps=0.01, sigma=4.0,
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build backbone with TIMMBackbone support
        if isinstance(backbone, dict) and backbone.get('type') == 'TIMMBackbone':
            backbone.setdefault('out_indices', (1, 2, 3))
            backbone.setdefault('frozen', True)
            self.backbone = MODELS.build(backbone)
            backbone_name = backbone.get('model_name', '')
        else:
            from baoiad.models.backbone_utils import build_feature_extractor
            self.backbone = build_feature_extractor(
                backbone, default_out_indices=(1, 2, 3), default_frozen=True)
            backbone_name = backbone if isinstance(backbone, str) else ''

        ch = self.backbone.out_channels  # e.g. (256, 512, 1024)
        total_dim = sum(ch)

        # Determine d_reduced
        if d_reduced is None:
            d_reduced = self._N_FEATURES_DEFAULTS.get(backbone_name, 100)
        self.d_reduced = min(d_reduced, total_dim)

        # Random dimension selection - match anomalib exactly:
        # Use random.sample() without sorting (anomalib behavior)
        rng = random.Random(42)
        self.register_buffer('idx', torch.tensor(rng.sample(range(total_dim), self.d_reduced)))

        self.eps = eps

        # Gaussian smoothing (anomalib uses sigma=4, kernel_size=33)
        self.sigma = sigma
        if sigma > 0:
            kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
            self.blur = _GaussianBlur2d(sigma=sigma, kernel_size=kernel_size)
        else:
            self.blur = None

        # Buffers for Gaussian stats (will be set by fit())
        self.register_buffer('mean', None)
        self.register_buffer('cov_inv', None)
        # Runtime-only feature cache: NOT serialized in state_dict. Resuming
        # from a mid-training checkpoint should preserve this cache so fit() can
        # still finalize the Gaussian statistics without re-running warmup.
        self._features = []
        self._register_state_dict_hook(self._save_feature_cache_to_state_dict)
        self._register_load_state_dict_pre_hook(self._load_feature_cache_from_state_dict)

    @staticmethod
    def _save_feature_cache_to_state_dict(module, state_dict, prefix, local_metadata):
        if module._features:
            state_dict[prefix + '_feature_cache'] = torch.stack(module._features, dim=0)
        return state_dict

    def _load_feature_cache_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs,
    ):
        key = prefix + '_feature_cache'
        if key not in state_dict:
            return
        tensor = state_dict.pop(key)
        self._features = [feature.detach().cpu() for feature in tensor]

    @torch.no_grad()
    def extract_features(self, x):
        feats = self.backbone(x)  # tuple of tensors
        f1, f2, f3 = feats[0], feats[1], feats[2]
        # Resize all to layer1 spatial size using nearest (matches anomalib)
        H, W = f1.shape[2], f1.shape[3]
        f2 = F.interpolate(f2, size=(H, W), mode='nearest')
        f3 = F.interpolate(f3, size=(H, W), mode='nearest')
        feat = torch.cat([f1, f2, f3], dim=1)  # B, C, H, W
        # Select random dimensions
        feat = feat[:, self.idx]
        return feat  # B, d_reduced, H, W

    def fit(self):
        """Compute Gaussian statistics from collected features."""
        if not self._features:
            return
        # _features: list of (d, H, W) tensors (on CPU)
        all_feats = torch.stack(self._features, dim=0)  # N, d, H, W
        N, d, H, W = all_feats.shape
        # Reshape to (H*W, N, d)
        all_feats = all_feats.permute(2, 3, 0, 1).reshape(H * W, N, d)
        mean = all_feats.mean(dim=1)  # H*W, d
        diff = all_feats - mean.unsqueeze(1)  # H*W, N, d
        cov = torch.bmm(diff.transpose(1, 2), diff) / max(N - 1, 1)  # H*W, d, d
        cov = cov + self.eps * torch.eye(d, device=cov.device).unsqueeze(0)
        # Cholesky-based inversion is more numerically stable than torch.linalg.inv
        # for symmetric positive-definite covariance matrices.
        cov_inv = torch.cholesky_inverse(torch.linalg.cholesky(cov))  # H*W, d, d

        # Move to model device (features are collected on CPU but model may be on GPU);
        # use the always-present `idx` buffer instead of `next(self.parameters())`,
        # which raises StopIteration when the model has no trainable parameters.
        device = self.idx.device
        self.register_buffer('mean', mean.to(device))
        self.register_buffer('cov_inv', cov_inv.to(device))
        self._features = []

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        device = self.idx.device
        if torch.is_tensor(inputs):
            return inputs.to(device)
        if isinstance(inputs, list):
            return [item.to(device) if torch.is_tensor(item) else item for item in inputs]
        if isinstance(inputs, tuple):
            return tuple(item.to(device) if torch.is_tensor(item) else item for item in inputs)
        return inputs

    def build_memory_bank(self, dataloader: Optional[Iterable[Any]] = None):
        """Build Gaussian statistics from cached features or a dataloader."""
        if self.mean is not None and self.cov_inv is not None and not self._features:
            return

        if dataloader is not None and not self._features:
            was_training = self.training
            self.train()
            with torch.no_grad():
                for batch in dataloader:
                    inputs = self._move_inputs_to_device(batch['inputs'])
                    self(inputs, batch.get('data_samples', []), mode='loss')
            self.train(was_training)

        self.fit()
        if self.mean is None or self.cov_inv is None:
            raise RuntimeError(
                'PaDiM Gaussian statistics are not built yet. '
                'Run loss-mode warmup or call build_memory_bank(train_loader) first.'
            )

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats = self.extract_features(inputs)  # B, d, H, W

        if mode == 'loss':
            # Collect features for fitting
            for i in range(feats.shape[0]):
                self._features.append(feats[i].cpu())
            return {'loss': torch.tensor(0.0, device=inputs.device, requires_grad=True)}

        elif mode == 'predict':
            if self.mean is None or self.cov_inv is None:
                raise RuntimeError(
                    'PaDiM Gaussian statistics are not built. '
                    'Call build_memory_bank()/fit() before predict.'
                )
            B, d, H, W = feats.shape
            # Compute Mahalanobis distance
            feats_r = feats.permute(0, 2, 3, 1).reshape(B, H * W, d)  # B, HW, d
            diff = feats_r - self.mean.unsqueeze(0)  # B, HW, d
            maha = torch.einsum('bhd,hde,bhe->bh', diff, self.cov_inv, diff)  # B, HW
            maha = torch.sqrt(torch.clamp(maha, min=0))
            score_map = maha.reshape(B, 1, H, W)
            score_map = F.interpolate(score_map, size=inputs.shape[-2:], mode='bilinear', align_corners=False)

            # Apply Gaussian smoothing (per anomalib / original paper)
            if self.blur is not None:
                score_map = self.blur(score_map)

            img_scores = score_map.view(B, -1).max(dim=1).values
            return build_predict_results(data_samples, img_scores, score_map)

        return feats


class _GaussianBlur2d(nn.Module):
    """2D Gaussian blur using depthwise convolution."""

    def __init__(self, sigma: float, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        # Build 1D Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        # 2D kernel via outer product
        kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)  # (K, K)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        self.register_buffer('kernel', kernel_2d)
        self.padding = kernel_size // 2

    def forward(self, x):
        # x: (B, C, H, W) — apply per-channel
        B, C, H, W = x.shape
        kernel = self.kernel.expand(C, -1, -1, -1)
        return F.conv2d(x, kernel, padding=self.padding, groups=C)
