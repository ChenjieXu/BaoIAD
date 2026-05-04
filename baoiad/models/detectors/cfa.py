"""CFA: Coupled-hypersphere-based Feature Adaptation (ECCV 2022).

Faithful reimplementation with:
- WideResNet-50-2 backbone (layer1, layer2, layer3 features)
- Descriptor with CoordConv2d for target-oriented features
- Memory bank with KMeans centroid initialization
- Coupled-hypersphere loss (attraction + repulsion)
- Anomaly map via k-NN distance with softmin weighting + Gaussian blur
"""

from typing import Any, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import DiscriminatorADModel


# ─── CoordConv components ───────────────────────────────────────────────────

class AddCoords(nn.Module):
    """Add normalized coordinate channels to input tensor."""

    def __init__(self, with_r: bool = False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, x_dim, y_dim = x.shape
        xx_range = torch.arange(x_dim, dtype=torch.int32, device=x.device)
        yy_range = torch.arange(y_dim, dtype=torch.int32, device=x.device)

        xx_channel = xx_range[None, None, :, None].expand(batch, 1, x_dim, y_dim).float()
        yy_channel = yy_range[None, None, None, :].expand(batch, 1, x_dim, y_dim).float()

        xx_channel = xx_channel / max(x_dim - 1, 1) * 2 - 1
        yy_channel = yy_channel / max(y_dim - 1, 1) * 2 - 1

        out = torch.cat([x, xx_channel, yy_channel], dim=1)
        if self.with_r:
            rr = torch.sqrt((xx_channel - 0.5) ** 2 + (yy_channel - 0.5) ** 2)
            out = torch.cat([out, rr], dim=1)
        return out


class CoordConv2d(nn.Module):
    """Conv2d with coordinate channels prepended."""

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.add_coords = AddCoords(with_r=False)
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        return self.conv(self.add_coords(x))


# ─── Descriptor network ────────────────────────────────────────────────────

def _weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)


class Descriptor(nn.Module):
    """Target-oriented feature descriptor using CoordConv."""

    def __init__(self, in_dim: int, gamma_d: int):
        super().__init__()
        out_channels = in_dim // gamma_d
        self.layer = CoordConv2d(in_dim, out_channels, kernel_size=1)
        self.apply(_weights_init)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        patch_features = None
        for feat in features:
            pooled = F.avg_pool2d(feat, 3, 1, 1)
            if patch_features is None:
                patch_features = pooled
            else:
                patch_features = torch.cat(
                    (patch_features, F.interpolate(feat, size=(patch_features.shape[2], patch_features.shape[3]), mode="bilinear", align_corners=False)),
                    dim=1,
                )
        return self.layer(patch_features)


# ─── Gaussian blur ──────────────────────────────────────────────────────────

class GaussianBlur2d(nn.Module):
    """Gaussian blur with fixed kernel."""

    def __init__(self, sigma: float, channels: int = 1, kernel_size: int = 0):
        super().__init__()
        if kernel_size == 0:
            kernel_size = int(2 * (4 * sigma + 0.5)) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
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


# ─── CFA Detector ──────────────────────────────────────────────────────────

@MODELS.register_module(force=True)
class CFADetector(DiscriminatorADModel):
    """CFA: Coupled-hypersphere-based Feature Adaptation.

    The model extracts multi-scale features with a frozen WRN-50-2,
    projects them through a descriptor (CoordConv), computes distances
    to a memory bank of normal-sample centroids, and uses a coupled
    hypersphere loss for training.

    Args:
        backbone (str): Backbone name. Default 'wide_resnet50_2'.
        gamma_c (int): Memory bank clustering ratio. Default 1.
        gamma_d (int): Descriptor channel reduction factor. Default 1.
        num_nearest_neighbors (int): k for anomaly scoring. Default 3.
        num_hard_negative_features (int): Hard negatives for loss. Default 3.
        radius (float): Initial hypersphere radius. Default 1e-5.
        num_init_batches (int): Number of batches for centroid init. Default 0.
        sigma (int): Gaussian blur sigma for anomaly map. Default 4.
    """

    def __init__(
        self,
        backbone: Union[str, dict] = 'wide_resnet50_2',
        gamma_c: int = 1,
        gamma_d: int = 1,
        num_nearest_neighbors: int = 3,
        num_hard_negative_features: int = 3,
        radius: float = 1e-5,
        num_init_batches: int = 3,
        sigma: int = 4,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        from baoiad.models.backbone_utils import build_feature_extractor

        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_hard_negative_features = num_hard_negative_features
        self.radius = nn.Parameter(torch.ones(1) * radius)
        self.sigma = sigma
        self.num_init_batches = num_init_batches

        # Feature extractor (frozen)
        self.backbone = build_feature_extractor(
            backbone, default_out_indices=(1, 2, 3), default_frozen=True)

        # Feature dims from backbone (layer1, layer2, layer3)
        ch = self.backbone.out_channels
        total_dim = sum(ch)
        self.descriptor = Descriptor(total_dim, gamma_d)

        # Memory bank: initialized as empty, needs to be set before training
        self.register_buffer('memory_bank', torch.zeros(0))
        self._init_count = 0
        self._init_accumulator = None

    @torch.no_grad()
    def _extract_features(self, x):
        feats = self.backbone(x)
        return list(feats)

    def _memory_bank_ready(self) -> bool:
        return self.memory_bank.numel() > 0

    def _accumulate_memory_bank_features(self, target_features: torch.Tensor) -> bool:
        if self._memory_bank_ready():
            return True

        if self._init_accumulator is None:
            self._init_accumulator = []
        self._init_accumulator.append(target_features.detach())
        self._init_count += 1

        if self._init_count >= max(int(self.num_init_batches), 1):
            self._finalize_memory_bank()
            return True
        return False

    def _finalize_memory_bank(self) -> bool:
        if self._memory_bank_ready():
            return True
        if not self._init_accumulator:
            return False

        self.initialize_memory_bank(self._init_accumulator)
        self._init_accumulator = None
        return True

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        device = next(self.parameters()).device
        if torch.is_tensor(inputs):
            return inputs.to(device)
        if isinstance(inputs, list):
            return [item.to(device) if torch.is_tensor(item) else item for item in inputs]
        if isinstance(inputs, tuple):
            return tuple(item.to(device) if torch.is_tensor(item) else item for item in inputs)
        return inputs

    def _compute_distance(self, target_features: torch.Tensor) -> torch.Tensor:
        """Compute squared L2 distance between features and memory bank.

        Args:
            target_features: (B, C, H, W) descriptor output
        Returns:
            distance: (B, H*W, num_centroids)
        """
        # Reshape to (B, HW, C)
        B, C, H, W = target_features.shape
        tf = target_features.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Efficient squared distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        feat_sq = tf.pow(2).sum(dim=2, keepdim=True)  # (B, HW, 1)
        center_sq = self.memory_bank.pow(2).sum(dim=0, keepdim=True)  # (1, num_centroids)
        cross = 2 * torch.matmul(tf, self.memory_bank)  # (B, HW, num_centroids)
        return feat_sq + center_sq - cross

    def _compute_loss(self, distance: torch.Tensor) -> torch.Tensor:
        """CFA coupled-hypersphere loss."""
        k = self.num_nearest_neighbors + self.num_hard_negative_features
        distance = distance.topk(k, largest=False).values  # (B, HW, k)

        # Attraction: pull close neighbors inside hypersphere
        score_att = distance[:, :, :self.num_nearest_neighbors] - self.radius ** 2
        l_att = torch.clamp(score_att, min=0).mean()

        # Repulsion: push hard negatives outside hypersphere
        score_rep = self.radius ** 2 - distance[:, :, self.num_hard_negative_features:]
        l_rep = torch.clamp(score_rep - 0.1, min=0).mean()

        return (l_att + l_rep) * 1000

    def _compute_anomaly_map(self, distance: torch.Tensor, scale: tuple, image_size: tuple) -> torch.Tensor:
        """Generate anomaly map from distances using softmin-weighted k-NN."""
        distance = torch.sqrt(distance.clamp(min=1e-10))
        topk = distance.topk(self.num_nearest_neighbors, largest=False).values  # (B, HW, k)
        weights = F.softmin(topk, dim=-1)[:, :, 0]  # (B, HW)
        score = weights * topk[:, :, 0]  # (B, HW)

        B = score.shape[0]
        score = score.reshape(B, 1, scale[0], scale[1])
        if image_size is not None:
            score = F.interpolate(score, size=image_size, mode='bilinear', align_corners=False)

        blur = GaussianBlur2d(sigma=self.sigma, channels=1).to(score.device)
        return blur(score)

    def initialize_memory_bank(self, features_list: List[torch.Tensor]):
        """Initialize memory bank from accumulated descriptor features.

        Call this with batches of normal training features.
        """
        # features_list is list of (B, C, H, W) descriptor outputs
        all_features = torch.cat(features_list, dim=0)  # (N, C, H, W)
        # Average over batch → single prototype
        mean_feat = all_features.mean(dim=0, keepdim=True)  # (1, C, H, W)
        C, H, W = mean_feat.shape[1:]
        bank = mean_feat.reshape(1, C, H * W).permute(0, 2, 1).reshape(H * W, C)  # (HW, C)

        if self.gamma_c > 1:
            from sklearn.cluster import KMeans
            n_clusters = (H * W) // self.gamma_c
            km = KMeans(n_clusters=n_clusters, max_iter=3000, n_init=1)
            bank = torch.tensor(km.fit(bank.cpu().numpy()).cluster_centers_,
                                dtype=torch.float32, device=mean_feat.device)

        self.memory_bank = bank.t().contiguous()  # (C, num_centroids)

    def build_memory_bank(self, dataloader: Optional[Iterable[Any]] = None) -> None:
        """Build the CFA memory bank from collected or loader-provided features."""
        if self._memory_bank_ready():
            return

        required_batches = max(int(self.num_init_batches), 1)
        was_training = self.training

        if dataloader is not None and self._init_count < required_batches:
            self.train()
            with torch.no_grad():
                for batch in dataloader:
                    inputs = self._move_inputs_to_device(batch['inputs'])
                    self(inputs, batch.get('data_samples', []), mode='loss')
                    if self._memory_bank_ready() or self._init_count >= required_batches:
                        break

        built = self._finalize_memory_bank()
        self.train(was_training)

        if not built:
            raise RuntimeError(
                'CFA memory bank is not built yet. '
                'Run loss-mode warmup or call build_memory_bank(train_loader) first.'
            )

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        features = self._extract_features(inputs)
        target_features = self.descriptor(features)

        if mode == 'loss':
            # Memory bank initialization: accumulate features in first batches
            if not self._memory_bank_ready():
                self._accumulate_memory_bank_features(target_features)
                # Return dummy loss during init
                return {'loss': torch.tensor(0.0, device=inputs.device, requires_grad=True)}

            distance = self._compute_distance(target_features)
            loss = self._compute_loss(distance)
            return {'loss': loss}

        elif mode == 'predict':
            if not self._memory_bank_ready():
                raise RuntimeError(
                    'CFA memory bank is not built. '
                    'Call build_memory_bank() first or run enough loss batches to initialize centroids.'
                )

            distance = self._compute_distance(target_features)
            scale = target_features.shape[-2:]
            anomaly_map = self._compute_anomaly_map(distance, scale, inputs.shape[-2:])

            B = inputs.shape[0]
            img_scores = anomaly_map.view(B, -1).max(dim=1).values
            return build_predict_results(data_samples, img_scores, anomaly_map)

        return target_features

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
