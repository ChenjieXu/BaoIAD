# Fixed: Added build_memory_bank() method so MemoryBankHook triggers PCA fit() after training.
"""DFM: Deep Feature Modeling anomaly detector."""
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import MemoryBankADModel


class _PCA(nn.Module):
    """PCA with variance-ratio or fixed component count, plus inverse_transform."""

    def __init__(self, n_components: float = 0.97):
        super().__init__()
        self.n_components = n_components
        self.register_buffer("singular_vectors", torch.empty(0))
        self.register_buffer("singular_values", torch.empty(0))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("num_components", torch.empty(0))

    def fit(self, dataset: torch.Tensor) -> None:
        mean = dataset.mean(dim=0)
        centered = dataset - mean
        _, sig, v_h = torch.linalg.svd(centered.double(), full_matrices=False)
        if self.n_components <= 1.0:
            total_variance = torch.sum(sig * sig)
            if total_variance <= 0:
                num_components = 1
            else:
                variance_ratios = torch.cumsum(sig * sig, dim=0) / total_variance
                qualified = torch.nonzero(variance_ratios >= self.n_components, as_tuple=False)
                num_components = int(qualified[0].item()) if len(qualified) > 0 else 1
            num_components = max(num_components, 1)
        else:
            num_components = int(self.n_components)
        self.num_components = torch.tensor([num_components], device=dataset.device)
        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components].float()
        self.singular_values = sig[:num_components].float()
        self.mean = mean

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return torch.matmul(features - self.mean, self.singular_vectors)

    def inverse_transform(self, features: torch.Tensor) -> torch.Tensor:
        return torch.matmul(features, self.singular_vectors.transpose(-2, -1)) + self.mean


class _SingleClassGaussian(nn.Module):
    """Gaussian distribution fitted via SVD for NLL scoring."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean_vec", torch.empty(0))
        self.register_buffer("u_mat", torch.empty(0))
        self.register_buffer("sigma_mat", torch.empty(0))

    def fit(self, dataset: torch.Tensor) -> None:
        """Fit Gaussian. dataset shape: (n_features, n_samples)."""
        num_samples = dataset.shape[1]
        self.mean_vec = dataset.mean(dim=1)
        data_centered = (dataset - self.mean_vec.reshape(-1, 1)) / math.sqrt(num_samples)
        self.u_mat, self.sigma_mat, _ = torch.linalg.svd(data_centered, full_matrices=False)

    def score_samples(self, features: torch.Tensor) -> torch.Tensor:
        """NLL scores. features shape: (n_samples, n_features)."""
        transformed = torch.matmul(features - self.mean_vec, self.u_mat / self.sigma_mat)
        return torch.sum(transformed * transformed, dim=1) + 2 * torch.sum(torch.log(self.sigma_mat))


@MODELS.register_module(force=True)
class DFMDetector(MemoryBankADModel):
    """Deep Feature Modeling for anomaly detection.

    Extracts features from a frozen pretrained backbone, applies PCA,
    and scores via either Feature Reconstruction Error (fre) or
    Negative Log-Likelihood (nll) of a fitted Gaussian.
    """

    _LAYER_TO_INDEX = {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4}

    def __init__(
        self,
        backbone: Union[str, dict] = "resnet50",
        layer: str = "layer3",
        pooling_kernel_size: int = 4,
        pca_level: float = 0.97,
        score_type: str = "fre",
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.layer = layer
        self.pooling_kernel_size = pooling_kernel_size
        self.score_type = score_type

        # Build backbone
        layer_idx = self._LAYER_TO_INDEX[layer]
        if isinstance(backbone, dict) and backbone.get('type') == 'TIMMBackbone':
            # TIMMBackbone: out_indices already filters to requested layer
            # timm index: layer1=1, layer2=2, layer3=3, layer4=4
            backbone.setdefault('out_indices', (layer_idx,))
            backbone.setdefault('frozen', True)
            self.backbone = MODELS.build(backbone)
            self._feat_idx = 0  # TIMMBackbone returns only requested layers
        else:
            from baoiad.models.backbone_utils import build_feature_extractor
            self._feat_idx = layer_idx - 1  # 0-based index for RawBackbone
            self.backbone = build_feature_extractor(
                backbone, default_out_indices=(layer_idx,), default_frozen=True)

        self.pca_model = _PCA(n_components=pca_level)
        self.gaussian_model = _SingleClassGaussian()
        self._memory_bank: List[torch.Tensor] = []

    def _ensure_fitted(self) -> None:
        """Fail fast when predict is called before memory-bank fitting."""
        if self.pca_model.mean.numel() == 0 or self.pca_model.singular_vectors.numel() == 0:
            raise RuntimeError('DFM PCA model is not fitted. Call build_memory_bank()/fit() before predict.')
        if self.score_type == "nll" and (
            self.gaussian_model.mean_vec.numel() == 0
            or self.gaussian_model.u_mat.numel() == 0
            or self.gaussian_model.sigma_mat.numel() == 0
        ):
            raise RuntimeError('DFM Gaussian model is not fitted. Call build_memory_bank()/fit() before predict.')

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor):
        """Returns (flat_features, feature_shapes)."""
        feats = self.backbone(x)
        out = feats[self._feat_idx]  # use correct layer index
        B = out.shape[0]
        if self.pooling_kernel_size > 1:
            out = F.avg_pool2d(out, kernel_size=self.pooling_kernel_size)
        feature_shapes = out.shape
        flat = out.view(B, -1).detach()
        return flat, feature_shapes

    def fit(self):
        """Fit PCA (and optionally Gaussian) on collected features."""
        if not self._memory_bank:
            return
        all_feats = torch.vstack(self._memory_bank)
        self.pca_model.fit(all_feats)
        if self.score_type == "nll":
            reduced = self.pca_model.transform(all_feats)
            self.gaussian_model.fit(reduced.T)
        self._memory_bank = []

    def build_memory_bank(self):
        """Called by MemoryBankHook after training to fit PCA/Gaussian models."""
        self.fit()

    def _score(self, features: torch.Tensor, feature_shapes: tuple):
        """Compute anomaly scores and optional anomaly map."""
        projected = self.pca_model.transform(features)
        if self.score_type == "nll":
            score = self.gaussian_model.score_samples(projected)
            return score, None
        elif self.score_type == "fre":
            reconstructed = self.pca_model.inverse_transform(projected)
            fre = torch.square(features - reconstructed).reshape(feature_shapes)
            score_map = torch.unsqueeze(torch.sum(fre, dim=1), 1)  # B,1,H,W
            score = torch.sum(torch.square(features - reconstructed), dim=1)  # B
            return score, score_map
        else:
            raise ValueError(f"Unsupported score type: {self.score_type}")

    def forward(self, inputs, data_samples=None, mode="tensor"):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        features, feature_shapes = self._extract_features(inputs)

        if mode == "loss":
            self._memory_bank.append(features)
            return {"loss": torch.tensor(0.0, device=inputs.device, requires_grad=True)}

        elif mode == "predict":
            self._ensure_fitted()
            scores, score_map = self._score(features, feature_shapes)
            if score_map is not None:
                score_map = F.interpolate(score_map, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            else:
                score_map = torch.zeros(inputs.shape[0], 1, inputs.shape[2], inputs.shape[3])
            return build_predict_results(data_samples, scores, score_map)

        return features

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, '_prefix') and self._prefix is not None:
            self._prefix.eval()
        if hasattr(self, '_blocks'):
            for m in self._blocks.values():
                m.eval()
        return self
