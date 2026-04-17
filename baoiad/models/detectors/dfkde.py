"""DFKDE: Deep Feature Kernel Density Estimation anomaly detector."""
import math
import random
from typing import List, Union

import torch
import torch.nn.functional as F

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import MemoryBankADModel


class _PCA:
    """Simple PCA via SVD (fit_transform only, for KDE pre-processing)."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.singular_vectors = None
        self.mean = None

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        self.mean = dataset.mean(dim=0)
        centered = dataset - self.mean
        v_h = torch.linalg.svd(centered)[-1]
        self.singular_vectors = v_h.transpose(-2, -1)[:, :self.n_components]
        return torch.matmul(centered, self.singular_vectors)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return torch.matmul(features - self.mean, self.singular_vectors)


class _GaussianKDE:
    """Gaussian KDE with Scott's bandwidth rule."""

    def __init__(self):
        self.bw_transform = None
        self.dataset = None
        self.norm = None

    def fit(self, dataset: torch.Tensor):
        num_samples, dimension = dataset.shape
        factor = num_samples ** (-1 / (dimension + 4))
        mean = dataset.mean(dim=0)
        dataset = dataset - mean
        centered = dataset.T
        cov_mat = torch.matmul(centered, centered.T) / (centered.size(1) - 1)
        inv_cov = torch.linalg.inv(cov_mat) / factor**2
        bw_transform = torch.linalg.cholesky(inv_cov)
        self.dataset = torch.matmul(dataset, bw_transform)
        self.bw_transform = bw_transform
        self.norm = torch.prod(torch.diag(bw_transform)) * math.pow(2 * math.pi, -dimension / 2)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.matmul(features, self.bw_transform)
        estimate = torch.zeros(features.shape[0], device=features.device)
        for i in range(features.shape[0]):
            embedding = ((self.dataset - features[i]) ** 2).sum(dim=1)
            embedding = torch.exp(-embedding / 2) * self.norm
            estimate[i] = torch.mean(embedding)
        return estimate


class _KDEClassifier:
    """KDE classifier: PCA -> scale -> KDE -> log-likelihood -> probability."""

    def __init__(self, n_pca_components=16, feature_scaling_method='scale', max_training_points=40000):
        self.n_pca_components = n_pca_components
        self.feature_scaling_method = feature_scaling_method
        self.max_training_points = max_training_points
        self.pca_model = _PCA(n_components=n_pca_components)
        self.kde_model = _GaussianKDE()
        self.max_length = None

    def _pre_process(self, feature_stack, max_length=None):
        if max_length is None:
            max_length = torch.max(torch.linalg.norm(feature_stack, ord=2, dim=1))
        if self.feature_scaling_method == 'norm':
            feature_stack = feature_stack / torch.linalg.norm(feature_stack, ord=2, dim=1, keepdim=True)
        elif self.feature_scaling_method == 'scale':
            feature_stack = feature_stack / max_length
        return feature_stack, max_length

    def fit(self, embeddings: torch.Tensor):
        if embeddings.shape[0] < self.n_pca_components:
            return False
        if embeddings.shape[0] > self.max_training_points:
            idx = torch.tensor(random.sample(range(embeddings.shape[0]), self.max_training_points),
                               device=embeddings.device)
            selected = embeddings[idx]
        else:
            selected = embeddings
        feature_stack = self.pca_model.fit_transform(selected)
        feature_stack, max_length = self._pre_process(feature_stack)
        self.max_length = max_length
        self.kde_model.fit(feature_stack)
        return True

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        features = self.pca_model.transform(features)
        features, _ = self._pre_process(features, self.max_length)
        kde_scores = self.kde_model.predict(features) + 1e-300
        log_scores = torch.log(kde_scores)
        # Convert to anomaly probability: higher = more anomalous
        return 1 / (1 + torch.exp(0.05 * (log_scores - 12)))


@MODELS.register_module()
class DFKDEDetector(MemoryBankADModel):
    """Deep Feature Kernel Density Estimation detector.

    Extracts deep features via a pretrained backbone, applies global average
    pooling, then fits a KDE classifier (PCA + Gaussian KDE) to detect anomalies.
    Image-level only (no pixel anomaly map).
    """

    def __init__(
        self,
        backbone: Union[str, dict] = 'resnet18',
        n_pca_components: int = 16,
        feature_scaling_method: str = 'scale',
        max_training_points: int = 40000,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        from baoiad.models.backbone_utils import build_feature_extractor
        # DFKDE uses only the last layer (layer4) feature via GAP
        self.backbone = build_feature_extractor(
            backbone, default_out_indices=(4,), default_frozen=True)

        self.classifier = _KDEClassifier(
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )
        self._memory_bank: List[torch.Tensor] = []

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled backbone features and concatenate them across layers."""
        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        pooled = []
        for feat in feats:
            pooled.append(F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1))
        return torch.cat(pooled, dim=1).detach()

    def fit(self):
        """Fit KDE classifier on collected features."""
        if not self._memory_bank:
            raise ValueError('Memory bank is empty. Cannot perform coreset selection.')
        embeddings = torch.vstack(self._memory_bank)
        self.classifier.fit(embeddings)
        self._memory_bank = []

    def build_memory_bank(self):
        """Called by MemoryBankHook after training to fit the KDE classifier."""
        self.fit()

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        features = self.extract_features(inputs)

        if mode == 'loss':
            self._memory_bank.append(features)
            return {'loss': torch.tensor(0.0, device=inputs.device, requires_grad=True)}

        elif mode == 'predict':
            scores = self.classifier.predict(features)
            B = inputs.shape[0]
            H, W = inputs.shape[2], inputs.shape[3]
            # No spatial anomaly map for DFKDE; provide uniform placeholder
            score_map = scores.view(B, 1, 1, 1).expand(B, 1, H, W)
            return build_predict_results(data_samples, scores, score_map)

        return features

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
