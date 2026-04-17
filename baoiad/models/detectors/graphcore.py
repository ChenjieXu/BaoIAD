"""GraphCore detector aligned to the official open-iad implementation."""

from __future__ import annotations

import warnings
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.utils.graphcore_alignment import (
    GRAPHCORE_STRICT_CORESET_INITIAL_INDEX,
    GRAPHCORE_STRICT_IMAGE_SCORE_MODE,
    normalize_graphcore_image_score_mode,
    normalize_graphcore_image_score_mode_overrides,
    reduce_graphcore_image_score,
)
from baoiad.models.base_ad_model import BaseADModel

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def _kcenter_greedy(
    features,
    n_select: int,
    seed: int = 42,
    initial_index: int | None = None,
) -> np.ndarray:
    """Select coreset indices with the official GraphCore sampling path."""
    n_points = int(features.shape[0])
    if n_points == 0:
        return np.empty(0, dtype=np.int64)

    n_select = max(1, min(int(n_select), n_points))
    rng = np.random.default_rng(seed)
    first = int(initial_index) if initial_index is not None else int(rng.integers(n_points))
    first = max(0, min(first, n_points - 1))
    selected = [first]

    min_distances = pairwise_distances(features, features[[first]], metric='euclidean').reshape(-1)
    min_distances[first] = 0.0

    for _ in range(1, n_select):
        idx = int(np.argmax(min_distances))
        if idx in selected:
            break
        selected.append(idx)
        distances = pairwise_distances(features, features[[idx]], metric='euclidean').reshape(-1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[idx] = 0.0

    return np.asarray(selected, dtype=np.int64)


@MODELS.register_module()
class GraphCoreDetector(BaseADModel):
    """GraphCore detector using official ViG embeddings and KCenter coreset.

    Diagnose-only score modes remain available for targeted scripts, but the
    strict benchmark mainline stays on a single global ``raw_max`` image score.
    """

    def __init__(
        self,
        backbone=None,
        net: str = 'vig_ti_224_gelu',
        pretrained: bool = True,
        checkpoint_path: str = '',
        n_neighbours: int = 9,
        sampler_percentage: float = 0.001,
        layer_num_1: int = 3,
        layer_num_2: int = 4,
        local_smoothing: bool = False,
        input_size: int | tuple[int, int] = 224,
        smoothing_sigma: float = 4.0,
        image_score_mode: str = GRAPHCORE_STRICT_IMAGE_SCORE_MODE,
        image_score_mode_overrides: dict | None = None,
        random_projection_eps: float = 0.9,
        random_seed: int = 42,
        coreset_initial_index: int | None = GRAPHCORE_STRICT_CORESET_INITIAL_INDEX,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = self._build_backbone(backbone, net, pretrained, checkpoint_path)
        self.n_neighbours = int(n_neighbours)
        self.sampler_percentage = float(sampler_percentage)
        self.layer_num_1 = int(layer_num_1)
        self.layer_num_2 = int(layer_num_2)
        self.local_smoothing = bool(local_smoothing)
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
        self.smoothing_sigma = float(smoothing_sigma)
        self.image_score_mode = normalize_graphcore_image_score_mode(image_score_mode)
        self.image_score_mode_overrides = normalize_graphcore_image_score_mode_overrides(
            image_score_mode_overrides
        )
        self.random_projection_eps = float(random_projection_eps)
        self.random_seed = int(random_seed)
        self.coreset_initial_index = (
            None if coreset_initial_index is None else int(coreset_initial_index)
        )
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

        self._feature_maps: List[torch.Tensor] = []
        self._hook_handles = []
        self._train_embeddings: List[np.ndarray] = []
        self.random_projector: Optional[SparseRandomProjection] = None
        self.embedding_coreset: Optional[np.ndarray] = None
        self._nn_index = None

        self._register_feature_hooks()
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    @staticmethod
    def _build_backbone(backbone, net: str, pretrained: bool, checkpoint_path: str):
        if backbone is None:
            backbone = dict(
                type='GraphCoreViGBackbone',
                model_name=net,
                pretrained=pretrained,
                checkpoint_path=checkpoint_path,
                frozen=True,
            )
        elif isinstance(backbone, str):
            backbone = dict(
                type='GraphCoreViGBackbone',
                model_name=backbone,
                pretrained=pretrained,
                checkpoint_path=checkpoint_path,
                frozen=True,
            )
        else:
            backbone = dict(backbone)
            if backbone.get('type') == 'GraphCoreViGBackbone':
                backbone.setdefault('model_name', net)
                backbone.setdefault('pretrained', pretrained)
                backbone.setdefault('checkpoint_path', checkpoint_path)
                backbone.setdefault('frozen', True)
        return MODELS.build(backbone)

    def _register_feature_hooks(self) -> None:
        for index in (self.layer_num_1, self.layer_num_2):
            if index < 0 or index >= len(self.backbone.backbone):
                raise IndexError(
                    f'GraphCore hook index {index} is out of range for backbone with '
                    f'{len(self.backbone.backbone)} blocks.'
                )
            module = self.backbone.backbone[index][-1]
            self._hook_handles.append(module.register_forward_hook(self._save_feature_hook))

    def _save_feature_hook(self, module, inputs, output) -> None:
        self._feature_maps.append(output)

    @staticmethod
    def embedding_concate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size, channels_x, height_x, width_x = x.shape
        _, channels_y, height_y, width_y = y.shape
        if height_x % height_y != 0 or width_x % width_y != 0:
            raise ValueError(
                'GraphCore embedding_concate expects integer spatial downsampling '
                f'ratios, got {(height_x, width_x)} vs {(height_y, width_y)}.'
            )

        scale = height_x // height_y
        if scale != width_x // width_y:
            raise ValueError('GraphCore embedding_concate requires matching H/W scale factors.')

        x = F.unfold(x, kernel_size=scale, dilation=1, stride=scale)
        x = x.reshape(batch_size, channels_x, -1, height_y, width_y)
        y = y.unsqueeze(2).expand(-1, -1, x.shape[2], -1, -1)
        z = torch.cat((x, y), dim=1)
        z = z.reshape(batch_size, -1, height_y * width_y)
        return F.fold(z, kernel_size=scale, output_size=(height_x, width_x), stride=scale)

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> np.ndarray:
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1]).detach().cpu().numpy().astype(np.float32)

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        self._feature_maps.clear()
        with torch.no_grad():
            _ = self.backbone(inputs)
        if len(self._feature_maps) != 2:
            raise RuntimeError(
                'GraphCore expected exactly 2 hooked feature maps, '
                f'but received {len(self._feature_maps)}.'
            )

        embeddings = []
        for feat in self._feature_maps:
            if self.local_smoothing:
                feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            embeddings.append(feat)
        return self.embedding_concate(embeddings[0], embeddings[1])

    def _build_nn_index(self) -> None:
        if self.embedding_coreset is None or self.embedding_coreset.size == 0:
            raise RuntimeError('GraphCore coreset is empty.')
        data = np.ascontiguousarray(self.embedding_coreset.astype(np.float32))
        if HAS_FAISS:
            self._nn_index = faiss.IndexFlatL2(data.shape[1])
            self._nn_index.add(data)
            return

        warnings.warn(
            'faiss is unavailable; GraphCore is falling back to sklearn NearestNeighbors.',
            RuntimeWarning,
        )
        self._nn_index = NearestNeighbors(metric='euclidean', algorithm='auto')
        self._nn_index.fit(data)

    def _search_nn(self, queries: np.ndarray, k: int) -> np.ndarray:
        if self._nn_index is None:
            raise RuntimeError('GraphCore memory bank is not built.')
        queries = np.ascontiguousarray(queries.astype(np.float32))
        if HAS_FAISS:
            distances, _ = self._nn_index.search(queries, k=k)
            return distances
        distances, _ = self._nn_index.kneighbors(queries, n_neighbors=k)
        return distances ** 2

    def build_memory_bank(self) -> None:
        if not self._train_embeddings:
            raise RuntimeError('No GraphCore embeddings were collected during training.')

        total_embeddings = np.concatenate(self._train_embeddings, axis=0).astype(np.float32)
        self._train_embeddings.clear()

        self.random_projector = SparseRandomProjection(
            n_components='auto',
            eps=self.random_projection_eps,
            random_state=self.random_seed,
        )
        self.random_projector.fit(total_embeddings)
        projected_embeddings = self.random_projector.transform(total_embeddings)

        n_select = max(1, int(total_embeddings.shape[0] * self.sampler_percentage))
        selected_idx = _kcenter_greedy(
            projected_embeddings,
            n_select,
            seed=self.random_seed,
            initial_index=self.coreset_initial_index,
        )
        self.embedding_coreset = total_embeddings[selected_idx]
        self._build_nn_index()

    def fit(self) -> None:
        self.build_memory_bank()

    def _resolve_image_score_mode(self, data_sample=None) -> str:
        if data_sample is not None:
            cls_name = getattr(data_sample, 'cls_name', None)
            if cls_name is None and hasattr(data_sample, 'metainfo'):
                cls_name = data_sample.metainfo.get('cls_name', None)
            if cls_name in self.image_score_mode_overrides:
                return self.image_score_mode_overrides[cls_name]
        return self.image_score_mode

    def _reduce_image_score(self, patch_map: np.ndarray, smooth_map: np.ndarray, data_sample=None) -> float:
        mode = self._resolve_image_score_mode(data_sample)
        return reduce_graphcore_image_score(patch_map, smooth_map, mode)

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        embeddings = self.extract_embeddings(inputs)

        if mode == 'loss':
            self._train_embeddings.append(self.reshape_embedding(embeddings))
            return {'loss': self._dummy.sum() * 0.0}

        if mode == 'predict':
            batch_size, _, height, width = embeddings.shape
            queries = self.reshape_embedding(embeddings)
            patch_distances = self._search_nn(queries, k=self.n_neighbours)
            patch_scores = patch_distances[:, 0].reshape(batch_size, height, width)

            image_scores = []
            score_maps = []
            target_size = self.input_size or tuple(inputs.shape[-2:])
            for i in range(batch_size):
                score_map = cv2.resize(patch_scores[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                score_map = gaussian_filter(score_map, sigma=self.smoothing_sigma).astype(np.float32)
                score_maps.append(score_map)
                sample = data_samples[i] if data_samples is not None and i < len(data_samples) else None
                image_scores.append(self._reduce_image_score(patch_scores[i], score_map, sample))

            score_maps = torch.from_numpy(np.stack(score_maps, axis=0)).unsqueeze(1)
            img_scores = torch.tensor(image_scores, dtype=torch.float32)
            return build_predict_results(data_samples, img_scores, score_maps)

        if mode == 'tensor':
            return embeddings

        raise RuntimeError(f'Invalid mode "{mode}".')

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self
