"""SPADE: Sub-Image Anomaly Detection with Deep Pyramid Correspondences (arXiv 2020).

kNN-based anomaly detection on multi-scale pretrained features.
Training: collect all patch features into per-layer memory banks + GAP features.
Testing: per-layer kNN distance maps combined; image score = GAP-kNN distance.
"""
import logging

import torch
import torch.nn.functional as F
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import MemoryBankADModel

logger = logging.getLogger(__name__)


@MODELS.register_module()
class SPADEDetector(MemoryBankADModel):
    """SPADE anomaly detector using per-layer kNN on multi-scale features.

    Paper: Cohen & Hoshen, "Sub-Image Anomaly Detection with Deep Pyramid
    Correspondences", 2020.

    Args:
        backbone: Backbone name or config for feature extraction.
        k: Number of nearest neighbors for kNN.
        data_preprocessor: Data preprocessor config.
        init_cfg: Initialization config.
    """

    def __init__(self, backbone='wide_resnet50_2', k=5,
                 max_memory_bank_size=200000, knn_chunk_size=256,
                 knn_memory_chunk_size=8192,
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # Normalize backbone to FeatureExtractor config
        if isinstance(backbone, str):
            backbone = dict(type='FeatureExtractor', backbone_name=backbone,
                            out_indices=(1, 2, 3), frozen=True)
        elif isinstance(backbone, dict) and backbone.get('type') == 'RawBackbone':
            backbone = dict(type='FeatureExtractor',
                            backbone_name=backbone.get('backbone_name', 'wide_resnet50_2'),
                            pretrained=backbone.get('pretrained', True),
                            out_indices=(1, 2, 3), frozen=backbone.get('frozen', True))
        self.feature_extractor = MODELS.build(backbone)
        self.k = k
        self.max_memory_bank_size = max_memory_bank_size
        self.knn_chunk_size = knn_chunk_size
        self.knn_memory_chunk_size = knn_memory_chunk_size

        # Per-layer memory banks for pixel-level scoring
        # memory_banks[i]: (N*H*W, C_i)
        self.register_buffer('memory_bank_0', None)
        self.register_buffer('memory_bank_1', None)
        self.register_buffer('memory_bank_2', None)

        # GAP-based memory bank for image-level scoring (SPADE-pytorch reference)
        # memory_bank_gap: (N, C_total) - concatenated GAP features
        self.register_buffer('memory_bank_gap', None)

        # Runtime-only feature caches: NOT serialized in state_dict. Resuming
        # from a mid-training checkpoint should preserve the partially collected
        # features so fit() can still build the banks without a full replay.
        self._layer_features = [[], [], []]  # per-layer patch features
        self._gap_features = []  # GAP features for image-level scoring
        self._register_state_dict_hook(self._save_feature_cache_to_state_dict)
        self._register_load_state_dict_pre_hook(self._load_feature_cache_from_state_dict)

    @staticmethod
    def _save_feature_cache_to_state_dict(module, state_dict, prefix, local_metadata):
        for layer_idx, features in enumerate(module._layer_features):
            if not features:
                continue
            state_dict[f'{prefix}_layer_feature_cache_{layer_idx}'] = torch.cat(features, dim=0)
        if module._gap_features:
            state_dict[prefix + '_gap_feature_cache'] = torch.cat(module._gap_features, dim=0)
        return state_dict

    def _load_feature_cache_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs,
    ):
        self._layer_features = [[], [], []]
        for layer_idx in range(3):
            key = f'{prefix}_layer_feature_cache_{layer_idx}'
            if key in state_dict:
                self._layer_features[layer_idx] = [state_dict.pop(key).detach().cpu()]
        gap_key = prefix + '_gap_feature_cache'
        self._gap_features = []
        if gap_key in state_dict:
            self._gap_features = [state_dict.pop(gap_key).detach().cpu()]

    @torch.no_grad()
    def extract_features(self, x):
        """Extract multi-scale features.

        Returns:
            feats: list of 3 feature maps [f1, f2, f3], each (B, C_i, H_i, W_i)
        """
        return self.feature_extractor(x)  # [f1, f2, f3]

    def fit(self):
        """Build per-layer memory banks with optional random subsampling."""
        if not self._layer_features[0]:
            return
        for i in range(3):
            all_feats = torch.cat(self._layer_features[i], dim=0)  # (N*H*W, C_i)
            # Random subsampling if memory bank exceeds limit
            if self.max_memory_bank_size and all_feats.shape[0] > self.max_memory_bank_size:
                logger.info(
                    f'SPADE layer {i}: subsampling memory bank from '
                    f'{all_feats.shape[0]} to {self.max_memory_bank_size} patches')
                idx = torch.randperm(all_feats.shape[0])[:self.max_memory_bank_size]
                all_feats = all_feats[idx]
            setattr(self, f'memory_bank_{i}', all_feats)
            self._layer_features[i] = []

        # Build GAP memory bank for image-level scoring
        if self._gap_features:
            self.memory_bank_gap = torch.cat(self._gap_features, dim=0)
            logger.info(f'SPADE: built GAP memory bank with shape {self.memory_bank_gap.shape}')
            self._gap_features = []

    def _knn_kth_distance(self, queries, memory, k, chunk_size=None):
        """Compute K-th nearest neighbor distance (paper uses K-th, not mean).

        Args:
            queries: (Q, C) query features
            memory: (M, C) memory bank features
            k: number of neighbors
            chunk_size: query batch size for cdist; defaults to ``self.knn_chunk_size``.

        Returns:
            dists: (Q,) K-th NN distance for each query
        """
        if chunk_size is None:
            chunk_size = self.knn_chunk_size
        dists = []
        for start in range(0, queries.shape[0], chunk_size):
            end = min(start + chunk_size, queries.shape[0])
            q = queries[start:end]
            best_dists = None
            memory_chunk_size = min(self.knn_memory_chunk_size, memory.shape[0])
            for mem_start in range(0, memory.shape[0], memory_chunk_size):
                mem_end = min(mem_start + memory_chunk_size, memory.shape[0])
                d = torch.cdist(q, memory[mem_start:mem_end])  # (chunk, mem_chunk)
                local_k = min(k, d.shape[1])
                local_topk, _ = d.topk(local_k, dim=1, largest=False)
                if best_dists is None:
                    best_dists = local_topk
                else:
                    merged = torch.cat([best_dists, local_topk], dim=1)
                    merged_k = min(k, merged.shape[1])
                    best_dists, _ = merged.topk(merged_k, dim=1, largest=False)
            dists.append(best_dists[:, -1])  # K-th distance (last of top-K)
        return torch.cat(dists, dim=0)

    def build_memory_bank(self):
        """Called by MemoryBankHook after training to build memory banks."""
        self.fit()

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats = self.extract_features(inputs)  # [f1, f2, f3]

        if mode == 'loss':
            # Collect per-layer patch features
            target_h, target_w = feats[0].shape[2], feats[0].shape[3]
            for layer_idx, fm in enumerate(feats):
                B, C, H, W = fm.shape
                # Store patch features (resized to layer-0 resolution)
                if layer_idx > 0:
                    fm_resized = F.interpolate(fm, size=(target_h, target_w),
                                               mode='bilinear', align_corners=False)
                else:
                    fm_resized = fm
                patches = fm_resized.permute(0, 2, 3, 1).reshape(-1, C).cpu()
                self._layer_features[layer_idx].append(patches)

            # Collect GAP features for image-level scoring (SPADE-pytorch reference)
            gap_features = [F.adaptive_avg_pool2d(fm, (1, 1)).flatten(1) for fm in feats]
            global_feat = torch.cat(gap_features, dim=1).cpu()  # (B, C_total)
            self._gap_features.append(global_feat)

            return {'loss': torch.tensor(0.0, device=inputs.device, requires_grad=True)}

        elif mode == 'predict':
            if self.memory_bank_0 is None:
                raise RuntimeError(
                    'SPADE memory banks are not built. '
                    'Call build_memory_bank()/fit() before predict.'
                )

            B = inputs.shape[0]
            target_h, target_w = feats[0].shape[2], feats[0].shape[3]
            input_h, input_w = inputs.shape[-2], inputs.shape[-1]

            # Per-layer kNN anomaly maps
            combined_map = torch.zeros(B, 1, input_h, input_w, device=inputs.device)

            for layer_idx, fm in enumerate(feats):
                C = fm.shape[1]
                if layer_idx > 0:
                    fm_resized = F.interpolate(fm, size=(target_h, target_w),
                                               mode='bilinear', align_corners=False)
                else:
                    fm_resized = fm
                H, W = fm_resized.shape[2], fm_resized.shape[3]
                patches = fm_resized.permute(0, 2, 3, 1).reshape(B * H * W, C)

                memory = getattr(self, f'memory_bank_{layer_idx}').to(patches.device)
                layer_dists = self._knn_kth_distance(patches, memory, self.k)
                layer_map = layer_dists.reshape(B, 1, H, W)
                layer_map_up = F.interpolate(layer_map, size=(input_h, input_w),
                                             mode='bilinear', align_corners=False)
                combined_map = combined_map + layer_map_up

            # Image score: GAP-kNN distance against the GAP memory bank (SPADE paper);
            # falls back to anomaly-map max when GAP bank is unavailable.
            if self.memory_bank_gap is not None:
                gap_q = torch.cat(
                    [F.adaptive_avg_pool2d(fm, (1, 1)).flatten(1) for fm in feats],
                    dim=1,
                )
                gap_memory = self.memory_bank_gap.to(gap_q.device)
                k_gap = min(self.k, gap_memory.shape[0])
                img_scores = self._knn_kth_distance(gap_q, gap_memory, k_gap)
            else:
                img_scores = combined_map.amax(dim=(2, 3)).squeeze(1)

            return build_predict_results(data_samples, img_scores, combined_map)

        return feats

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self
