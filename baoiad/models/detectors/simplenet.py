"""SimpleNet anomaly detector with legacy and strict official paths."""

from __future__ import annotations

import math
from typing import Optional

import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.optim import OptimWrapperDict

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import DiscriminatorADModel


class PatchMaker:
    """Extracts local neighborhood patches (patchsize x patchsize) around each spatial location."""

    def __init__(self, patchsize, stride=1, top_k=0):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    @staticmethod
    def unpatch_scores(scores, batchsize):
        return scores.reshape(batchsize, -1, *scores.shape[1:])

    def score(self, scores):
        while scores.ndim > 2:
            scores = torch.max(scores, dim=-1).values
        if scores.ndim == 2:
            if self.top_k > 1:
                scores = torch.topk(scores, self.top_k, dim=1).values.mean(1)
            else:
                scores = torch.max(scores, dim=1).values
        return scores


class MeanMapper(nn.Module):
    """Adaptive average pooling on flattened patch features."""

    def __init__(self, preprocessing_dim):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(nn.Module):
    """Per-layer adaptive pooling to a common dimension, then stack."""

    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.preprocessing_modules = nn.ModuleList()
        for input_dim in input_dims:
            self.preprocessing_modules.append(MeanMapper(output_dim))

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class Aggregator(nn.Module):
    """Aggregate stacked layer features into target_dim via adaptive avg pool."""

    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features):
        # features: (B*HW, n_layers, preprocessing_dim) -> (B*HW, target_dim)
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class Projection(nn.Module):
    """Single-layer linear projection (n_layers=1, layer_type=0 → no activation)."""

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super().__init__()
        if out_planes is None:
            out_planes = in_planes
        self.layers = nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", nn.LeakyReLU(0.2))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    """SimpleNet discriminator: hidden=1024, n_layers=2 (1 hidden block + 1 tail)."""

    def __init__(self, in_planes=1536, n_layers=2, hidden=1024):
        super().__init__()
        _hidden = in_planes if hidden is None else hidden
        self.body = nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module(f'block{i + 1}', nn.Sequential(
                nn.Linear(_in, _hidden),
                nn.BatchNorm1d(_hidden),
                nn.LeakyReLU(0.2)
            ))
        self.tail = nn.Linear(_hidden, 1, bias=False)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class RescaleSegmentor:
    """Official SimpleNet score-map upsampling with Gaussian smoothing."""

    def __init__(self, target_size: Optional[int | tuple[int, int]] = None, smoothing: float = 4.0):
        self.target_size = target_size
        self.smoothing = smoothing

    def convert_to_segmentation(self, patch_scores, target_size: Optional[int | tuple[int, int]] = None):
        size = target_size or self.target_size
        if size is None:
            raise ValueError('RescaleSegmentor requires a target_size.')

        with torch.no_grad():
            if not torch.is_tensor(patch_scores):
                patch_scores = torch.as_tensor(patch_scores)
            scores = patch_scores.float().unsqueeze(1)
            scores = F.interpolate(
                scores,
                size=size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            maps = scores.cpu().numpy()
        return [ndimage.gaussian_filter(score_map, sigma=self.smoothing) for score_map in maps]


@MODELS.register_module(force=True)
class SimpleNetDetector(DiscriminatorADModel):
    """SimpleNet: feature extraction + patch aggregation + projection + discriminator.

    Faithful to the CVPR 2023 paper:
    - Patch-level neighborhood aggregation (patchsize=3)
    - Preprocessing + Aggregator pipeline
    - Projection: single Linear layer, no activation (n_layers=1, layer_type=0)
    - Mix noise with noise_std * 1.1^k
    - Discriminator: hidden=1024, n_layers=2
    """

    def __init__(self, backbone='wide_resnet50_2', target_dim=1536, pretrain_embed_dim=1536,
                 noise_std=0.015, mix_noise=1, dsc_margin=0.5,
                 patchsize=3, patchstride=1,
                 dsc_layers=2, dsc_hidden=1024,
                 pre_proj=1, proj_layer_type=0,
                 strict: bool = False,
                 image_size: Optional[int | tuple[int, int]] = None,
                 gaussian_sigma: float = 4.0,
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # Backward compat: str -> config dict
        if isinstance(backbone, str):
            backbone = dict(type='FeatureExtractor', backbone_name=backbone,
                            out_indices=(2, 3), frozen=True)
        self.feature_extractor = MODELS.build(backbone)
        ch = self.feature_extractor.channels  # (layer2_ch, layer3_ch)

        # Patch maker for neighborhood aggregation
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        # Preprocessing: per-layer adaptive pooling to pretrain_embed_dim
        self.preprocessing = Preprocessing(
            input_dims=[ch[0], ch[1]],  # layer2, layer3
            output_dim=pretrain_embed_dim
        )

        # Aggregator: merge stacked layer features to target_dim
        self.aggregator = Aggregator(target_dim=target_dim)

        # Projection: single linear, no activation (paper: n_layers=1, layer_type=0)
        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.projection = Projection(target_dim, target_dim, n_layers=pre_proj, layer_type=proj_layer_type)

        # Discriminator: hidden=1024, n_layers=2
        self.discriminator = Discriminator(in_planes=target_dim, n_layers=dsc_layers, hidden=dsc_hidden)

        self.noise_std = noise_std
        self.mix_noise = mix_noise
        self.dsc_margin = dsc_margin
        self.strict = strict
        self.segmentor = RescaleSegmentor(target_size=image_size, smoothing=gaussian_sigma) if strict else None

    @staticmethod
    def _prepare_inputs(inputs):
        if isinstance(inputs, (list, tuple)):
            return torch.stack(list(inputs))
        return inputs

    @torch.no_grad()
    def extract_features(self, x):
        """Extract features with patch-level neighborhood aggregation."""
        raw_features = self.feature_extractor(x)  # [layer2, layer3]
        B = raw_features[0].shape[0]
        features = raw_features

        # Patchify each layer (neighborhood aggregation)
        features = [
            self.patch_maker.patchify(feat, return_spatial_info=True) for feat in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        # Align spatial sizes of later layers to first layer
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # Preprocessing: per-layer pooling + stack
        features = self.preprocessing(features)
        # Aggregator: merge to target_dim
        features = self.aggregator(features)

        H, W = ref_num_patches
        return features, B, H, W

    def train_step(self, data, optim_wrapper):
        if not self.strict:
            return super().train_step(data, optim_wrapper)

        if not isinstance(optim_wrapper, OptimWrapperDict) or 'discriminator' not in optim_wrapper:
            raise TypeError(
                'Strict SimpleNet training requires an OptimWrapperDict with a discriminator optimizer.'
            )

        data = self.data_preprocessor(data, True)
        inputs = self._prepare_inputs(data['inputs'])
        data_samples = data.get('data_samples', None)

        projection_optim = optim_wrapper['projection'] if 'projection' in optim_wrapper else None
        discriminator_optim = optim_wrapper['discriminator']

        if projection_optim is not None:
            projection_optim.zero_grad()
        discriminator_optim.zero_grad()

        losses = self(inputs, data_samples, mode='loss')
        scaled_loss = discriminator_optim.scale_loss(losses['loss'])
        discriminator_optim.backward(scaled_loss)

        if projection_optim is not None:
            projection_optim._inner_count += 1
            if projection_optim.should_update():
                projection_optim.step()
                projection_optim.zero_grad()

        if discriminator_optim.should_update():
            discriminator_optim.step()
            discriminator_optim.zero_grad()

        return {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in losses.items()
        }

    def forward(self, inputs, data_samples=None, mode='tensor'):
        inputs = self._prepare_inputs(inputs)
        feats, B, H, W = self.extract_features(inputs)

        if self.pre_proj > 0:
            feats = self.projection(feats)

        if mode == 'loss':
            # Mix noise: noise_std * 1.1^k for k in [0, mix_noise)
            noise_idxs = torch.randint(0, self.mix_noise, torch.Size([feats.shape[0]]))
            noise_one_hot = F.one_hot(noise_idxs, num_classes=self.mix_noise).to(feats.device)  # (N, K)
            noise = torch.stack([
                torch.normal(0, self.noise_std * 1.1 ** k, feats.shape)
                for k in range(self.mix_noise)
            ], dim=1).to(feats.device)  # (N, K, C)
            noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
            fake_feats = feats + noise

            scores = self.discriminator(torch.cat([feats, fake_feats]))
            true_scores = scores[:feats.shape[0]]
            fake_scores = scores[feats.shape[0]:]

            th = self.dsc_margin
            loss = torch.clamp(-true_scores + th, min=0).mean() + torch.clamp(fake_scores + th, min=0).mean()
            return {'loss': loss}
        elif mode == 'predict':
            patch_scores = -self.discriminator(feats).squeeze(-1)
            if self.strict:
                score_map = self.patch_maker.unpatch_scores(patch_scores, batchsize=B).reshape(B, H, W)
                maps = self.segmentor.convert_to_segmentation(
                    score_map,
                    target_size=tuple(int(dim) for dim in inputs.shape[-2:]),
                )
                img_scores = self.patch_maker.score(
                    self.patch_maker.unpatch_scores(patch_scores, batchsize=B)
                )
                return build_predict_results(data_samples, img_scores, maps)

            score_map = patch_scores.reshape(B, H, W)
            img_scores = score_map.view(B, -1).max(dim=1).values
            return build_predict_results(data_samples, img_scores, score_map)
        return feats

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self
