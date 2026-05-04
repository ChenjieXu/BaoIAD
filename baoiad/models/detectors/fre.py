"""FRE: Feature Reconstruction Error anomaly detector."""
import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import ReconstructionADModel


class TiedAE(nn.Module):
    """Tied Autoencoder: encoder and decoder share (transposed) weights.

    Args:
        input_dim: Dimension of input features.
        latent_dim: Dimension of latent space.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(latent_dim, input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.encoder_bias = nn.Parameter(torch.zeros(latent_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = F.linear(features, self.weight, self.encoder_bias)
        return F.linear(encoded, self.weight.t(), self.decoder_bias)


# Layer name to timm out_index mapping
_LAYER_TO_INDEX = {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4}


@MODELS.register_module(force=True)
class FREDetector(ReconstructionADModel):
    """Feature Reconstruction Error for anomaly detection.

    Extracts features from a frozen pretrained backbone, then trains a
    tied autoencoder to reconstruct them. Anomaly is measured by the
    reconstruction error (squared difference).
    """

    def __init__(
        self,
        backbone: Union[str, dict] = "resnet50",
        layer: str = "layer3",
        layers: Sequence[str] | None = None,
        pooling_kernel_size: int = 2,
        pooling_kernel_sizes: Sequence[int] | None = None,
        input_dim: int = 65536,
        input_dims: Sequence[int] | None = None,
        latent_dim: int = 220,
        latent_dims: Sequence[int] | None = None,
        layer_weights: Sequence[float] | None = None,
        layer_fusion_mode: str = "weighted_sum",
        layer_norm_mode: str = "none",
        image_score_mode: str = "sum",
        topk_ratio: float = 0.01,
        loss=dict(type='MSELoss'),
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.layers = list(layers) if layers is not None else [layer]
        if not self.layers:
            raise ValueError("`layers` must not be empty.")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("`layers` must not contain duplicates.")
        for layer_name in self.layers:
            if layer_name not in _LAYER_TO_INDEX:
                raise ValueError(f"Unsupported layer {layer_name!r}.")

        self.layer = self.layers[0]
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_kernel_sizes = self._expand_per_layer_arg(
            pooling_kernel_sizes,
            pooling_kernel_size,
            "pooling_kernel_sizes",
        )
        self.input_dims = self._expand_per_layer_arg(
            input_dims,
            input_dim,
            "input_dims",
        )
        self.latent_dims = self._expand_per_layer_arg(
            latent_dims,
            latent_dim,
            "latent_dims",
        )
        if any(kernel_size < 1 for kernel_size in self.pooling_kernel_sizes):
            raise ValueError("All pooling kernel sizes must be >= 1.")

        self.layer_fusion_mode = layer_fusion_mode
        self.layer_norm_mode = layer_norm_mode
        self.image_score_mode = image_score_mode
        self.topk_ratio = topk_ratio
        self.layer_weights = self._build_layer_weights(layer_weights)
        self._use_timm = False

        layer_indices = tuple(_LAYER_TO_INDEX[layer_name] for layer_name in self.layers)

        # anomalib's FREModel creates TiedAE BEFORE the feature extractor, so
        # TiedAE's ``xavier_uniform_`` starts from the clean RNG state (seed).
        # In BaoIAD, TIMMBackbone is built first, consuming RNG.  Save the
        # RNG state here so we can restore it before TiedAE creation.
        _use_timm = isinstance(backbone, dict) and backbone.get('type') == 'TIMMBackbone'
        _rng_state = torch.random.get_rng_state() if _use_timm else None

        if _use_timm:
            # TIMMBackbone: request exactly the configured layers.
            backbone.setdefault('out_indices', layer_indices)
            backbone.setdefault('frozen', True)
            self.backbone = MODELS.build(backbone)
            self._use_timm = True
        else:
            # Legacy: RawBackbone with manual layer traversal
            if isinstance(backbone, dict):
                net = MODELS.build(backbone)
            else:
                net = MODELS.build(dict(type='RawBackbone', backbone_name=backbone))
            self._prefix = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self._blocks = nn.ModuleDict()
            for name in ("layer1", "layer2", "layer3", "layer4"):
                if hasattr(net, name):
                    self._blocks[name] = getattr(net, name)

            # Freeze backbone
            for p in self._prefix.parameters():
                p.requires_grad = False
            for p in self._blocks.parameters():
                p.requires_grad = False

        # Restore RNG so TiedAE gets the same seed-origin state as anomalib.
        if _rng_state is not None:
            torch.random.set_rng_state(_rng_state)

        self.tied_aes = nn.ModuleList(
            [TiedAE(layer_input_dim, layer_latent_dim)
             for layer_input_dim, layer_latent_dim in zip(self.input_dims, self.latent_dims)]
        )
        self.tied_ae = self.tied_aes[0]
        self.loss_fn = MODELS.build(loss)

    @torch.no_grad()
    def _extract_backbone_features(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Size]]:
        """Extract pooled features from each configured backbone layer.

        Returns:
            Lists of flattened features and their spatial shapes.
        """
        if self._use_timm:
            feats = self.backbone(x)
            layer_outputs = dict(zip(self.layers, feats))
        else:
            out = self._prefix(x)
            layer_outputs = {}
            for name in ("layer1", "layer2", "layer3", "layer4"):
                if name not in self._blocks:
                    break
                out = self._blocks[name](out)
                if name in self.layers:
                    layer_outputs[name] = out
                if len(layer_outputs) == len(self.layers):
                    break

        flat_features: List[torch.Tensor] = []
        feature_shapes: List[torch.Size] = []
        for layer_name, pooling_kernel_size in zip(self.layers, self.pooling_kernel_sizes):
            out = layer_outputs[layer_name]
            batch_size = out.shape[0]
            if pooling_kernel_size > 1:
                out = F.avg_pool2d(out, kernel_size=pooling_kernel_size)
            feature_shapes.append(out.shape)
            flat_features.append(out.view(batch_size, -1).detach())
        return flat_features, feature_shapes

    def _get_features(self, x: torch.Tensor):
        """Extract features and reconstruct them via tied AEs.

        Returns:
            Lists of inputs, reconstructions and feature shapes.
        """
        features_in, feature_shapes = self._extract_backbone_features(x)
        features_out = [ae(layer_features) for ae, layer_features in zip(self.tied_aes, features_in)]
        return features_in, features_out, feature_shapes

    def _compute_layer_anomaly_maps(
        self,
        features_in: Sequence[torch.Tensor],
        features_out: Sequence[torch.Tensor],
        feature_shapes: Sequence[torch.Size],
        output_size: tuple[int, int],
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        lowres_maps: List[torch.Tensor] = []
        layer_maps: List[torch.Tensor] = []
        for layer_in, layer_out, feature_shape in zip(features_in, features_out, feature_shapes):
            fre = torch.square(layer_in - layer_out).reshape(feature_shape)
            lowres_map = torch.sum(fre, dim=1, keepdim=True)
            anomaly_map = F.interpolate(
                lowres_map,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            lowres_maps.append(lowres_map)
            layer_maps.append(self._normalize_layer_map(anomaly_map))
        return lowres_maps, layer_maps

    def _use_pre_upsample_score(self) -> bool:
        """Whether image scores should be reduced before map upsampling.

        Official anomalib FRE computes image-level scores from the native pooled
        residual map and only upsamples the anomaly map for visualization /
        pixel-level evaluation. Restrict this behavior to the strict single-layer
        path so multi-layer experimental variants keep their existing semantics.
        """
        return (
            len(self.layers) == 1
            and self.layer_norm_mode == "none"
            and self.image_score_mode == "sum"
        )

    def _normalize_layer_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        if self.layer_norm_mode == "none":
            return anomaly_map

        flat_map = anomaly_map.flatten(2)
        if self.layer_norm_mode == "zscore":
            mean = flat_map.mean(dim=2, keepdim=True)
            std = flat_map.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
            return ((flat_map - mean) / std).reshape_as(anomaly_map)
        if self.layer_norm_mode == "minmax":
            min_value = flat_map.min(dim=2, keepdim=True).values
            max_value = flat_map.max(dim=2, keepdim=True).values
            return ((flat_map - min_value) / (max_value - min_value).clamp_min(1e-6)).reshape_as(anomaly_map)
        raise ValueError(f"Unsupported layer_norm_mode {self.layer_norm_mode!r}.")

    def _fuse_layer_maps(self, layer_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(layer_maps) == 1:
            return layer_maps[0]

        stacked_maps = torch.stack(list(layer_maps), dim=0)
        if self.layer_fusion_mode == "weighted_sum":
            fused_map = torch.zeros_like(layer_maps[0])
            for weight, layer_map in zip(self.layer_weights, layer_maps):
                fused_map = fused_map + weight * layer_map
            return fused_map
        if self.layer_fusion_mode == "sum":
            return stacked_maps.sum(dim=0)
        if self.layer_fusion_mode == "mean":
            return stacked_maps.mean(dim=0)
        if self.layer_fusion_mode == "max":
            return stacked_maps.max(dim=0).values
        if self.layer_fusion_mode == "p95":
            return torch.quantile(stacked_maps, 0.95, dim=0)
        raise ValueError(f"Unsupported layer_fusion_mode {self.layer_fusion_mode!r}.")

    def _reduce_map_to_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        flat_map = anomaly_map.flatten(1)
        if self.image_score_mode == "sum":
            return flat_map.sum(dim=1)
        if self.image_score_mode == "mean":
            return flat_map.mean(dim=1)
        if self.image_score_mode == "max":
            return flat_map.max(dim=1).values
        if self.image_score_mode == "p95":
            return torch.quantile(flat_map, 0.95, dim=1)
        if self.image_score_mode == "topk_mean":
            if not 0 < self.topk_ratio <= 1:
                raise ValueError("`topk_ratio` must be in (0, 1].")
            k = max(1, math.ceil(flat_map.shape[1] * self.topk_ratio))
            return flat_map.topk(k, dim=1).values.mean(dim=1)
        raise ValueError(f"Unsupported image_score_mode {self.image_score_mode!r}.")

    def _expand_per_layer_arg(
        self,
        values: Sequence[int] | None,
        fallback: int,
        name: str,
    ) -> List[int]:
        if values is None:
            return [fallback] * len(self.layers)
        expanded = list(values)
        if len(expanded) != len(self.layers):
            raise ValueError(
                f"`{name}` must have the same length as `layers`: "
                f"{len(expanded)} != {len(self.layers)}."
            )
        return expanded

    def _build_layer_weights(self, layer_weights: Sequence[float] | None) -> List[float]:
        if layer_weights is None:
            return [1.0 / len(self.layers)] * len(self.layers)
        weights = list(layer_weights)
        if len(weights) != len(self.layers):
            raise ValueError(
                f"`layer_weights` must have the same length as `layers`: "
                f"{len(weights)} != {len(self.layers)}."
            )
        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError("`layer_weights` must sum to a positive value.")
        return [float(weight) / total_weight for weight in weights]

    def forward(self, inputs, data_samples=None, mode="tensor"):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == "loss":
            features_in, features_out, _ = self._get_features(inputs)
            layer_losses = [self.loss_fn(layer_in, layer_out)
                            for layer_in, layer_out in zip(features_in, features_out)]
            loss = sum(weight * layer_loss for weight, layer_loss in zip(self.layer_weights, layer_losses))
            return {"loss": loss}

        elif mode == "predict":
            features_in, features_out, feature_shapes = self._get_features(inputs)
            lowres_layer_maps, layer_maps = self._compute_layer_anomaly_maps(
                features_in,
                features_out,
                feature_shapes,
                inputs.shape[-2:],
            )
            anomaly_map = self._fuse_layer_maps(layer_maps)
            if self._use_pre_upsample_score():
                score_map = self._fuse_layer_maps(lowres_layer_maps)
                score = self._reduce_map_to_score(score_map)
            else:
                score = self._reduce_map_to_score(anomaly_map)
            return build_predict_results(data_samples, score, anomaly_map)

        # mode == 'tensor'
        features_in, features_out, _ = self._get_features(inputs)
        if len(features_in) == 1:
            return features_in[0], features_out[0]
        return features_in, features_out

    def train(self, mode=True):
        super().train(mode)
        if self._use_timm:
            self.backbone.eval()
        else:
            self._prefix.eval()
            for m in self._blocks.values():
                m.eval()
        return self
