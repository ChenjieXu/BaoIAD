"""Dinomaly anomaly detector aligned to the official reference repository.

Reference: https://github.com/guojiajeremy/Dinomaly
"""

import copy
import math
from functools import partial
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.backbones.dinomaly_backbone import (
    DINOV2_ARCHITECTURES, DinomalyMLP, LinearAttention, DecoderViTBlock,
    _parse_encoder_name,
)
from baoiad.models.base_ad_model import KnowledgeDistillationADModel
from torch.nn.init import trunc_normal_

DEFAULT_FUSE_LAYERS = [[0, 1, 2, 3], [4, 5, 6, 7]]

# Inference defaults
DEFAULT_PREDICT_MAP_SIZE = 256
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 4
DEFAULT_IMAGE_SCORE_MAX_RATIO = 0.01

TRANSFORMER_CONFIG = {
    "mlp_ratio": 4.0,
    "layer_norm_eps": 1e-8,
    "qkv_bias": True,
    "attn_drop": 0.0,
}


# ========================== Gaussian blur ==================================

class GaussianBlur2d(nn.Module):
    """2D Gaussian blur."""

    def __init__(self, sigma, channels=1, kernel_size=5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = self._make_kernel(kernel_size, sigma)
        self.register_buffer("weight", kernel.repeat(channels, 1, 1, 1))

    @staticmethod
    def _make_kernel(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = torch.outer(g, g)
        return (g / g.sum()).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        pad = self.kernel_size // 2
        x = F.pad(x, [pad] * 4, mode="reflect")
        return F.conv2d(x, self.weight, groups=self.channels)


# ========================== Loss ===========================================

class CosineHardMiningLoss(nn.Module):
    """Cosine similarity loss with progressive hard mining."""

    def __init__(self, p_final=0.9, p_schedule_steps=1000, factor=0.1):
        super().__init__()
        self.p_final = p_final
        self.factor = factor
        self.p_schedule_steps = p_schedule_steps
        self.p = 0.0

    def forward(self, encoder_features, decoder_features, global_step):
        self.p = min(self.p_final * global_step / self.p_schedule_steps, self.p_final)
        cos_loss = nn.CosineSimilarity()
        loss = torch.tensor(0.0, device=encoder_features[0].device)
        for en_, de_ in zip(encoder_features, decoder_features):
            en_ = en_.detach()
            with torch.no_grad():
                point_dist = 1 - cos_loss(en_, de_).unsqueeze(1)
            k = max(1, int(point_dist.numel() * (1 - self.p)))
            thresh = torch.topk(point_dist.reshape(-1), k=k)[0][-1]
            loss += torch.mean(1 - cos_loss(en_.reshape(en_.shape[0], -1),
                                            de_.reshape(de_.shape[0], -1)))
            de_.register_hook(
                lambda x, idx=point_dist < thresh, f=self.factor: self._modify_grad(x, idx, f)
            )
        return loss / len(encoder_features)

    @staticmethod
    def _modify_grad(x, indices_to_modify, factor=0.0):
        indices_to_modify = indices_to_modify.expand_as(x)
        result = x.clone()
        result[indices_to_modify] = result[indices_to_modify] * factor
        return result


# ========================== Main Detector ==================================

@MODELS.register_module()
class DinomalyDetector(KnowledgeDistillationADModel):
    """Dinomaly: DINOv2 encoder + reconstruction decoder for anomaly detection.

    Args:
        encoder_name: DINOv2 variant, e.g. 'dinov2reg_vit_base_14'.
        bottleneck_dropout: Dropout for bottleneck MLP.
        decoder_depth: Number of decoder transformer layers.
        target_layers: Encoder layer indices to extract features from.
        fuse_layer_encoder: Layer groupings for encoder feature fusion.
        fuse_layer_decoder: Layer groupings for decoder feature fusion.
        remove_class_token: Whether to remove CLS token before processing.
    """

    def __init__(
        self,
        encoder_name="dinov2reg_vit_base_14",
        bottleneck_dropout=0.2,
        decoder_depth=8,
        target_layers=None,
        fuse_layer_encoder=None,
        fuse_layer_decoder=None,
        remove_class_token=False,
        loss_p_final=0.9,
        loss_schedule_steps=1000,
        loss_factor=0.1,
        predict_map_size=DEFAULT_PREDICT_MAP_SIZE,
        gaussian_kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
        gaussian_sigma=DEFAULT_GAUSSIAN_SIGMA,
        image_score_max_ratio=DEFAULT_IMAGE_SCORE_MAX_RATIO,
        backbone: Union[str, dict] = None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Resolve architecture config
        _, arch, _ = _parse_encoder_name(encoder_name)
        arch_config = DINOV2_ARCHITECTURES[arch]
        embed_dim = arch_config["embed_dim"]
        num_heads = arch_config["num_heads"]
        if target_layers is None:
            target_layers = list(arch_config["target_layers"])
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [list(g) for g in DEFAULT_FUSE_LAYERS]
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [list(g) for g in DEFAULT_FUSE_LAYERS]

        # Build encoder via registry
        if backbone is None:
            backbone_cfg = dict(type='DinomalyEncoder',
                                encoder_name=encoder_name, frozen=True)
        elif isinstance(backbone, dict):
            backbone_cfg = copy.deepcopy(backbone)
        else:
            raise ValueError(f"backbone must be None or dict, got {type(backbone)}")
        self.encoder = MODELS.build(backbone_cfg)

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            DinomalyMLP(in_features=embed_dim, hidden_features=embed_dim * 4,
                        out_features=embed_dim, act_layer=nn.GELU,
                        drop=bottleneck_dropout, bias=False, apply_input_dropout=True)
        ])

        # Decoder
        norm_layer = partial(nn.LayerNorm, eps=TRANSFORMER_CONFIG["layer_norm_eps"])
        self.decoder = nn.ModuleList([
            DecoderViTBlock(
                dim=embed_dim, num_heads=num_heads,
                mlp_ratio=TRANSFORMER_CONFIG["mlp_ratio"],
                qkv_bias=TRANSFORMER_CONFIG["qkv_bias"],
                norm_layer=norm_layer,
                attn_drop=TRANSFORMER_CONFIG["attn_drop"],
                attn=LinearAttention,
            )
            for _ in range(decoder_depth)
        ])

        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.predict_map_size = predict_map_size
        self.image_score_max_ratio = float(image_score_max_ratio)

        self.gaussian_blur = GaussianBlur2d(
            sigma=gaussian_sigma,
            channels=1,
            kernel_size=gaussian_kernel_size,
        )
        self.loss_fn = CosineHardMiningLoss(
            p_final=loss_p_final,
            p_schedule_steps=loss_schedule_steps,
            factor=loss_factor,
        )

        # Initialize trainable modules
        for m in list(self.bottleneck.modules()) + list(self.decoder.modules()):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.register_buffer('_global_step', torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _fuse_feature(feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def _get_encoder_decoder_outputs(self, x):
        # Auto-resize to nearest multiple of patch_size
        ps = self.encoder.patch_size
        _, _, h, w = x.shape
        new_h = round(h / ps) * ps
        new_w = round(w / ps) * ps
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

        tokens = self.encoder.prepare_tokens(x)
        encoder_features = []
        for i, block in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    tokens = block(tokens)
            else:
                continue
            if i in self.target_layers:
                encoder_features.append(tokens)

        side = int(math.sqrt(encoder_features[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            encoder_features = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in encoder_features]

        x_dec = self._fuse_feature(encoder_features)
        for block in self.bottleneck:
            x_dec = block(x_dec)

        decoder_features = []
        for block in self.decoder:
            x_dec = block(x_dec, attn_mask=None)
            decoder_features.append(x_dec)
        decoder_features = decoder_features[::-1]

        en = [self._fuse_feature([encoder_features[idx] for idx in idxs])
              for idxs in self.fuse_layer_encoder]
        de = [self._fuse_feature([decoder_features[idx] for idx in idxs])
              for idxs in self.fuse_layer_decoder]

        # Remove cls/register tokens and reshape to spatial
        def to_spatial(feats):
            if not self.remove_class_token:
                feats = [f[:, 1 + self.encoder.num_register_tokens:, :] for f in feats]
            B = feats[0].shape[0]
            return [f.permute(0, 2, 1).reshape(B, -1, side, side).contiguous() for f in feats]

        return to_spatial(en), to_spatial(de)

    @staticmethod
    def _calculate_anomaly_maps(source, target, out_size):
        if not isinstance(out_size, tuple):
            out_size = (out_size, out_size)
        anomaly_map_list = []
        for fs, ft in zip(source, target):
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = a_map.unsqueeze(1)
            a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
            anomaly_map_list.append(a_map)
        return torch.cat(anomaly_map_list, dim=1).mean(dim=1, keepdim=True), anomaly_map_list

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        en, de = self._get_encoder_decoder_outputs(inputs)
        image_size = inputs.shape[2]

        if mode == 'loss':
            self._global_step.add_(1)
            loss = self.loss_fn(en, de, self._global_step.item())
            return {'loss': loss}

        elif mode == 'predict':
            anomaly_map, _ = self._calculate_anomaly_maps(en, de, out_size=image_size)
            if self.predict_map_size is not None:
                anomaly_map = F.interpolate(
                    anomaly_map,
                    size=self.predict_map_size,
                    mode="bilinear",
                    align_corners=False,
                )
            anomaly_map = self.gaussian_blur(anomaly_map)

            if self.image_score_max_ratio <= 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                flat = anomaly_map.flatten(1)
                topk = max(1, int(flat.shape[1] * self.image_score_max_ratio))
                sp_score = torch.sort(flat, dim=1, descending=True)[0][:, :topk]
                sp_score = sp_score.mean(dim=1)

            return build_predict_results(data_samples, sp_score, anomaly_map)

        return en, de

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self
