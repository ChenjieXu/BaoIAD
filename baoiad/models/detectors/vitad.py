"""ViTAD (ViT-based Anomaly Detection) detector.

Faithful reimplementation based on ADer's ViTAD:
- ViT/DeiT encoder (not WRN50) as teacher backbone
- Fusion module: linear layer to fuse intermediate ViT features
- ViT decoder with configurable depth
- Cosine distance loss
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Union
from mmengine.model import BaseModel
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.backbones.vitad_backbone import VisionTransformer, trunc_normal_
from baoiad.models.base_ad_model import BaseADModel


# ========================= Fusion =========================

class Fusion(nn.Module):
    """Linear fusion of multiple ViT intermediate features (B, L, C*mul) -> (B, L, C)."""
    def __init__(self, dim, mul):
        super().__init__()
        self.fc = nn.Linear(dim * mul, dim)

    def forward(self, features):
        """features: list of (B, L, C) tensors."""
        feature_align = torch.cat(features, dim=2)  # B, L, C*mul
        return self.fc(feature_align)


# ========================= ViT Decoder =========================

class ViTDecoder(VisionTransformer):
    """ViT-based decoder that outputs multi-scale features from specified blocks.

    Takes token input (B, L, C), runs through transformer blocks,
    and returns 2D feature maps from specified 'students' blocks.
    """
    def __init__(self, students, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.students = students
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        """x: (B, L, C) token input."""
        x = x + self.pos_embed
        x = self.pos_drop(x)

        out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if (idx + 1) in self.students:
                fea = x
                B, L, C = fea.shape
                H = int(np.sqrt(L))
                fea_2d = fea.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
                out.append(fea_2d)

        # Reverse order: from deep to shallow
        return [out[len(out) - 1 - i] for i in range(len(out))]


# ========================= ViTAD Detector =========================

# ========================= Decoder Builder =========================

def _build_decoder(embed_dim, img_size, students, depth, num_heads=None,
                   patch_size=16, mlp_ratio=4.0, qkv_bias=True, norm_eps=1e-6):
    """Build matching decoder for encoder."""
    if num_heads is None:
        num_heads = 6 if embed_dim == 384 else 12 if embed_dim == 768 else 3
    model = ViTDecoder(
        students=students,
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
        depth=depth, num_heads=num_heads,
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        norm_layer=partial(nn.LayerNorm, eps=norm_eps))
    return model


def _gaussian_blur_bchw(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply ADer's scipy gaussian_filter to each sample/channel map."""
    if sigma <= 0:
        return x
    from scipy.ndimage import gaussian_filter

    x_np = x.detach().float().cpu().numpy()
    for batch_index in range(x_np.shape[0]):
        for channel_index in range(x_np.shape[1]):
            x_np[batch_index, channel_index] = gaussian_filter(
                x_np[batch_index, channel_index], sigma=float(sigma))
    return torch.from_numpy(x_np).to(device=x.device, dtype=x.dtype)


def _vitad_cos_loss(feats_t, feats_s) -> torch.Tensor:
    """Match ADer's CosLoss(flat=True, avg=False) implementation."""
    loss = feats_t[0].new_tensor(0.0)
    for ft, fs in zip(feats_t, feats_s):
        ft_flat = ft.contiguous().view(ft.shape[0], -1)
        fs_flat = fs.contiguous().view(fs.shape[0], -1)
        loss = loss + (1 - F.cosine_similarity(ft_flat, fs_flat, dim=1)).mean()
    return loss


def _vitad_score_map(feats_t, feats_s, out_size, gaussian_sigma: float) -> torch.Tensor:
    """Match ADer's cal_anomaly_map(..., amap_mode='add', gaussian_sigma=4)."""
    scores = []
    for ft, fs in zip(feats_t, feats_s):
        dist = 1 - F.cosine_similarity(ft, fs, dim=1)
        dist = F.interpolate(
            dist.unsqueeze(1),
            size=out_size,
            mode='bilinear',
            align_corners=True,
        ).squeeze(1)
        scores.append(dist)
    num_scales = len(scores)
    score_map = sum(scores) / (num_scales * num_scales)
    if gaussian_sigma > 0:
        score_map = _gaussian_blur_bchw(
            score_map.unsqueeze(1), float(gaussian_sigma)).squeeze(1)
    return score_map


def _vitad_image_scores(score_map: torch.Tensor, pooling_kernel_size: int = 16) -> torch.Tensor:
    """Match ADer's evaluator pooling_ks=[16, 16] + max image scoring."""
    pooled = F.avg_pool2d(
        score_map.unsqueeze(1),
        kernel_size=pooling_kernel_size,
        stride=1,
        padding=0,
    )
    return pooled.view(pooled.shape[0], -1).max(dim=1).values


# ========================= Encoder name → config helper ===================

@MODELS.register_module(force=True)
class ViTADDetector(BaseADModel):
    """ViTAD: ViT-based Anomaly Detection.

    Faithful reimplementation based on ADer's ViTAD:
    - ViT/DeiT teacher encoder extracts multi-block intermediate features
    - Fusion module concatenates neck features and projects via linear layer
    - ViT decoder reconstructs teacher features from fused representation
    - Cosine distance loss between teacher and student outputs

    Args:
        encoder_name: Name of ViT encoder variant.
        img_size: Input image size.
        teachers: List of block indices to extract teacher features from.
        neck: List of block indices for neck features fed to fusion.
        students: List of block indices for decoder output extraction.
        decoder_depth: Number of transformer blocks in decoder.
        fusion_mul: Multiplier for fusion input dim (number of neck features).
        pretrained: Whether to load pretrained encoder weights.
    """

    def __init__(self, encoder_name='vit_small_patch16_224_dino', img_size=256,
                 teachers=(3, 6, 9), neck=(12,), students=(3, 6, 9),
                 decoder_depth=9, fusion_mul=None, pretrained=True,
                 gaussian_sigma: float = 4.0,
                 backbone: Union[str, dict] = None,
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.gaussian_sigma = gaussian_sigma

        self.teachers_idx = list(teachers)
        self.neck_idx = list(neck)
        self.students_idx = list(students)

        # Build encoder via registry
        if backbone is not None:
            if isinstance(backbone, dict):
                backbone_cfg = copy.deepcopy(backbone)
            else:
                raise ValueError(f"backbone must be None or dict, got {type(backbone)}")
            # Inject teachers/neck if not in config
            backbone_cfg.setdefault('teachers', self.teachers_idx)
            backbone_cfg.setdefault('neck', self.neck_idx)
            backbone_cfg.setdefault('img_size', img_size)
            self.net_t = MODELS.build(backbone_cfg)
            embed_dim = self.net_t.embed_dim
        else:
            # Legacy: build from encoder_name string
            from baoiad.models.backbones.vitad_backbone import get_vitad_encoder_config
            backbone_cfg = get_vitad_encoder_config(
                encoder_name, img_size, self.teachers_idx, self.neck_idx, pretrained)
            self.net_t = MODELS.build(backbone_cfg)
            embed_dim = self.net_t.embed_dim

        # ADer uses mul=1 with neck=[12]; infer from neck size when not set.
        if fusion_mul is None:
            fusion_mul = max(len(self.neck_idx), 1)
        if fusion_mul != max(len(self.neck_idx), 1):
            raise ValueError(
                f'fusion_mul={fusion_mul} does not match number of neck features '
                f'({len(self.neck_idx)}).'
            )

        # Fusion: concat neck features → linear projection
        self.net_fusion = Fusion(dim=embed_dim, mul=fusion_mul)

        # Build decoder with teacher-aligned transformer hyperparameters.
        decoder_heads = getattr(self.net_t.blocks[0].attn, 'num_heads', None) if len(self.net_t.blocks) > 0 else None
        decoder_mlp_ratio = 4.0
        decoder_qkv_bias = True
        decoder_norm_eps = 1e-6
        if len(self.net_t.blocks) > 0:
            block0 = self.net_t.blocks[0]
            if hasattr(block0, 'mlp') and hasattr(block0.mlp, 'fc1'):
                decoder_mlp_ratio = float(block0.mlp.fc1.out_features) / float(block0.mlp.fc1.in_features)
            if hasattr(block0, 'attn') and hasattr(block0.attn, 'qkv'):
                decoder_qkv_bias = block0.attn.qkv.bias is not None
        if hasattr(self.net_t, 'norm') and hasattr(self.net_t.norm, 'eps'):
            decoder_norm_eps = float(self.net_t.norm.eps)
        teacher_patch_size = 16
        if hasattr(self.net_t, 'patch_embed') and hasattr(self.net_t.patch_embed, 'patch_size'):
            patch = self.net_t.patch_embed.patch_size
            teacher_patch_size = patch[0] if isinstance(patch, tuple) else int(patch)
        self.net_s = _build_decoder(embed_dim, img_size, self.students_idx,
                                    decoder_depth, num_heads=decoder_heads,
                                    patch_size=teacher_patch_size,
                                    mlp_ratio=decoder_mlp_ratio,
                                    qkv_bias=decoder_qkv_bias,
                                    norm_eps=decoder_norm_eps)

        # Freeze teacher
        for param in self.net_t.parameters():
            param.requires_grad = False

        # Construction already matches the official initialized state.
        # Letting MMEngine call BaseModel.init_weights() would recurse into the
        # timm ViT modules and reinitialize teacher/decoder weights again.
        self._is_init = True

    def init_weights(self):
        """Skip MMEngine's recursive re-initialization.

        ViTAD builds and initializes all submodules eagerly in ``__init__``.
        The default BaseModel.init_weights() would call child init_weights()
        again, which re-randomizes the timm encoder/decoder modules and breaks
        strict alignment with the official ADer path.
        """
        self._is_init = True

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        # Teacher forward
        with torch.no_grad():
            feats_t, feats_neck = self.net_t(inputs)
            feats_t = [f.detach() for f in feats_t]
            feats_neck = [f.detach() for f in feats_neck]

        # Fusion + Decoder
        fused = self.net_fusion(feats_neck)  # B, L, C
        feats_s = self.net_s(fused)  # list of (B, C, H, W), reversed order

        if mode == 'loss':
            return {'loss': _vitad_cos_loss(feats_t, feats_s)}

        elif mode == 'predict':
            score_map = _vitad_score_map(
                feats_t,
                feats_s,
                out_size=inputs.shape[-2:],
                gaussian_sigma=float(self.gaussian_sigma),
            )
            img_scores = _vitad_image_scores(score_map)
            return build_predict_results(data_samples, img_scores, score_map)

        return feats_t, feats_s

    def train(self, mode=True):
        super().train(mode)
        self.net_t.eval()
        return self
