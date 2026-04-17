"""UniAD (Unified Anomaly Detection, NeurIPS 2022) detector — faithful to paper.

Supports both WRN-50 (via RawBackbone) and TIMMBackbone (e.g., EfficientNet-B4).
ADer reference uses EfficientNet-B4 with 4 layers (out_indices=[0,1,2,3]).
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.models.base_ad_model import BaseADModel
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        self.feature_size = feature_size

    def forward(self, device):
        h, w = self.feature_size
        i = torch.arange(w, device=device)
        j = torch.arange(h, device=device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).expand(h, -1, -1),
            y_emb.unsqueeze(1).expand(-1, w, -1),
        ], dim=-1).flatten(0, 1)
        return pos


class UniADEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, pos, mask=None):
        q = k = src + pos
        src2 = self.self_attn(q, k, value=src, attn_mask=mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class UniADDecoderLayer(nn.Module):
    def __init__(self, d_model, feature_size, nhead, dim_feedforward=1024, dropout=0.1, activation='relu'):
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, out, memory, pos, tgt_mask=None, mem_mask=None):
        B = memory.shape[1]
        tgt = self.learned_embed.weight.unsqueeze(1).expand(-1, B, -1)
        tgt2 = self.self_attn(tgt + pos, memory + pos, memory, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt + pos, out + pos, out, attn_mask=mem_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MFCN(nn.Module):
    """Multi-scale Feature Concatenation Network.

    Resizes all feature scales to the COARSEST resolution and concatenates.
    This follows ADer's UniAD implementation which aligns all scales to
    stride 16 for 256px inputs.
    """

    def __init__(self, outstride=16):
        super().__init__()
        self.outstride = outstride

    def forward(self, features):
        """features: list of [B, C_i, H_i, W_i] tensors at different scales.

        All features are resized to the COARSEST resolution (last feature's size).
        """
        target_size = features[-1].shape[2:]  # coarsest resolution
        feature_list = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            feature_list.append(feat)
        return torch.cat(feature_list, dim=1)


def _generate_neighbor_mask(feature_size, neighbor_size):
    h, w = feature_size
    hm, wm = neighbor_size
    mask = torch.ones(h, w, h, w)
    for i in range(h):
        for j in range(w):
            h_s, h_e = max(i - hm // 2, 0), min(i + hm // 2 + 1, h)
            w_s, w_e = max(j - wm // 2, 0), min(j + wm // 2 + 1, w)
            mask[i, j, h_s:h_e, w_s:w_e] = 0
    mask = mask.view(h * w, h * w)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu/glu, not {activation}.')


@MODELS.register_module()
class UniADDetector(BaseADModel):
    """UniAD: Unified Anomaly Detection (NeurIPS 2022).

    Backbone → MFCN (resize all scales to coarsest resolution) →
    Transformer encoder-decoder → reconstruct features.
    Anomaly = L2 reconstruction error.

    Supports both:
    - WRN-50 backbone via RawBackbone (3 layers: layer1, layer2, layer3)
    - TIMMBackbone (e.g., EfficientNet-B4 with 4 layers)

    Key params aligned with ADer implementation:
    - feature_size=(16, 16) — stride 16 resolution for 256×256 input
    - neighbor_size=(8, 8) — 256 // 32 for the frozen MUAD benchmark config
    - hidden_dim=256, nhead=8, 4 enc + 4 dec layers
    - feature_jitter_scale=20.0
    - image_score_mode='pooled_max' — ADer-style avg_pool2d(16) before max
    """

    def __init__(self,
                 backbone,  # dict config for backbone (TIMMBackbone or RawBackbone)
                 hidden_dim: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 normalize_before: bool = False,
                 feature_jitter_scale: float = 20.0,
                 feature_jitter_prob: float = 1.0,
                 neighbor_size: tuple = (8, 8),
                 use_neighbor_mask: bool = True,
                 neighbor_mask_layers: tuple = (True, True, True),  # enc, dec_tgt, dec_mem
                 feature_size: tuple = (16, 16),
                 image_score_mode: str = 'pooled_max',
                 image_score_topk: int = 64,
                 image_score_pool_kernel: int = 16,
                 loss=dict(type='MSELoss'),
                 data_preprocessor=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.loss_fn = MODELS.build(loss)

        # Build backbone - supports both TIMMBackbone and RawBackbone (WRN-50)
        if isinstance(backbone, dict):
            self.backbone = MODELS.build(backbone)
        else:
            # Legacy: string backbone name for WRN-50
            self.backbone = MODELS.build(dict(type='RawBackbone', backbone_name=backbone))

        self.feature_size = feature_size
        self.neighbor_size = neighbor_size
        self.use_neighbor_mask = use_neighbor_mask
        valid_score_modes = {
            'pooled_max',
            'raw_max',
            'pooled_topk_mean',
            'raw_topk_mean',
        }
        if image_score_mode not in valid_score_modes:
            raise ValueError(
                f'Unsupported UniAD image_score_mode={image_score_mode!r}. '
                f'Expected one of {sorted(valid_score_modes)}.'
            )
        self.image_score_mode = image_score_mode
        self.image_score_topk = int(image_score_topk)
        self.image_score_pool_kernel = int(image_score_pool_kernel)

        # Detect backbone type and feature info
        # TIMMBackbone has 'reduction' attribute, RawBackbone doesn't
        self._is_timm_backbone = hasattr(self.backbone, 'reduction')

        if self._is_timm_backbone:
            # TIMMBackbone: get channels and strides from feature_info
            self._feature_channels = list(self.backbone.out_channels)
            self._feature_strides = list(self.backbone.reduction)
        else:
            # RawBackbone (WRN-50): use channel_dims and fixed strides
            # Old UniAD only uses layer1, layer2, layer3 (first 3 of 4 layers)
            self._feature_channels = list(self.backbone.channel_dims[:3])
            self._feature_strides = [4, 8, 16]  # WRN-50 layer1/2/3 strides

        self.num_layers = len(self._feature_channels)
        self.total_dim = sum(self._feature_channels)

        # MFCN: resize all features to coarsest resolution
        self.mfcn = MFCN(outstride=self._feature_strides[-1])

        # Transformer components
        self.pos_embed = PositionEmbeddingLearned(self.feature_size, hidden_dim // 2)
        self.input_proj = nn.Linear(self.total_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.total_dim)

        self.encoder_layers = nn.ModuleList([
            UniADEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_encoder_layers)])
        self.encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.decoder_layers = nn.ModuleList([
            UniADDecoderLayer(hidden_dim, self.feature_size, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_decoder_layers)])
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.jitter_scale = feature_jitter_scale
        self.jitter_prob = feature_jitter_prob

        # Validate neighbor_mask_layers length (must be 3 elements)
        if len(neighbor_mask_layers) != 3:
            raise ValueError('neighbor_mask_layers must be a 3-element tuple (enc, dec_tgt, dec_mem).')
        self.neighbor_mask_layers = tuple(bool(v) for v in neighbor_mask_layers)

        if use_neighbor_mask:
            mask = _generate_neighbor_mask(self.feature_size, neighbor_size)
            self.register_buffer('neighbor_mask', mask)

        # ADer-style init: only trainable linear/conv layers
        self._init_reconstruction_weights()

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _init_reconstruction_weights(self):
        modules = [
            self.input_proj,
            self.output_proj,
            self.encoder_layers,
            self.decoder_layers,
            self.encoder_norm,
            self.decoder_norm,
            self.pos_embed,
        ]
        for module in modules:
            if module is None:
                continue
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.constant_(submodule.bias, 0.0)

    def extract_feats(self, x):
        """Extract multi-scale features from backbone."""
        with torch.no_grad():
            feats = self.backbone(x)
        # For RawBackbone (WRN-50), only use first 3 layers (layer1, layer2, layer3)
        if not self._is_timm_backbone:
            feats = feats[:3]
        return feats

    def _merge_feats(self, feats):
        """MFCN: resize all to coarsest resolution and concat, then resize to target feature_size."""
        merged = self.mfcn(feats)
        # Resize to target feature_size if different
        if merged.shape[2:] != self.feature_size:
            merged = F.interpolate(merged, size=self.feature_size, mode='bilinear', align_corners=False)
        return merged

    def _add_jitter(self, tokens):
        if random.uniform(0, 1) <= self.jitter_prob:
            L, B, C = tokens.shape
            norms = tokens.norm(dim=2).unsqueeze(2) / C
            noise = torch.randn_like(tokens) * norms * self.jitter_scale
            tokens = tokens + noise
        return tokens

    @staticmethod
    def _topk_mean(flat_scores: torch.Tensor, topk: int) -> torch.Tensor:
        topk = max(1, min(int(topk), int(flat_scores.shape[1])))
        return flat_scores.topk(topk, dim=1).values.mean(dim=1)

    @classmethod
    def _compute_image_scores_from_map(
        cls,
        score_map: torch.Tensor,
        mode: str = 'pooled_max',
        topk: int = 64,
        pool_kernel: int = 16,
    ) -> torch.Tensor:
        if score_map.ndim != 3:
            raise ValueError(f'Expected score_map with shape [B, H, W], got {tuple(score_map.shape)}.')

        if mode == 'raw_max':
            return score_map.view(score_map.shape[0], -1).max(dim=1).values
        if mode == 'raw_topk_mean':
            flat = score_map.view(score_map.shape[0], -1)
            return cls._topk_mean(flat, topk)

        pooled = F.avg_pool2d(
            score_map.unsqueeze(1),
            kernel_size=pool_kernel,
            stride=1,
            padding=0,
        ).squeeze(1)
        flat = pooled.view(pooled.shape[0], -1)
        if mode == 'pooled_max':
            return flat.max(dim=1).values
        if mode == 'pooled_topk_mean':
            return cls._topk_mean(flat, topk)
        raise ValueError(f'Unsupported UniAD image score mode: {mode!r}')

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        # Extract features from backbone
        feats = self.extract_feats(inputs)
        merged = self._merge_feats(feats)
        B, C, H, W = merged.shape

        tokens = merged.view(B, C, H * W).permute(2, 0, 1)  # (HW, B, C)
        tokens = tokens.detach()

        if self.training:
            tokens = self._add_jitter(tokens)

        tokens = self.input_proj(tokens)
        pos = self.pos_embed(tokens.device).unsqueeze(1).expand(-1, B, -1)

        if self.use_neighbor_mask:
            base_mask = self.neighbor_mask
            mask_enc = base_mask if self.neighbor_mask_layers[0] else None
            mask_dec_tgt = base_mask if self.neighbor_mask_layers[1] else None
            mask_dec_mem = base_mask if self.neighbor_mask_layers[2] else None
        else:
            mask_enc = mask_dec_tgt = mask_dec_mem = None

        # Encoder
        enc = tokens
        for layer in self.encoder_layers:
            enc = layer(enc, pos, mask=mask_enc)
        if self.encoder_norm is not None:
            enc = self.encoder_norm(enc)

        # Decoder
        dec = enc
        for layer in self.decoder_layers:
            dec = layer(dec, enc, pos, tgt_mask=mask_dec_tgt, mem_mask=mask_dec_mem)
        dec = self.decoder_norm(dec)

        rec = self.output_proj(dec)
        feature_rec = rec.permute(1, 2, 0).view(B, C, H, W)
        feature_align = merged.detach()

        pred = torch.sqrt((feature_rec - feature_align).pow(2).sum(dim=1, keepdim=True))
        # Upsample to input size
        pred_up = F.interpolate(pred, size=inputs.shape[-2:], mode='bilinear', align_corners=False)

        if mode == 'loss':
            loss = self.loss_fn(feature_rec, feature_align)
            return {'loss': loss}
        elif mode == 'predict':
            score_map = pred_up.squeeze(1)  # B, H, W
            img_scores = self._compute_image_scores_from_map(
                score_map,
                mode=self.image_score_mode,
                topk=self.image_score_topk,
                pool_kernel=self.image_score_pool_kernel,
            )
            return build_predict_results(data_samples, img_scores, score_map)
        return feature_rec

    def train(self, mode=True):
        super().train(mode)
        # Keep backbone in eval mode
        self.backbone.eval()
        return self
