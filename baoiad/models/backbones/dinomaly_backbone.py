"""Dinomaly DINOv2 encoder backbone.

Registered as 'DinomalyEncoder' in MODELS registry.
Includes the full DinoVisionTransformer architecture and weight loading.
"""
import math
from functools import partial

import torch
import torch.nn as nn
from mmengine.model import BaseModule
import torch.nn.functional as F

try:
    from timm.layers.drop import DropPath
    from timm.layers.patch_embed import PatchEmbed
except ImportError:  # timm<0.9 compatibility
    from timm.models.layers import DropPath, PatchEmbed

from timm.models.vision_transformer import Attention, LayerScale
from torch.nn.init import trunc_normal_

from baoiad.checkpoint import load_checkpoint as load_baoiad_checkpoint
from baoiad.registry import MODELS

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"

DINOV2_ARCHITECTURES = {
    "small": {"embed_dim": 384, "num_heads": 6, "depth": 12,
              "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "base": {"embed_dim": 768, "num_heads": 12, "depth": 12,
             "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "large": {"embed_dim": 1024, "num_heads": 16, "depth": 24,
              "target_layers": [4, 6, 8, 10, 12, 14, 16, 18]},
}


# ========================== Layer components ===============================

class MemEffAttention(Attention):
    """Memory-efficient attention using PyTorch scaled_dot_product_attention."""

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_bias,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class DinomalyMLP(nn.Module):
    """MLP with optional input dropout."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, bias=False, apply_input_dropout=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.apply_input_dropout = apply_input_dropout

    def forward(self, x):
        if self.apply_input_dropout:
            x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class EncoderBlock(nn.Module):
    """DINOv2 encoder transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, proj_bias=True,
                 ffn_bias=True, drop=0.0, attn_drop=0.0, init_values=None,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_class=MemEffAttention, ffn_layer=DinomalyMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop, bias=ffn_bias,
                             apply_input_dropout=False)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, return_attention=False):
        if isinstance(self.attn, MemEffAttention):
            y = self.attn(self.norm1(x))
            attn = None
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + self.ls1(y)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class DinoVisionTransformer(nn.Module):
    """DINOv2 Vision Transformer (encoder only)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 ffn_bias=True, proj_bias=True, drop_path_rate=0.0,
                 drop_path_uniform=False, init_values=None,
                 embed_layer=PatchEmbed, act_layer=nn.GELU,
                 block_chunks=0, num_register_tokens=0,
                 interpolate_antialias=False, interpolate_offset=0.1):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim,
                                       strict_img_size=False)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens else None
        )
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        dpr = ([drop_path_rate] * depth if drop_path_uniform
               else [x.item() for x in torch.linspace(0, drop_path_rate, depth)])

        block_fn = partial(EncoderBlock, attn_class=MemEffAttention)
        blocks_list = [
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                     drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                     ffn_layer=DinomalyMLP, init_values=init_values)
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        n_pos = self.pos_embed.shape[1] - 1
        if npatch == n_pos and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0_float = float(w0) + self.interpolate_offset
        h0_float = float(h0) + self.interpolate_offset
        sqrt_n = math.sqrt(n_pos)
        sx, sy = w0_float / sqrt_n, h0_float / sqrt_n
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_n), int(sqrt_n), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy), mode="bicubic", antialias=self.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x, masks=None):
        _B, _C, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1],
                           self.register_tokens.expand(x.shape[0], -1, -1),
                           x[:, 1:]), dim=1)
        return x


# ========================== Weight loading =================================

def _parse_encoder_name(name):
    """Parse encoder name like 'dinov2reg_vit_base_14' into components."""
    parts = name.split("_")
    is_reg = "reg" in name
    arch = parts[-2]  # small / base / large
    patch_size = int(parts[-1])
    return ("dinov2_reg" if is_reg else "dinov2"), arch, patch_size


def _build_dinov2_vit(arch, patch_size, is_reg):
    cfg = DINOV2_ARCHITECTURES[arch]
    kwargs = dict(
        patch_size=patch_size, img_size=518, embed_dim=cfg["embed_dim"],
        depth=cfg["depth"], num_heads=cfg["num_heads"], mlp_ratio=4,
        block_chunks=0, init_values=1e-8,
        interpolate_antialias=False, interpolate_offset=0.1,
    )
    if is_reg:
        kwargs["num_register_tokens"] = 4
    return DinoVisionTransformer(**kwargs)


def _download_and_load_weights(encoder, model_type, arch, patch_size, cache_dir="./pre_trained/"):
    import os
    from pathlib import Path
    os.makedirs(cache_dir, exist_ok=True)
    arch_code = arch[0]
    if model_type == "dinov2_reg":
        fname = f"dinov2_vit{arch_code}{patch_size}_reg4_pretrain.pth"
    else:
        fname = f"dinov2_vit{arch_code}{patch_size}_pretrain.pth"
    weight_path = Path(cache_dir) / fname
    if not weight_path.exists():
        model_dir = f"dinov2_vit{arch_code}{patch_size}"
        url = f"{DINOV2_BASE_URL}/{model_dir}/{fname}"
        from baoiad.runtime import require_network

        require_network('download DINOv2 weights', url=url)
        print(f"Downloading DINOv2 weights from {url}")
        torch.hub.download_url_to_file(url, str(weight_path))
    state_dict = load_baoiad_checkpoint(weight_path, map_location="cpu")
    encoder.load_state_dict(state_dict, strict=False)


# ========================== Decoder components (shared with detector) ======

class LinearAttention(nn.Module):
    """Softmax-free linear attention for decoder (ELU-based feature maps)."""

    def __init__(self, input_dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.matmul(k.transpose(-2, -1), v)
        k_sum = k.sum(dim=-2, keepdim=True)
        z = 1.0 / torch.sum(q * k_sum, dim=-1, keepdim=True)
        x = torch.matmul(q, kv) * z
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class DecoderViTBlock(nn.Module):
    """Decoder ViT block with linear attention."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn=LinearAttention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                         qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DinomalyMLP(in_features=dim, hidden_features=mlp_hidden_dim,
                               out_features=dim, act_layer=act_layer, drop=drop,
                               apply_input_dropout=False, bias=False)

    def forward(self, x, attn_mask=None):
        if attn_mask is not None:
            y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ========================== DinomalyEncoder ================================

@MODELS.register_module(force=True)
class DinomalyEncoder(BaseModule):
    """DINOv2 encoder for Dinomaly, registered in MODELS registry.

    Wraps DinoVisionTransformer with pretrained weight loading.

    Args:
        encoder_name (str): DINOv2 variant name, e.g. 'dinov2reg_vit_base_14'.
        frozen (bool): Freeze all parameters. Default True.
    """

    def __init__(self, encoder_name='dinov2reg_vit_base_14', frozen=True, init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        model_type, arch, patch_size = _parse_encoder_name(encoder_name)
        is_reg = model_type == "dinov2_reg"
        self.encoder = _build_dinov2_vit(arch, patch_size, is_reg)
        _download_and_load_weights(self.encoder, model_type, arch, patch_size)

        # Expose key attributes
        arch_config = DINOV2_ARCHITECTURES[arch]
        self.embed_dim = arch_config["embed_dim"]
        self.num_heads = arch_config["num_heads"]
        self.target_layers = list(arch_config["target_layers"])
        self.patch_size = self.encoder.patch_size
        self.num_register_tokens = self.encoder.num_register_tokens

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    @property
    def blocks(self):
        return self.encoder.blocks

    @property
    def norm(self):
        return self.encoder.norm

    def prepare_tokens(self, x, masks=None):
        return self.encoder.prepare_tokens(x, masks)

    def forward(self, x):
        return self.encoder.prepare_tokens(x)

    def train(self, mode=True):
        if mode and not any(p.requires_grad for p in self.parameters()):
            return super().train(False)
        return super().train(mode)
