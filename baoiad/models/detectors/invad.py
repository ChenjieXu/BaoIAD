"""InvAD (Inverse Anomaly Detection) detector.

Faithful reimplementation based on ADer's InvAD.
Uses StyleGAN2-inspired decoder with ModulatedConv2d, ConstantInput, FusedLeakyReLU,
and EqualConv2d-based Fuser with Bottleneck structure.
All ops are pure PyTorch (no custom CUDA kernels).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baoiad.models.backbone_utils import ResNetBottleneck as Bottleneck
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import BaseADModel


# ========================= StyleGAN2 Ops (Pure PyTorch) =========================

class FusedLeakyReLU(nn.Module):
    """FusedLeakyReLU: bias + leaky_relu + scale, pure PyTorch replacement."""
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return F.leaky_relu(x + self.bias.view(1, -1, 1, 1), self.negative_slope) * self.scale


def fused_leaky_relu(x, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(x + bias.view(1, -1, *([1] * (x.ndim - 2))), negative_slope) * scale


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope) * math.sqrt(2)


class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        return self.input.repeat(batch, 1, 1, 1)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


def upfirdn2d_native(input, kernel, up=1, down=1, pad=(0, 0)):
    """Pure PyTorch upfirdn2d implementation."""
    batch, channel, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    # Reshape for upsampling
    x = input.view(batch * channel, 1, in_h, in_w)

    if up > 1:
        x = F.interpolate(x, scale_factor=up, mode='nearest')
        # Actually we need to insert zeros, not interpolate
        # Use pixel shuffle approach
        x = input.view(batch * channel, in_h, 1, in_w, 1)
        x = F.pad(x, [0, up - 1, 0, 0, 0, up - 1])
        x = x.view(batch * channel, 1, in_h * up, in_w * up)

    # Pad
    pad_x0, pad_x1 = pad
    x = F.pad(x, [pad_x0, pad_x1, pad_x0, pad_x1])

    # Convolve with kernel
    w = kernel.flip([0, 1]).view(1, 1, kernel_h, kernel_w).to(x)
    x = F.conv2d(x, w)

    # Downsample
    if down > 1:
        x = x[:, :, ::down, ::down]

    return x.view(batch, channel, x.shape[2], x.shape[3])


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, x):
        return upfirdn2d_native(x, self.kernel, pad=self.pad)


class EqualConv2d(nn.Module):
    """Equalized learning rate Conv2d from StyleGAN2."""
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                 lr_mul=1, bias=True, bias_init=0, conv_transpose2d=False, activation=False):
        super().__init__()
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size).div_(lr_mul))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
            self.lr_mul = lr_mul
        else:
            self.bias = None
            self.lr_mul = None
        self.conv_transpose2d = conv_transpose2d
        self.activation = ScaledLeakyReLU(0.2) if activation else None

    def forward(self, x):
        bias = self.bias * self.lr_mul if self.bias is not None else None
        if self.conv_transpose2d:
            batch, in_channel, height, width = x.shape
            x_in = x.view(1, batch * in_channel, height, width)
            weight = self.weight.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x_in, weight * self.scale, bias=bias,
                                     padding=self.padding, stride=2, groups=batch)
            _, _, h, w = out.shape
            out = out.view(batch, self.out_channel, h, w)
        else:
            out = F.conv2d(x, self.weight * self.scale, bias=bias,
                          stride=self.stride, padding=self.padding)
        if self.activation:
            out = self.activation(out)
        return out


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(x, self.weight * self.scale,
                          bias=self.bias * self.lr_mul if self.bias is not None else None)
        return out


class ConvLayer(nn.Sequential):
    """Conv layer with optional upsample/downsample and equalized lr."""
    def __init__(self, in_channel, out_channel, kernel_size, upsample=False,
                 downsample=False, blur_kernel=(1, 3, 3, 1), bias=True, activate=True, lr_mul=1.):
        assert not (upsample and downsample)
        layers = []

        if upsample:
            stride = 2
            self_padding = 0
            layers.append(EqualConv2d(in_channel, out_channel, kernel_size,
                                       padding=self_padding, stride=stride,
                                       bias=bias and not activate, conv_transpose2d=True, lr_mul=lr_mul))
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))
        else:
            if downsample:
                factor = 2
                p = (len(blur_kernel) - factor) + (kernel_size - 1)
                pad0 = (p + 1) // 2
                pad1 = p // 2
                layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
                stride = 2
                self_padding = 0
            else:
                stride = 1
                self_padding = kernel_size // 2
            layers.append(EqualConv2d(in_channel, out_channel, kernel_size,
                                       padding=self_padding, stride=stride,
                                       bias=bias and not activate))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


# ========================= Modulated Conv =========================

class ModulatedConv2d(nn.Module):
    """ModulatedConv2d from InvAD/StyleMapGAN — uses spatial modulation (not per-sample demod)."""
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 normalize_mode, blur_kernel=(1, 3, 3, 1),
                 upsample=False, downsample=False, modulate=True):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.modulate = modulate

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        self.normalize_mode = normalize_mode
        if normalize_mode == 'InstanceNorm2d':
            self.norm = nn.InstanceNorm2d(in_channel, affine=False)
        elif normalize_mode == 'BatchNorm2d':
            self.norm = nn.BatchNorm2d(in_channel, affine=False)

        if modulate:
            self.gamma = EqualConv2d(style_dim, in_channel, kernel_size=3, padding=1, stride=1, bias=True, bias_init=1)
            self.beta = EqualConv2d(style_dim, in_channel, kernel_size=3, padding=1, stride=1, bias=True, bias_init=0)

    def forward(self, input, stylecode=None):
        batch, in_channel, height, width = input.shape
        weight = self.scale * self.weight
        weight = weight.repeat(batch, 1, 1, 1, 1)

        # Normalize
        if self.normalize_mode in ('InstanceNorm2d', 'BatchNorm2d'):
            input = self.norm(input)
        elif self.normalize_mode == 'LayerNorm':
            input = F.layer_norm(input, input.shape[1:])
        elif self.normalize_mode == 'GroupNorm':
            input = F.group_norm(input, 8)
        # else: None — no normalization

        if self.modulate and stylecode is not None:
            gamma = self.gamma(stylecode)
            beta = self.beta(stylecode)
            input = input * gamma + beta

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight_t = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight_t = weight_t.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight_t, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 blur_kernel, normalize_mode, upsample=False, activate=True, modulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim,
                                     upsample=upsample, blur_kernel=blur_kernel,
                                     normalize_mode=normalize_mode, modulate=modulate)
        self.activate = FusedLeakyReLU(out_channel) if activate else None

    def forward(self, input, style):
        out = self.conv(input, style)
        if self.activate is not None:
            out = self.activate(out)
        return out


class StyledResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, blur_kernel,
                 normalize_mode, upsample=True, act_layer='none'):
        super().__init__()
        self.conv1 = StyledConv(in_channel, out_channel, 3, style_dim,
                                 blur_kernel=blur_kernel, upsample=upsample,
                                 normalize_mode=normalize_mode, modulate=not upsample)
        self.conv2 = StyledConv(out_channel, out_channel, 3, style_dim,
                                 blur_kernel=blur_kernel, upsample=False,
                                 normalize_mode=normalize_mode, modulate=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, upsample=upsample, activate=False, bias=False)

    def forward(self, input, stylecodes):
        out = self.conv1(input, stylecodes[0])
        out = self.conv2(out, stylecodes[1])
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


# ========================= ConvNormAct helper =========================

class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, bias=False,
                 norm_layer='bn_2d', act_layer='relu'):
        super().__init__()
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, bias=bias)
        if norm_layer == 'bn_2d':
            # Match ADer's `get_norm('bn_2d')`, which uses eps=1e-6.
            self.norm = nn.BatchNorm2d(dim_out, eps=1e-6)
        else:
            self.norm = nn.Identity()
        if act_layer == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ========================= Decoder =========================

class Decoder(nn.Module):
    """InvAD Decoder with ConstantInput and StyledResBlocks."""
    def __init__(self, in_chas=(256, 512, 1024), style_chas=(256, 256, 256),
                 latent_spatial_size=16, latent_channel_size=64,
                 blur_kernel=(1, 3, 3, 1), normalize_mode='LayerNorm',
                 lr_mul=0.01, small_generator=True, layers=(2, 2, 2)):
        super().__init__()
        map_dim_pre = in_chas[-1]
        self.input = ConstantInput(map_dim_pre, size=latent_spatial_size)
        self.conv1 = nn.Sequential(
            ConvLayer(map_dim_pre, map_dim_pre, 3, upsample=False, bias=True, activate=True, lr_mul=lr_mul),
        )

        self.convs_latent = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.depth = len(in_chas)
        for i in range(self.depth):
            latent_cha = latent_channel_size if small_generator else style_chas[self.depth - i - 1]
            self.convs_latent.append(nn.Sequential(
                ConvLayer(style_chas[self.depth - i - 1], style_chas[self.depth - i - 1], 3,
                         upsample=False, bias=True, activate=True, lr_mul=lr_mul),
                ConvLayer(style_chas[self.depth - i - 1], latent_cha, 3,
                         upsample=False, bias=True, activate=True, lr_mul=lr_mul),
            ))

            map_dim_cur = in_chas[self.depth - 1 - i]
            convs_depth = nn.ModuleList()
            convs_depth.append(StyledResBlock(map_dim_pre, map_dim_cur, latent_cha, blur_kernel,
                                               normalize_mode=normalize_mode,
                                               upsample=False if i == 0 else True))
            for j in range(layers[self.depth - 1 - i] - 1):
                convs_depth.append(StyledResBlock(map_dim_cur, map_dim_cur, latent_cha, blur_kernel,
                                                   normalize_mode=normalize_mode, upsample=False))
            self.convs.append(convs_depth)
            map_dim_pre = map_dim_cur

    def forward(self, style_codes):
        batch = style_codes[-1].shape[0]
        out = self.input(batch)
        out = self.conv1(out)
        outs = []
        for i in range(self.depth):
            style_code = self.convs_latent[i](style_codes[i])
            for j in range(len(self.convs[i])):
                if i > 0 and j == 0:
                    out = self.convs[i][j](out, [None, style_code])
                else:
                    out = self.convs[i][j](out, [style_code, style_code])
            outs.append(out)
        return outs


# ========================= Fuser =========================

class Fuser(nn.Module):
    """Fuser with EqualConv2d-based ConvLayers and Bottleneck structure."""
    def __init__(self, in_chas=(256, 512, 1024), style_chas=(256, 256, 256),
                 in_strides=(4, 2, 1), down_conv=True, bottle_num=1,
                 conv_num=1, conv_type='conv', lr_mul=0.01):
        super().__init__()
        self.stage_num = len(in_chas)
        assert len(in_chas) == len(in_strides)

        if down_conv:
            self.downsamples = nn.ModuleList([
                ConvNormAct(in_cha, in_cha, kernel_size=in_stride, stride=in_stride,
                           bias=True, norm_layer='bn_2d', act_layer='none')
                for in_cha, in_stride in zip(in_chas, in_strides)
            ])
        else:
            self.downsamples = nn.ModuleList([
                nn.AvgPool2d(kernel_size=s, stride=s) for s in in_strides
            ])

        self.conv_cat = ConvNormAct(sum(in_chas), style_chas[-1], kernel_size=1, stride=1,
                                     norm_layer='bn_2d', act_layer='relu')
        if bottle_num < 1:
            self.conv_bottle = nn.Identity()
        else:
            self.conv_bottle = nn.Sequential(*[
                Bottleneck(style_chas[-1], style_chas[-1] // Bottleneck.expansion)
                for _ in range(bottle_num)
            ])

        self.convs = nn.ModuleList()
        dim_cur = None
        for i in range(self.stage_num):
            if i == 0:
                dim_pre, dim_cur, upsample = style_chas[-1], style_chas[-1], False
            else:
                dim_pre, upsample = dim_cur, True
                dim_cur = style_chas[len(in_chas) - i - 1]
            convs = [ConvLayer(dim_pre, dim_cur, 3, upsample=upsample, bias=True, activate=True, lr_mul=lr_mul)]
            for _ in range(conv_num):
                convs.append(ConvLayer(dim_cur, dim_cur, 3, upsample=False, bias=True, activate=True, lr_mul=lr_mul))
            self.convs.append(nn.Sequential(*convs))

    def forward(self, feats):
        feat_list = [self.downsamples[i](feats[i]) for i in range(self.stage_num)]
        feat_cat = torch.cat(feat_list, dim=1)
        feat_cat = self.conv_cat(feat_cat)
        feat_bottle = self.conv_bottle(feat_cat)

        x, xs = feat_bottle, []
        for i in range(self.stage_num):
            x = self.convs[i](x)
            xs.append(x)
        return xs


# ========================= InvAD Detector =========================

@MODELS.register_module()
class InvADDetector(BaseADModel):
    """InvAD: Inverse Anomaly Detection.

    Faithful reimplementation based on ADer's InvAD:
    - WRN50 teacher encoder (frozen)
    - Fuser: EqualConv2d + Bottleneck structure
    - Decoder: StyleGAN2-based with ConstantInput, StyledResBlocks, ModulatedConv2d
    - Training loss: MSE on reconstructed teacher features
    - Inference score: unified cosine distance anomaly map
    """

    def __init__(self, backbone: str = 'wide_resnet50_2', out_cha: int = 64, latent_channel_size: int = 16,
                 lr_mul: float = 0.01, gaussian_sigma: float = 4.0,
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.gaussian_sigma = gaussian_sigma

        # ADer's official strict path uses a timm features_only encoder
        # (timm_wide_resnet50_2, out_indices=[1, 2, 3]). Keep supporting the
        # legacy RawBackbone path for older configs and tests.
        if isinstance(backbone, dict):
            teacher = MODELS.build(backbone)
        else:
            teacher = MODELS.build(dict(type='RawBackbone', backbone_name=backbone))

        self.teacher = teacher
        if hasattr(teacher, 'out_channels') and teacher.out_channels is not None:
            self._teacher_mode = 'features_only'
            out_channels = tuple(int(c) for c in teacher.out_channels)
            in_chas = tuple(out_channels[:3])
        else:
            self._teacher_mode = 'raw'
            self.teacher_conv1 = teacher.conv1
            self.teacher_bn1 = teacher.bn1
            self.teacher_relu = teacher.relu
            self.teacher_maxpool = teacher.maxpool
            self.teacher_layer1 = teacher.layer1
            self.teacher_layer2 = teacher.layer2
            self.teacher_layer3 = teacher.layer3
            ch = teacher.channel_dims
            in_chas = (ch[0], ch[1], ch[2])

        if len(in_chas) != 3:
            raise ValueError(
                f'InvAD expects exactly 3 encoder feature levels, got {in_chas}.'
            )

        style_chas = tuple(min(c, out_cha) for c in in_chas)
        in_strides = tuple(2 ** (len(in_chas) - i - 1) for i in range(len(in_chas)))

        self.fuser = Fuser(
            in_chas=in_chas, style_chas=style_chas, in_strides=in_strides,
            down_conv=True, bottle_num=1, conv_num=1, conv_type='conv', lr_mul=lr_mul)

        # Assuming 256x256 input → latent_spatial_size = 256 / 16 = 16
        latent_spatial_size = 16
        self.decoder = Decoder(
            in_chas=in_chas, style_chas=style_chas,
            latent_spatial_size=latent_spatial_size, latent_channel_size=latent_channel_size,
            blur_kernel=(1, 3, 3, 1), normalize_mode='LayerNorm',
            lr_mul=lr_mul, small_generator=True, layers=(2, 2, 2))

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def extract_teacher_feats(self, x):
        with torch.no_grad():
            if self._teacher_mode == 'features_only':
                feats = self.teacher(x)
                return list(feats[:3])

            x = self.teacher_maxpool(self.teacher_relu(self.teacher_bn1(self.teacher_conv1(x))))
            f1 = self.teacher_layer1(x)
            f2 = self.teacher_layer2(f1)
            f3 = self.teacher_layer3(f2)
            return [f1, f2, f3]

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        feats_t = self.extract_teacher_feats(inputs)
        feats_fusion = self.fuser(feats_t)
        feats_pred = self.decoder(feats_fusion)
        # Decoder outputs coarse→fine, reverse to match teacher order
        feats_pred = feats_pred[::-1]
        feats_pred = feats_pred[:len(feats_t)]

        if mode == 'loss':
            loss = 0.0
            for ft, fp in zip(feats_t, feats_pred):
                loss += F.mse_loss(fp, ft)
            return {'loss': loss}
        elif mode == 'predict':
            # ADer-style unified anomaly map: uni_am=True, use_cos=True
            # Concat all scales after normalizing, then compute unified cosine distance
            target_size = feats_t[0].shape[-2:]  # largest feature map
            ft_list, fp_list = [], []
            for ft, fp in zip(feats_t, feats_pred):
                ft_n = F.normalize(ft, p=2, dim=1)
                fp_n = F.normalize(fp, p=2, dim=1)
                ft_list.append(F.interpolate(ft_n, size=target_size, mode='bilinear', align_corners=True))
                fp_list.append(F.interpolate(fp_n, size=target_size, mode='bilinear', align_corners=True))
            ft_cat = torch.cat(ft_list, dim=1)
            fp_cat = torch.cat(fp_list, dim=1)
            score_map = 1 - F.cosine_similarity(ft_cat, fp_cat, dim=1)  # (B, H, W)
            score_map = F.interpolate(score_map.unsqueeze(1), size=inputs.shape[-2:],
                                      mode='bilinear', align_corners=True).squeeze(1)
            # Gaussian smoothing
            if self.gaussian_sigma > 0:
                from scipy.ndimage import gaussian_filter
                import numpy as np
                sm_np = score_map.detach().cpu().numpy()
                for i in range(sm_np.shape[0]):
                    sm_np[i] = gaussian_filter(sm_np[i], sigma=self.gaussian_sigma)
                score_map = torch.from_numpy(sm_np).to(
                    device=inputs.device, dtype=inputs.dtype)
            # ADer's metric computes mAUROC_sp_max from the final anomaly map
            # directly when pooling_ks=None.
            img_scores = score_map.view(score_map.shape[0], -1).max(dim=1).values
            return build_predict_results(data_samples, img_scores, score_map)
        return feats_t, feats_pred

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self
