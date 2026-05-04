"""RealNet detector aligned to the official MVTec configuration."""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import BaseADModel

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block used by the simple reconstruction path."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1, stride=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualStack(nn.Module):
    """Stack of residual blocks."""

    def __init__(self, in_channels: int, num_residual_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualBlock(in_channels) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)


class SimpleUNet(nn.Module):
    """Simple UNet from the official SimpleReconstructionLayer path."""

    def __init__(self, in_channels: int, num_residual_layers: int = 2):
        super().__init__()
        norm_layer = nn.InstanceNorm2d

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.AvgPool2d(2)

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1),
            norm_layer(in_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.AvgPool2d(2)

        self.residual_stack = ResidualStack(in_channels * 4, num_residual_layers)
        self.upblock1 = nn.ConvTranspose2d(
            in_channels * 8,
            in_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.upblock2 = nn.ConvTranspose2d(
            in_channels * 4,
            in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.block1(inputs)
        b1 = self.mp1(x)
        x = self.block2(b1)
        b2 = self.mp2(x)
        x = self.residual_stack(b2)
        x = self.upblock1(torch.cat([x, b2], dim=1))
        x = F.relu(x)
        x = self.upblock2(torch.cat([x, b1], dim=1))
        return x


class RRSResidualBlock(nn.Module):
    """Residual block used by the official RRS decoder."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, stride=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class RRSResidualStack(nn.Module):
    """Residual stack used by the official RRS decoder."""

    def __init__(self, in_channels: int, num_residual_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [RRSResidualBlock(in_channels) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 math, adapted for small unit-test channel counts."""

    def __init__(self, channels: int):
        groups = min(32, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        super().__init__(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> nn.Module:
    return GroupNorm32(channels)


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class Upsample(nn.Module):
    """Transpose-convolution upsample used by the official reconstruction path."""

    def __init__(self, channels: int, use_conv: bool, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=self.out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            return self.conv(x)
        return F.interpolate(x, scale_factor=2, mode='bilinear')


class Downsample(nn.Module):
    """Downsample block for the official reconstruction path."""

    def __init__(self, channels: int, use_conv: bool, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        if use_conv:
            self.op = nn.Conv2d(channels, self.out_channels, 3, stride=2, padding=1)
        else:
            assert channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class QKVAttentionLegacy(nn.Module):
    """QKV attention matching the official diffusion UNet helper."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        channels = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(batch_size * self.n_heads, channels * 3, length).split(channels, dim=1)
        scale = 1 / math.sqrt(math.sqrt(channels))
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        attended = torch.einsum('bts,bcs->bct', weight, v)
        return attended.reshape(batch_size, -1, length)


class AttentionBlock(nn.Module):
    """Spatial attention block used by the official reconstruction UNet."""

    def __init__(self, channels: int, num_heads: int = 1, num_head_channels: int = -1):
        super().__init__()
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            if channels % num_head_channels == 0:
                self.num_heads = channels // num_head_channels
            else:
                self.num_heads = max(1, min(num_heads, channels))
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, *spatial = x.shape
        residual = x.reshape(batch_size, channels, -1)
        qkv = self.qkv(self.norm(residual))
        out = self.attention(qkv)
        out = self.proj_out(out)
        return (residual + out).reshape(batch_size, channels, *spatial)


class ReconstructionResBlock(nn.Module):
    """Residual block used by the official reconstruction UNet."""

    def __init__(self, channels: int, out_channels: Optional[int] = None, use_conv: bool = False, up: bool = False, down: bool = False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, True)
            self.x_upd = Upsample(channels, True)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest = self.in_layers[:-1]
            in_conv = self.in_layers[-1]
            hidden = in_rest(x)
            hidden = self.h_upd(hidden)
            x = self.x_upd(x)
            hidden = in_conv(hidden)
        else:
            hidden = self.in_layers(x)
        return self.skip_connection(x) + hidden


class ReconstructionUNet(nn.Module):
    """Official ReconstructionLayer UNet for one feature block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        num_res_blocks: int,
        channel_mult: Sequence[int],
        attention_mult: Sequence[int],
        num_heads: int = 4,
        num_heads_upsample: int = -1,
        num_head_channels: int = 64,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        channels = input_channels = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([nn.Conv2d(in_channels, channels, 3, padding=1)])
        input_block_channels = [channels]
        downsample_scale = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: List[nn.Module] = [
                    ReconstructionResBlock(
                        channels,
                        out_channels=int(mult * model_channels),
                    )
                ]
                channels = int(mult * model_channels)
                if downsample_scale in attention_mult:
                    layers.append(
                        AttentionBlock(
                            channels,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(channels)

            if level != len(channel_mult) - 1:
                out_ch = channels
                self.input_blocks.append(
                    ReconstructionResBlock(channels, out_channels=out_ch, down=True)
                )
                channels = out_ch
                input_block_channels.append(channels)
                downsample_scale *= 2

        self.middle_block = nn.Sequential(
            ReconstructionResBlock(channels),
            AttentionBlock(
                channels,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ReconstructionResBlock(channels),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_idx in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [
                    ReconstructionResBlock(
                        channels + skip_channels,
                        out_channels=int(model_channels * mult),
                    )
                ]
                channels = int(model_channels * mult)
                if downsample_scale in attention_mult:
                    layers.append(
                        AttentionBlock(
                            channels,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                        )
                    )
                if level and block_idx == num_res_blocks:
                    out_ch = channels
                    layers.append(
                        ReconstructionResBlock(channels, out_channels=out_ch, up=True)
                    )
                    downsample_scale //= 2
                self.output_blocks.append(nn.Sequential(*layers))

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = []
        hidden = x
        for module in self.input_blocks:
            hidden = module(hidden)
            hidden_states.append(hidden)
        hidden = self.middle_block(hidden)
        for module in self.output_blocks:
            hidden = torch.cat([hidden, hidden_states.pop()], dim=1)
            hidden = module(hidden)
        return self.out(hidden)


class RRSDecoder(nn.Module):
    """Reconstruction Refinement Strategy aligned to the official code."""

    def __init__(
        self,
        inplanes: Dict[str, int],
        instrides: Dict[str, int],
        modes: Sequence[str],
        mode_numbers: Sequence[int],
        num_residual_layers: int = 2,
        stop_grad: bool = False,
    ):
        super().__init__()
        self.inplanes = dict(inplanes)
        self.instrides = dict(instrides)
        self.modes = list(modes)
        self.mode_numbers = list(mode_numbers)
        self.stop_grad = stop_grad
        self.total_select_number = sum(self.mode_numbers)

        align_stride = min(self.instrides.values())
        self.upsample_modules = nn.ModuleDict()
        for block_name, stride in self.instrides.items():
            scale_factor = stride / align_stride
            if scale_factor > 1:
                self.upsample_modules[block_name] = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            else:
                self.upsample_modules[block_name] = nn.Identity()

        align_inplanes = sum(self.inplanes.values())
        self.bn_idx = nn.BatchNorm2d(align_inplanes, momentum=0.9, affine=False)

        self.decoder1 = nn.Sequential(
            RRSResidualStack(self.total_select_number, num_residual_layers),
            nn.Conv2d(self.total_select_number, 128, 3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 2, 3, padding=1, bias=True),
        )

    @torch.no_grad()
    def select_anomaly_index(self, residual: torch.Tensor, mode: str, k: int) -> torch.Tensor:
        batch_size, channels, height, width = residual.shape
        residual = residual.view(batch_size, channels, height * width)
        if mode == 'max':
            residual, _ = torch.max(residual, dim=-1)
        elif mode == 'mean':
            residual = torch.mean(residual, dim=-1)
        else:
            raise ValueError(f"Unsupported RRS mode: {mode}")
        _, indices = torch.topk(residual, dim=1, largest=True, k=k, sorted=True)
        return indices

    def forward(self, residuals: Dict[str, torch.Tensor], image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.stop_grad:
            residuals = {key: value.detach() for key, value in residuals.items()}

        residual = torch.cat(
            [self.upsample_modules[name](residuals[name]) for name in residuals],
            dim=1,
        )
        residual_idx = self.bn_idx(residual)
        batch_size, _, height, width = residual.shape

        selected_residuals = []
        for mode, mode_n in zip(self.modes, self.mode_numbers):
            indices = self.select_anomaly_index(residual_idx, mode, mode_n)
            selected_residuals.append(
                torch.gather(
                    residual,
                    dim=1,
                    index=indices.view(batch_size, mode_n, 1, 1).repeat(1, 1, height, width),
                )
            )

        residual = torch.cat(selected_residuals, dim=1)
        decoded = self.decoder1(residual)
        decoded = self.decoder2(decoded)
        decoded = F.interpolate(
            decoded,
            (decoded.size(-2) * 2, decoded.size(-1) * 2),
            mode='bilinear',
            align_corners=True,
        )
        logit_mask = self.decoder3(decoded)
        logit_mask = F.interpolate(logit_mask, image_size, mode='bilinear', align_corners=True)
        pred = torch.softmax(logit_mask, dim=1)[:, 1:2]
        return logit_mask, pred


@MODELS.register_module(force=True)
class RealNetDetector(BaseADModel):
    """RealNet detector with official AFS / reconstruction / RRS semantics."""

    _WRN50_LAYERS = {
        'layer1': {'idx': 1, 'stride': 4, 'planes': 256},
        'layer2': {'idx': 2, 'stride': 8, 'planes': 512},
        'layer3': {'idx': 3, 'stride': 16, 'planes': 1024},
        'layer4': {'idx': 4, 'stride': 32, 'planes': 2048},
    }
    _OFFICIAL_STRUCTURE = [
        dict(name='block1', layers=[dict(idx='layer1', planes=256)], stride=4),
        dict(name='block2', layers=[dict(idx='layer2', planes=512)], stride=8),
        dict(name='block3', layers=[dict(idx='layer3', planes=512)], stride=16),
        dict(name='block4', layers=[dict(idx='layer4', planes=256)], stride=32),
    ]

    def __init__(
        self,
        backbone: dict,
        afs_topk: int = 3,
        structure: Optional[List[Dict]] = None,
        init_bsn: int = 64,
        reconstruction_type: str = 'official',
        num_res_blocks: int = 2,
        hide_channels_ratio: float = 0.5,
        channel_mult: Sequence[int] = (1, 2, 4),
        attention_mult: Sequence[int] = (2, 4),
        num_residual_layers: int = 2,
        rrs_modes: Optional[Sequence[str]] = None,
        rrs_mode_numbers: Optional[Sequence[int]] = None,
        stop_grad: bool = False,
        image_score_pool_size: Tuple[int, int] = (16, 16),
        anomaly_channel_index: int = 1,
        predict_invert_map: bool = False,
        seg_loss: Optional[dict] = None,
        feat_loss: Optional[dict] = None,
        dtd_path: Optional[str] = None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        del dtd_path, kwargs
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if rrs_modes is None:
            rrs_modes = ['max', 'mean']
        if rrs_mode_numbers is None:
            rrs_mode_numbers = [256, 256]
        if seg_loss is None:
            seg_loss = dict(type='CrossEntropyLoss')
        if feat_loss is None:
            feat_loss = dict(type='MSELoss')

        self.afs_topk = afs_topk
        self.init_bsn = init_bsn
        self.reconstruction_type = reconstruction_type
        self.image_score_pool_size = tuple(int(v) for v in image_score_pool_size)
        self.anomaly_channel_index = int(anomaly_channel_index)
        if self.anomaly_channel_index not in (0, 1):
            raise ValueError('RealNet anomaly_channel_index must be 0 or 1.')
        self.predict_invert_map = bool(predict_invert_map)

        self.backbone = MODELS.build(backbone)
        self.backbone_out_indices = tuple(backbone.get('out_indices', (1, 2, 3, 4)))
        self.structure = structure or self._build_default_structure(self.backbone_out_indices, afs_topk)
        self.layer_names = [block['name'] for block in self.structure]

        self.selected_channels: Dict[str, int] = {}
        self.layer_strides: Dict[str, int] = {}
        self.upsample_modules = nn.ModuleDict()
        for block in self.structure:
            block_name = block['name']
            self.selected_channels[block_name] = sum(layer['planes'] for layer in block['layers'])
            self.layer_strides[block_name] = int(block['stride'])
            for layer in block['layers']:
                layer_name = layer['idx']
                init_indices = torch.arange(layer['planes'], dtype=torch.long)
                self.register_buffer(self._afs_buffer_name(block_name, layer_name), init_indices)

                scale_factor = self._WRN50_LAYERS[layer_name]['stride'] // block['stride']
                key = self._upsample_key(block_name, layer_name)
                if scale_factor > 1:
                    self.upsample_modules[key] = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
                else:
                    self.upsample_modules[key] = nn.Identity()

        self.recon_modules = nn.ModuleDict()
        for name in self.layer_names:
            in_channels = self.selected_channels[name]
            if self.reconstruction_type == 'official':
                self.recon_modules[name] = ReconstructionUNet(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    model_channels=max(1, int(hide_channels_ratio * in_channels)),
                    channel_mult=tuple(int(v) for v in channel_mult),
                    num_res_blocks=num_res_blocks,
                    attention_mult=tuple(int(v) for v in attention_mult),
                )
            elif self.reconstruction_type == 'simple':
                self.recon_modules[name] = SimpleUNet(
                    in_channels=in_channels,
                    num_residual_layers=num_residual_layers,
                )
            else:
                raise ValueError(f'Unsupported reconstruction_type: {self.reconstruction_type}')

        self.rrs = RRSDecoder(
            inplanes=self.selected_channels,
            instrides=self.layer_strides,
            modes=rrs_modes,
            mode_numbers=rrs_mode_numbers,
            num_residual_layers=num_residual_layers,
            stop_grad=stop_grad,
        )
        self.seg_loss = MODELS.build(seg_loss)
        self.feat_loss = MODELS.build(feat_loss)
        self.afs_initialized = False

    def _pred_map_from_logits(self, logit_mask: torch.Tensor) -> torch.Tensor:
        pred_map = torch.softmax(
            logit_mask, dim=1
        )[:, self.anomaly_channel_index:self.anomaly_channel_index + 1]
        if self.predict_invert_map:
            pred_map = 1.0 - pred_map
        return pred_map

    @classmethod
    def _build_default_structure(cls, out_indices: Tuple[int, ...], afs_topk: int) -> List[Dict]:
        structure = []
        for idx in out_indices:
            layer_name = f'layer{idx}'
            layer_info = cls._WRN50_LAYERS[layer_name]
            planes = min(afs_topk * 64, layer_info['planes'])
            structure.append(
                dict(
                    name=layer_name,
                    layers=[dict(idx=layer_name, planes=planes)],
                    stride=layer_info['stride'],
                )
            )
        return structure

    @staticmethod
    def _afs_buffer_name(block_name: str, layer_name: str) -> str:
        return f'afs_{block_name}_{layer_name}_indices'

    @staticmethod
    def _upsample_key(block_name: str, layer_name: str) -> str:
        return f'{block_name}_{layer_name}_upsample'

    @property
    def afs_indices(self) -> Dict[str, torch.Tensor]:
        grouped = {}
        for block in self.structure:
            block_indices = [
                getattr(self, self._afs_buffer_name(block['name'], layer['idx']))
                for layer in block['layers']
            ]
            grouped[block['name']] = torch.cat(block_indices, dim=0)
        return grouped

    def _as_batch_tensor(self, inputs) -> torch.Tensor:
        if isinstance(inputs, (list, tuple)):
            return torch.stack(list(inputs))
        return inputs

    def _extract_backbone_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        return {f'layer{idx}': feat for idx, feat in zip(self.backbone_out_indices, feats)}

    def _apply_afs(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        block_feats = {}
        for block in self.structure:
            block_name = block['name']
            selected_layers = []
            for layer in block['layers']:
                layer_name = layer['idx']
                feat = feats[layer_name]
                indices = getattr(self, self._afs_buffer_name(block_name, layer_name))
                selected = torch.index_select(feat, 1, indices)
                selected = self.upsample_modules[self._upsample_key(block_name, layer_name)](selected)
                selected_layers.append(selected)
            block_feats[block_name] = torch.cat(selected_layers, dim=1)
        return block_feats

    def _reconstruct_features(
        self,
        selected_feats: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        recon_feats = {}
        residuals = {}
        for name in self.layer_names:
            feat = selected_feats[name]
            if min(feat.shape[-2:]) < 4:
                recon = feat
            else:
                recon = self.recon_modules[name](feat)
            recon_feats[name] = recon
            residuals[name] = (feat - recon) ** 2
        return recon_feats, residuals

    def _extract_train_targets(self, data_samples, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_imgs = []
        masks = []
        for sample in data_samples:
            if not hasattr(sample, 'clean_img'):
                raise ValueError('RealNet training requires clean_img in data_samples metainfo.')
            clean_imgs.append(sample.clean_img.to(device))
            if not hasattr(sample, 'gt_mask'):
                raise ValueError('RealNet training requires gt_mask on data_samples.')
            masks.append(sample.gt_mask.to(device))
        clean_batch = torch.stack(clean_imgs)
        mask_batch = torch.stack(masks).unsqueeze(1).float()
        return clean_batch, mask_batch

    @staticmethod
    def _normalize_anomaly_types(anomaly_types: Dict[str, float]) -> Dict[str, float]:
        total = sum(float(value) for value in anomaly_types.values())
        if total <= 0:
            raise ValueError('RealNet AFS init requires positive anomaly sampling weights.')
        return {key: float(value) / total for key, value in anomaly_types.items()}

    @torch.no_grad()
    def init_afs(self, train_dataloader) -> None:
        if self.afs_initialized:
            return

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        dataset = getattr(train_dataloader, 'dataset', None)
        original_anomaly_types = None

        if dataset is not None and hasattr(dataset, 'anomaly_types'):
            original_anomaly_types = copy.deepcopy(dataset.anomaly_types)
            anomaly_types = dict(dataset.anomaly_types)
            anomaly_types.pop('normal', None)
            if anomaly_types:
                dataset.anomaly_types = self._normalize_anomaly_types(anomaly_types)
            else:
                logger.warning('RealNet AFS init found no non-normal anomaly types; keeping original sampling.')

        distributed = dist.is_available() and dist.is_initialized()
        iterator = iter(train_dataloader)
        criterion = nn.MSELoss(reduction='none').to(device)

        try:
            for block in self.structure:
                loss_sums = [
                    torch.zeros(self._WRN50_LAYERS[layer['idx']]['planes'], device=device)
                    for layer in block['layers']
                ]

                for _ in range(self.init_bsn):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(train_dataloader)
                        batch = next(iterator)

                    inputs = self._as_batch_tensor(batch['inputs']).to(device)
                    data_samples = batch['data_samples']
                    clean_inputs, gt_mask = self._extract_train_targets(data_samples, device)

                    anomaly_feats = self._extract_backbone_features(inputs)
                    clean_feats = self._extract_backbone_features(clean_inputs)

                    for i, layer in enumerate(block['layers']):
                        layer_name = layer['idx']
                        anomaly_feat = self.upsample_modules[self._upsample_key(block['name'], layer_name)](
                            anomaly_feats[layer_name]
                        )
                        clean_feat = self.upsample_modules[self._upsample_key(block['name'], layer_name)](
                            clean_feats[layer_name]
                        )
                        layer_pred = (anomaly_feat - clean_feat) ** 2
                        _, channels, height, width = layer_pred.shape
                        layer_pred = layer_pred.permute(1, 0, 2, 3).contiguous().view(channels, -1)
                        min_v = torch.min(layer_pred, dim=1).values
                        max_v = torch.max(layer_pred, dim=1).values
                        layer_pred = (layer_pred - min_v.unsqueeze(1)) / (max_v.unsqueeze(1) - min_v.unsqueeze(1) + 1e-4)

                        label = F.interpolate(gt_mask, (height, width), mode='nearest')
                        label = label.permute(1, 0, 2, 3).contiguous().view(1, -1).repeat(channels, 1)
                        mse_loss = torch.mean(criterion(layer_pred, label), dim=1)

                        if distributed:
                            gathered = [torch.zeros_like(mse_loss) for _ in range(dist.get_world_size())]
                            dist.all_gather(gathered, mse_loss)
                            mse_loss = torch.stack(gathered, dim=0).mean(dim=0)

                        loss_sums[i] += mse_loss

                for i, layer in enumerate(block['layers']):
                    values = loss_sums[i]
                    if torch.isnan(values).any():
                        valid = values[~torch.isnan(values)]
                        replacement = valid.max() if valid.numel() > 0 else torch.tensor(0.0, device=device)
                        values = values.clone()
                        values[torch.isnan(values)] = replacement
                    selected = torch.topk(values, k=layer['planes'], dim=-1, largest=False).indices
                    selected, _ = torch.sort(selected)
                    if distributed:
                        dist.broadcast(selected, src=0)
                    getattr(self, self._afs_buffer_name(block['name'], layer['idx'])).copy_(selected.long())
        finally:
            if dataset is not None and original_anomaly_types is not None:
                dataset.anomaly_types = original_anomaly_types

        self.afs_initialized = True
        if was_training:
            self.train()

    def _image_scores_from_map(self, pred_map: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(pred_map, self.image_score_pool_size, stride=1)
        return pooled.reshape(pooled.size(0), -1).max(dim=1).values

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        inputs = self._as_batch_tensor(inputs)
        batch_size, _, height, width = inputs.shape

        if mode == 'loss':
            if data_samples is None:
                raise ValueError('RealNet loss mode requires data_samples.')

            clean_inputs, target_masks = self._extract_train_targets(data_samples, inputs.device)
            with torch.no_grad():
                anomaly_feats = self._extract_backbone_features(inputs)
                clean_feats = self._extract_backbone_features(clean_inputs)
                selected_feats = self._apply_afs(anomaly_feats)
                clean_selected_feats = self._apply_afs(clean_feats)

            recon_feats, residuals = self._reconstruct_features(selected_feats)
            logit_mask, _ = self.rrs(residuals, (height, width))

            if logit_mask.shape[-2:] != target_masks.shape[-2:]:
                target_masks = F.interpolate(target_masks, logit_mask.shape[-2:], mode='nearest')
            target_masks = target_masks.squeeze(1).long()
            if self.anomaly_channel_index == 0:
                target_masks = 1 - target_masks

            loss_seg = self.seg_loss(logit_mask, target_masks)
            pred_map = self._pred_map_from_logits(logit_mask)
            loss_feat = 0.0
            for name in self.layer_names:
                loss_feat = loss_feat + self.feat_loss(recon_feats[name], clean_selected_feats[name].detach())
            loss = loss_seg + loss_feat
            return {
                'loss': loss,
                'loss_seg': loss_seg,
                'loss_feat': loss_feat,
                'pred_mean': pred_map.mean().detach(),
            }

        if mode == 'predict':
            with torch.no_grad():
                feats = self._extract_backbone_features(inputs)
                selected_feats = self._apply_afs(feats)
            _, residuals = self._reconstruct_features(selected_feats)
            logit_mask, _ = self.rrs(residuals, (height, width))
            pred_map = self._pred_map_from_logits(logit_mask)
            img_scores = self._image_scores_from_map(pred_map)
            return build_predict_results(data_samples, img_scores, pred_map)

        with torch.no_grad():
            feats = self._extract_backbone_features(inputs)
            selected_feats = self._apply_afs(feats)
        recon_feats, residuals = self._reconstruct_features(selected_feats)
        logit_mask, _ = self.rrs(residuals, (height, width))
        pred_map = self._pred_map_from_logits(logit_mask)
        img_scores = self._image_scores_from_map(pred_map)
        return {
            'selected_feats': selected_feats,
            'recon_feats': recon_feats,
            'residuals': residuals,
            'logit_mask': logit_mask,
            'pred_map': pred_map,
            'img_scores': img_scores,
        }
