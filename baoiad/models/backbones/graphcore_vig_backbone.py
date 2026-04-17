"""Official-style ViG backbone for GraphCore.

Adapted from the GraphCore implementation in M-3LAB/open-iad.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from timm.models.layers import DropPath
from torch import nn

from baoiad.registry import MODELS


def _act_layer(act: str, inplace: bool = False) -> nn.Module:
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    if act == 'gelu':
        return nn.GELU()
    if act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace)
    if act == 'hswish':
        return nn.Hardswish(inplace)
    raise NotImplementedError(f'Unsupported activation layer: {act}')


def _norm_layer(norm: str, num_channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == 'batch':
        return nn.BatchNorm2d(num_channels, affine=True)
    if norm == 'instance':
        return nn.InstanceNorm2d(num_channels, affine=False)
    raise NotImplementedError(f'Unsupported normalization layer: {norm}')


class _BasicConv(nn.Sequential):
    def __init__(
        self,
        channels,
        act: str = 'relu',
        norm: str | None = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        modules = []
        for i in range(1, len(channels)):
            modules.append(nn.Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                modules.append(_norm_layer(norm, channels[i]))
            if act is not None and act.lower() != 'none':
                modules.append(_act_layer(act))
            if drop > 0:
                modules.append(nn.Dropout2d(drop))
        super().__init__(*modules)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def _batched_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = (idx + idx_base).reshape(-1)

    x = x.transpose(2, 1)
    feature = x.reshape(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.reshape(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


def _get_2d_relative_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    def _get_1d_sincos_pos_embed_from_grid(dim: int, pos: np.ndarray) -> np.ndarray:
        assert dim % 2 == 0
        omega = np.arange(dim // 2, dtype=np.float32)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    return 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]


def _pairwise_distance(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(x * x, dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def _part_pairwise_distance(x: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(x_part * x_part, dim=-1, keepdim=True)
        x_inner = -2 * torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(x * x, dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def _xy_pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(x * x, dim=-1, keepdim=True)
        y_square = torch.sum(y * y, dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def _dense_knn_matrix(
    x: torch.Tensor,
    k: int = 16,
    relative_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, _ = x.shape
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = int(np.ceil(n_points / n_part))
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = _part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list.append(nn_idx_part)
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = _pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def _xy_dense_knn_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int = 16,
    relative_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, _ = x.shape
        dist = _xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class _DenseDilated(nn.Module):
    def __init__(self, k: int = 9, dilation: int = 1, stochastic: bool = False, epsilon: float = 0.0) -> None:
        super().__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        if self.stochastic:
            if torch.rand(1, device=edge_index.device) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num, device=edge_index.device)[:self.k]
                return edge_index[:, :, :, randnum]
            return edge_index[:, :, :, ::self.dilation]
        return edge_index[:, :, :, ::self.dilation]


class _DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k: int = 9, dilation: int = 1, stochastic: bool = False, epsilon: float = 0.0) -> None:
        super().__init__()
        self.dilation = dilation
        self.k = k
        self._dilated = _DenseDilated(k, dilation, stochastic, epsilon)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = _xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = _dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        return self._dilated(edge_index)


class _MRConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = 'relu', norm: str | None = None, bias: bool = True) -> None:
        super().__init__()
        self.nn = _BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x_i = _batched_index_select(x, edge_index[1])
        x_j = _batched_index_select(y if y is not None else x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        batch_size, channels, n_points, width = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(batch_size, 2 * channels, n_points, width)
        return self.nn(x)


class _GraphConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv: str = 'mr', act: str = 'relu', norm: str | None = None, bias: bool = True) -> None:
        super().__init__()
        if conv != 'mr':
            raise NotImplementedError(f'Unsupported graph convolution: {conv}')
        self.gconv = _MRConv2d(in_channels, out_channels, act, norm, bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.gconv(x, edge_index, y)


class _DyGraphConv2d(_GraphConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        dilation: int = 1,
        conv: str = 'mr',
        act: str = 'relu',
        norm: str | None = None,
        bias: bool = True,
        stochastic: bool = False,
        epsilon: float = 0.0,
        r: int = 1,
    ) -> None:
        super().__init__(in_channels, out_channels, conv, act, norm, bias)
        self.r = r
        self.dilated_knn_graph = _DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x: torch.Tensor, relative_pos: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(batch_size, channels, -1, 1).contiguous()
        x = x.reshape(batch_size, channels, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super().forward(x, edge_index, y)
        return x.reshape(batch_size, -1, height, width).contiguous()


class _Grapher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 9,
        dilation: int = 1,
        conv: str = 'mr',
        act: str = 'relu',
        norm: str | None = None,
        bias: bool = True,
        stochastic: bool = False,
        epsilon: float = 0.0,
        r: int = 1,
        n: int = 196,
        drop_path: float = 0.0,
        relative_pos: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = _DyGraphConv2d(
            in_channels,
            in_channels * 2,
            kernel_size,
            dilation,
            conv,
            act,
            norm,
            bias,
            stochastic,
            epsilon,
            r,
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(np.float32(_get_2d_relative_pos_embed(in_channels, int(n**0.5))))
            relative_pos_tensor = relative_pos_tensor.unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(relative_pos_tensor, size=(n, n // (r * r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos: torch.Tensor | None, height: int, width: int) -> torch.Tensor | None:
        if relative_pos is None or height * width == self.n:
            return relative_pos
        n_points = height * width
        reduced = n_points // (self.r * self.r)
        return F.interpolate(relative_pos.unsqueeze(0), size=(n_points, reduced), mode='bicubic').squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.fc1(x)
        _, _, height, width = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, height, width)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        return self.drop_path(x) + shortcut


class _FFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, act: str = 'relu', drop_path: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = _act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop_path(x) + shortcut


class _Stem(nn.Module):
    def __init__(self, out_dim: int = 768, act: str = 'relu') -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, out_dim // 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 8),
            _act_layer(act),
            nn.Conv2d(out_dim // 8, out_dim // 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 4),
            _act_layer(act),
            nn.Conv2d(out_dim // 4, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            _act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            _act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


_GRAPHCORE_VARIANTS: Dict[str, Dict[str, int]] = {
    'vig_ti_224_gelu': dict(n_blocks=12, n_filters=192),
    'vig_s_224_gelu': dict(n_blocks=16, n_filters=320),
    'vig_b_224_gelu': dict(n_blocks=16, n_filters=640),
}


@MODELS.register_module()
class GraphCoreViGBackbone(BaseModule):
    """ViG backbone used by the official GraphCore implementation."""

    def __init__(
        self,
        model_name: str = 'vig_ti_224_gelu',
        pretrained: bool = True,
        checkpoint_path: str = '',
        frozen: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        conv: str = 'mr',
        act: str = 'gelu',
        norm: str = 'batch',
        bias: bool = True,
        dropout: float = 0.0,
        num_knn: int = 9,
        use_dilation: bool = True,
        epsilon: float = 0.2,
        use_stochastic: bool = False,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if model_name not in _GRAPHCORE_VARIANTS:
            raise ValueError(f'Unsupported GraphCore backbone: {model_name}')

        variant_cfg = _GRAPHCORE_VARIANTS[model_name]
        self.model_name = model_name
        self.default_cfg = {
            'input_size': (3, 224, 224),
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
        }

        self.stem = _Stem(out_dim=variant_cfg['n_filters'], act=act)
        self.n_blocks = variant_cfg['n_blocks']
        self.pos_embed = nn.Parameter(torch.zeros(1, variant_cfg['n_filters'], 14, 14))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.n_blocks)]
        num_knn_schedule = [int(x.item()) for x in torch.linspace(num_knn, 2 * num_knn, self.n_blocks)]
        max_dilation = 196 // max(num_knn_schedule)

        blocks = []
        for i in range(self.n_blocks):
            dilation = min(i // 4 + 1, max_dilation) if use_dilation else 1
            blocks.append(nn.Sequential(
                _Grapher(
                    variant_cfg['n_filters'],
                    num_knn_schedule[i],
                    dilation,
                    conv,
                    act,
                    norm,
                    bias,
                    use_stochastic,
                    epsilon,
                    1,
                    drop_path=dpr[i],
                ),
                _FFN(variant_cfg['n_filters'], variant_cfg['n_filters'] * 4, act=act, drop_path=dpr[i]),
            ))
        self.backbone = nn.Sequential(*blocks)
        self.prediction = nn.Sequential(
            nn.Conv2d(variant_cfg['n_filters'], 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            _act_layer(act),
            nn.Dropout(dropout if dropout > 0 else drop_rate),
            nn.Conv2d(1024, 1000, 1, bias=True),
        )
        self._model_init()
        self._load_pretrained_if_available(pretrained, checkpoint_path)

        if frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def _model_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        if not checkpoint_path:
            checkpoint_path = os.path.join('pretrained', 'graphcore')
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        model_tokens = self.model_name.split('_')
        filename = f'{model_tokens[0]}_{model_tokens[1]}.pth'
        return os.path.join(checkpoint_path, filename)

    def _load_pretrained_if_available(self, pretrained: bool, checkpoint_path: str) -> None:
        if not pretrained:
            return
        resolved = self._resolve_checkpoint_path(checkpoint_path)
        if not os.path.exists(resolved):
            warnings.warn(
                f'GraphCore pretrained checkpoint not found at {resolved}; '
                'continuing without official ViG weights.',
                RuntimeWarning,
            )
            return
        checkpoint = torch.load(resolved, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state_dict, dict) and state_dict:
            first_key = next(iter(state_dict))
            if first_key.startswith('module.'):
                state_dict = {key[7:]: value for key, value in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x) + self.pos_embed
        for i in range(self.n_blocks):
            x = self.backbone[i](x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

    def train(self, mode: bool = True):
        if mode and not any(param.requires_grad for param in self.parameters()):
            return super().train(False)
        return super().train(mode)
