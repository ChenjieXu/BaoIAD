"""MambaAD detector aligned to the official MVTec architecture.

Reference:
    He et al., "MambaAD: Exploring State Space Models for Multi-class
    Unsupervised Anomaly Detection", NeurIPS 2024.
"""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Callable, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:  # timm<0.9 compatibility
    from timm.models.layers import DropPath, trunc_normal_

from timm.models.resnet import Bottleneck

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import BaseADModel
from baoiad.utils.score_utils import minmax_normalize

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_fn
    _HAS_SELECTIVE_SCAN = True
except ImportError:
    _selective_scan_fn = None
    _HAS_SELECTIVE_SCAN = False

_WARNED_NO_SELECTIVE_SCAN = False


def _warn_missing_selective_scan() -> None:
    """Warn once when falling back to the pure PyTorch selective scan."""
    global _WARNED_NO_SELECTIVE_SCAN
    if _WARNED_NO_SELECTIVE_SCAN:
        return
    if _HAS_SELECTIVE_SCAN:
        message = (
            'MambaAD is using the pure PyTorch selective-scan fallback because '
            'the current execution path is not using CUDA tensors. This is '
            'acceptable for CPU tests, but official alignment checks should run '
            'the CUDA selective-scan kernels.'
        )
    else:
        message = (
            'mamba_ssm is not installed; MambaAD is using the pure PyTorch '
            'selective-scan fallback. This path is suitable for structure/tests, '
            'but final alignment benchmarks should use the official CUDA kernels.'
        )
    warnings.warn(message, stacklevel=3)
    _WARNED_NO_SELECTIVE_SCAN = True


def _rot(size: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    """Rotate/flip a quadrant for Hilbert indexing."""
    if ry == 0:
        if rx == 1:
            x = size - 1 - x
            y = size - 1 - y
        x, y = y, x
    return x, y


def _hilbert_d2xy(size: int, index: int) -> Tuple[int, int]:
    """Convert a Hilbert distance to x/y coordinates."""
    x = 0
    y = 0
    t = index
    step = 1
    while step < size:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(step, x, y, rx, ry)
        x += step * rx
        y += step * ry
        t //= 4
        step *= 2
    return x, y


def _build_scan_order(size: int, scan_type: str) -> Tuple[int, ...]:
    """Build flatten-order indices for a square feature map."""
    if size <= 0:
        raise ValueError(f'Invalid scan size: {size}')

    if scan_type == 'sweep':
        return tuple(range(size * size))

    if scan_type == 'scan':
        order = []
        for row in range(size):
            cols = range(size) if row % 2 == 0 else range(size - 1, -1, -1)
            for col in cols:
                order.append(row * size + col)
        return tuple(order)

    if scan_type == 'hilbert':
        if size & (size - 1):
            raise ValueError(f'Hilbert scan requires a power-of-two size, got {size}.')
        order = []
        for distance in range(size * size):
            x, y = _hilbert_d2xy(size, distance)
            order.append(y * size + x)
        return tuple(order)

    raise ValueError(f'Unsupported scan_type: {scan_type!r}')


class HSCANS(nn.Module):
    """Hybrid scanning order helper used by the official decoder."""

    def __init__(self, size: int, scan_type: str = 'hilbert'):
        super().__init__()
        order = torch.tensor(_build_scan_order(size, scan_type), dtype=torch.long)
        inverse = torch.argsort(order)
        self.register_buffer('index_flat', order.view(1, 1, -1), persistent=False)
        self.register_buffer('index_flat_inv', inverse.view(1, 1, -1), persistent=False)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Reorder a flattened feature map into scan order."""
        return img.gather(2, self.index_flat.expand_as(img))

    def decode(self, img: torch.Tensor) -> torch.Tensor:
        """Restore a scan-ordered sequence to row-major order."""
        return img.gather(2, self.index_flat_inv.expand_as(img))


def _minmax_normalize_per_sample(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each sample independently to [0, 1]."""
    return minmax_normalize(tensor, eps=eps, dim=0)


def _gaussian_blur_bchw(tensor: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    """Apply torchvision Gaussian blur to a BCHW tensor."""
    from torchvision.transforms.functional import gaussian_blur

    return gaussian_blur(
        tensor,
        kernel_size=[kernel_size, kernel_size],
        sigma=[float(sigma), float(sigma)],
    )


def _selective_scan_fallback(
    xs: torch.Tensor,
    dts: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    Cs: torch.Tensor,
    Ds: torch.Tensor,
    *,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    return_last_state: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback for the Mamba selective scan kernel.

    This mirrors the subset of the official ``selective_scan_fn`` signature
    used by MambaAD.
    """

    xs = xs.float()
    dts = dts.float()
    As = As.float()
    Bs = Bs.float()
    Cs = Cs.float()
    Ds = Ds.float()

    if delta_bias is not None:
        dts = dts + delta_bias.view(1, -1, 1)
    if delta_softplus:
        dts = F.softplus(dts)

    batch_size, kd_channels, seq_len = xs.shape
    num_directions = int(Bs.shape[1])
    state_dim = int(Bs.shape[2])
    channels = kd_channels // num_directions

    xs = xs.view(batch_size, num_directions, channels, seq_len)
    dts = dts.view(batch_size, num_directions, channels, seq_len)
    As = As.view(num_directions, channels, state_dim).unsqueeze(0)
    Ds = Ds.view(num_directions, channels).unsqueeze(0)

    state = xs.new_zeros(batch_size, num_directions, channels, state_dim)
    outputs = []
    for step in range(seq_len):
        x_t = xs[..., step]
        dt_t = dts[..., step]
        b_t = Bs[..., step]
        c_t = Cs[..., step]

        d_a = torch.exp(dt_t.unsqueeze(-1) * As)
        d_b = dt_t.unsqueeze(-1) * b_t.unsqueeze(2) * x_t.unsqueeze(-1)
        state = d_a * state + d_b
        y_t = (state * c_t.unsqueeze(2)).sum(dim=-1) + Ds * x_t
        outputs.append(y_t.reshape(batch_size, kd_channels))

    output = torch.stack(outputs, dim=-1)
    if return_last_state:
        return output, state.reshape(batch_size, kd_channels, state_dim)
    return output


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution used by the official feature fusion path."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution used by the official feature fusion path."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PatchExpand2D(nn.Module):
    """Official 2x patch expansion block."""

    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        x = self.expand(x)
        x = x.view(
            batch_size,
            height,
            width,
            self.dim_scale,
            self.dim_scale,
            channels // self.dim_scale,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size,
            height * self.dim_scale,
            width * self.dim_scale,
            channels // self.dim_scale,
        )
        return self.norm(x)


class SS2D(nn.Module):
    """2-D selective scan block from the official decoder."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: int | str = 'auto',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = 'random',
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        size: int = 8,
        scan_type: str = 'hilbert',
        num_direction: int = 8,
        official_ssm_required: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else int(dt_rank)
        self.scan_type = scan_type
        self.configured_size = int(size)
        self.num_direction = int(num_direction)
        self.official_ssm_required = bool(official_ssm_required)
        self._scan_cache: Dict[Tuple[int, str, int | None], HSCANS] = {}

        if self.official_ssm_required and not _HAS_SELECTIVE_SCAN:
            raise ImportError(
                'official_ssm_required=True but mamba_ssm selective_scan kernels '
                'are not available.'
            )

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            padding=(self.d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        x_proj_weight = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False).weight
            for _ in range(self.num_direction)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weight, dim=0))
        dt_projs = [
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
            )
            for _ in range(self.num_direction)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.num_direction, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.num_direction, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = 'random',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> nn.Linear:
        """Initialize the delta projection as in the official implementation."""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f'Unsupported dt_init: {dt_init}')

        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state: int, d_inner: int, copies: int = 1, merge: bool = True) -> nn.Parameter:
        """Initialize the diagonal state matrix in log space."""
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, d_state).repeat(d_inner, 1)
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        param = nn.Parameter(A_log)
        param._no_weight_decay = True
        return param

    @staticmethod
    def D_init(d_inner: int, copies: int = 1, merge: bool = True) -> nn.Parameter:
        """Initialize the skip parameter."""
        D = torch.ones(d_inner)
        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        param = nn.Parameter(D)
        param._no_weight_decay = True
        return param

    def _get_scans(self, size: int, device: torch.device) -> HSCANS:
        """Cache scan helpers per spatial size/device."""
        key = (size, device.type, device.index)
        scans = self._scan_cache.get(key)
        if scans is None:
            scans = HSCANS(size=size, scan_type=self.scan_type).to(device)
            self._scan_cache[key] = scans
        return scans

    def forward_core(self, x: torch.Tensor) -> torch.Tensor:
        """Core directional selective scan."""
        batch_size, _, height, width = x.shape
        if height != width:
            raise ValueError(f'MambaAD expects square feature maps, got {(height, width)}.')

        seq_len = height * width
        num_direction = self.num_direction
        scans = self._get_scans(height, x.device)

        xs = []
        if num_direction >= 2:
            xs.append(scans.encode(x.view(batch_size, -1, seq_len)))
        if num_direction >= 4:
            xs.append(scans.encode(torch.transpose(x, dim0=2, dim1=3).contiguous().view(batch_size, -1, seq_len)))
        if num_direction >= 8:
            rot_x = torch.rot90(x, k=1, dims=(2, 3)).contiguous()
            xs.append(scans.encode(rot_x.view(batch_size, -1, seq_len)))
            xs.append(scans.encode(torch.transpose(rot_x, dim0=2, dim1=3).contiguous().view(batch_size, -1, seq_len)))

        xs = torch.stack(xs, dim=1).view(batch_size, num_direction // 2, -1, seq_len)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)

        x_dbl = torch.einsum('b k d l, k c d -> b k c l', xs.view(batch_size, num_direction, -1, seq_len), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum('b k r l, k d r -> b k d l', dts.view(batch_size, num_direction, -1, seq_len), self.dt_projs_weight)

        xs = xs.float().view(batch_size, -1, seq_len)
        dts = dts.contiguous().float().view(batch_size, -1, seq_len)
        Bs = Bs.float().view(batch_size, num_direction, -1, seq_len)
        Cs = Cs.float().view(batch_size, num_direction, -1, seq_len)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        if x.is_cuda and _HAS_SELECTIVE_SCAN:
            out_y = _selective_scan_fn(
                xs,
                dts,
                As,
                Bs,
                Cs,
                Ds,
                z=None,
                delta_bias=dt_bias,
                delta_softplus=True,
                return_last_state=False,
            )
        elif not x.is_cuda and self.official_ssm_required:
            raise RuntimeError(
                'official_ssm_required=True but selective_scan requires CUDA inputs. '
                'Run MambaAD on a CUDA device for official alignment checks.'
            )
        else:
            _warn_missing_selective_scan()
            out_y = _selective_scan_fallback(
                xs,
                dts,
                As,
                Bs,
                Cs,
                Ds,
                delta_bias=dt_bias,
                delta_softplus=True,
                return_last_state=False,
            )

        out_y = out_y.view(batch_size, num_direction, -1, seq_len)
        inv_y = torch.flip(out_y[:, num_direction // 2:num_direction], dims=[-1]).view(
            batch_size,
            num_direction // 2,
            -1,
            seq_len,
        )

        ys = []
        if num_direction >= 2:
            ys.append(scans.decode(out_y[:, 0]))
            ys.append(scans.decode(inv_y[:, 0]))
        if num_direction >= 4:
            ys.append(
                torch.transpose(scans.decode(out_y[:, 1]).view(batch_size, -1, width, height), dim0=2, dim1=3)
                .contiguous()
                .view(batch_size, -1, seq_len)
            )
            ys.append(
                torch.transpose(scans.decode(inv_y[:, 1]).view(batch_size, -1, width, height), dim0=2, dim1=3)
                .contiguous()
                .view(batch_size, -1, seq_len)
            )
        if num_direction >= 8:
            ys.append(
                torch.rot90(scans.decode(out_y[:, 2]).view(batch_size, -1, width, height), k=3, dims=(2, 3))
                .contiguous()
                .view(batch_size, -1, seq_len)
            )
            ys.append(
                torch.rot90(scans.decode(inv_y[:, 2]).view(batch_size, -1, width, height), k=3, dims=(2, 3))
                .contiguous()
                .view(batch_size, -1, seq_len)
            )
            ys.append(
                torch.rot90(
                    torch.transpose(scans.decode(out_y[:, 3]).view(batch_size, -1, width, height), dim0=2, dim1=3),
                    k=3,
                    dims=(2, 3),
                )
                .contiguous()
                .view(batch_size, -1, seq_len)
            )
            ys.append(
                torch.rot90(
                    torch.transpose(scans.decode(inv_y[:, 3]).view(batch_size, -1, width, height), dim0=2, dim1=3),
                    k=3,
                    dims=(2, 3),
                )
                .contiguous()
                .view(batch_size, -1, seq_len)
            )

        return sum(ys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the spatial Mamba block on channel-last feature maps."""
        batch_size, height, width, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(batch_size, height, width, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class HSSBlock(nn.Module):
    """Single hybrid state-space block."""

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        size: int = 8,
        scan_type: str = 'hilbert',
        num_direction: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        official_ssm_required: bool = False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            size=size,
            scan_type=scan_type,
            num_direction=num_direction,
            d_conv=d_conv,
            expand=expand,
            official_ssm_required=official_ssm_required,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual selective scan."""
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class LSSModule(nn.Module):
    """Locality-enhanced state-space module from the official decoder."""

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        depth: int = 2,
        size: int = 8,
        scan_type: str = 'hilbert',
        num_direction: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        official_ssm_required: bool = False,
    ):
        super().__init__()
        self.smm_blocks = nn.ModuleList([
            HSSBlock(
                hidden_dim=hidden_dim,
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                size=size,
                scan_type=scan_type,
                num_direction=num_direction,
                d_conv=d_conv,
                expand=expand,
                official_ssm_required=official_ssm_required,
            )
            for _ in range(depth)
        ])
        self.conv1b7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv1a7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv1b5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv1a5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv55 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2, bias=False, groups=hidden_dim),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv77 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3, bias=False, groups=hidden_dim),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.finalconv11 = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, stride=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Official Conv2d initialization."""
        if isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine SSM and local depthwise convolution branches."""
        out_ssm = x
        for block in self.smm_blocks:
            out_ssm = block(out_ssm)

        input_conv = x.permute(0, 3, 1, 2).contiguous()
        out_77 = self.conv1a7(self.conv77(self.conv1b7(input_conv)))
        out_55 = self.conv1a5(self.conv55(self.conv1b5(input_conv)))
        output = torch.cat((out_ssm.permute(0, 3, 1, 2).contiguous(), out_55, out_77), dim=1)
        output = self.finalconv11(output).permute(0, 2, 3, 1).contiguous()
        return output + x


class LSSLayerUp(nn.Module):
    """One decoder stage from the official MambaUPNet."""

    def __init__(
        self,
        dim: int,
        depth: int,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        upsample: type[nn.Module] | None = None,
        use_checkpoint: bool = False,
        d_state: int = 16,
        size: int = 8,
        scan_type: str = 'hilbert',
        num_direction: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        official_ssm_required: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if depth % 3 == 0:
            module_depth = 3
            repeats = depth // 3
        elif depth % 2 == 0:
            module_depth = 2
            repeats = depth // 2
        else:
            module_depth = depth
            repeats = 1

        self.blocks = nn.ModuleList([
            LSSModule(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                size=size,
                scan_type=scan_type,
                depth=module_depth,
                num_direction=num_direction,
                d_conv=d_conv,
                expand=expand,
                official_ssm_required=official_ssm_required,
            )
            for i in range(repeats)
        ])

        def _init_out_proj(module: nn.Module) -> None:
            for name, parameter in module.named_parameters():
                if name == 'out_proj.weight':
                    with torch.no_grad():
                        nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))

        self.apply(_init_out_proj)
        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample then run repeated LSS modules."""
        if self.upsample is not None:
            x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class MambaUPNet(nn.Module):
    """Official multi-stage decoder."""

    def __init__(
        self,
        dims_decoder: Sequence[int],
        depths_decoder: Sequence[int],
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        scan_type: str = 'hilbert',
        num_direction: int = 8,
        official_ssm_required: bool = False,
    ):
        super().__init__()
        if len(dims_decoder) != len(depths_decoder):
            raise ValueError('dims_decoder and depths_decoder must have the same length.')

        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        self.layers_up = nn.ModuleList()
        for i_layer, depth in enumerate(depths_decoder):
            start = sum(depths_decoder[:i_layer])
            end = sum(depths_decoder[:i_layer + 1])
            layer = LSSLayerUp(
                dim=dims_decoder[i_layer],
                depth=depth,
                d_state=d_state,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[start:end],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if i_layer != 0 else None,
                size=8 * (2 ** i_layer),
                scan_type=scan_type,
                num_direction=num_direction,
                d_conv=d_conv,
                expand=expand,
                official_ssm_required=official_ssm_required,
            )
            self.layers_up.append(layer)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Official linear/layernorm initialization."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        """Decode one fused low-resolution feature into multi-scale features."""
        x = x.permute(0, 2, 3, 1).contiguous()
        out_features = []
        for index, layer in enumerate(self.layers_up):
            x = layer(x)
            if index != 0:
                out_features.insert(0, x.permute(0, 3, 1, 2).contiguous())
        return out_features


class MFFOCE(nn.Module):
    """Official multi-scale feature fusion block."""

    def __init__(
        self,
        in_channels_list: Sequence[int],
        out_channels: int,
        block: type[nn.Module] = Bottleneck,
        layers: int = 3,
        width_per_group: int = 64,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        if len(in_channels_list) != 3:
            raise ValueError(f'MFFOCE expects 3 encoder features, got {len(in_channels_list)}.')

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.in_channels_list = tuple(int(c) for c in in_channels_list)
        self.out_channels = int(out_channels)
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.inplanes = self.in_channels_list[-1]
        self.dilation = 1
        self.bn_layer = self._make_layer(block, self.out_channels // block.expansion, layers, stride=2)

        self.conv1 = conv3x3(self.in_channels_list[0], self.in_channels_list[1], stride=2)
        self.bn1 = norm_layer(self.in_channels_list[1])
        self.conv2 = conv3x3(self.in_channels_list[1], self.in_channels_list[2], stride=2)
        self.bn2 = norm_layer(self.in_channels_list[2])
        self.conv21 = nn.Conv2d(self.in_channels_list[1], self.in_channels_list[1], kernel_size=1)
        self.bn21 = norm_layer(self.in_channels_list[1])
        self.conv31 = nn.Conv2d(self.in_channels_list[2], self.in_channels_list[2], kernel_size=1)
        self.bn31 = norm_layer(self.in_channels_list[2])
        self.convf = nn.Conv2d(self.in_channels_list[2], self.in_channels_list[2], kernel_size=1)
        self.bnf = norm_layer(self.in_channels_list[2])
        self.relu = nn.ReLU(inplace=True)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Fuse three teacher features into one low-resolution tensor."""
        if len(features) != 3:
            raise ValueError(f'Expected 3 teacher features, got {len(features)}.')
        fpn0 = self.relu(self.bn1(self.conv1(features[0])))
        fpn1 = self.relu(self.bn21(self.conv21(features[1]))) + fpn0
        sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(features[2])))
        sv_features = self.relu(self.bnf(self.convf(sv_features)))
        return self.bn_layer(sv_features).contiguous()


@MODELS.register_module(force=True)
class MambaADDetector(BaseADModel):
    """MambaAD anomaly detector aligned to the official MVTec setting."""

    def __init__(
        self,
        backbone: dict | None = None,
        depths_decoder: Sequence[int] = (3, 4, 6, 3),
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        drop_path_rate: float = 0.2,
        pixel_loss_weight: float = 5.0,
        scan_type: str = 'hilbert',
        num_direction: int = 8,
        smooth_sigma: float = 4.0,
        smooth_kernel_size: int | None = None,
        official_ssm_required: bool = False,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if backbone is None:
            backbone = dict(
                type='TIMMBackbone',
                model_name='resnet34',
                pretrained=True,
                features_only=True,
                out_indices=(1, 2, 3),
                frozen=True,
            )
        self.encoder = MODELS.build(backbone)

        in_channels_list = tuple(int(c) for c in self.encoder.out_channels)
        if len(in_channels_list) != 3:
            raise ValueError(f'MambaAD requires 3 encoder features, got {in_channels_list}.')

        fused_channels = int(in_channels_list[-1] * 2)
        dims_decoder = (fused_channels, in_channels_list[-1], in_channels_list[1], in_channels_list[0])
        self.mff_oce = MFFOCE(in_channels_list=in_channels_list, out_channels=fused_channels)
        self.decoder = MambaUPNet(
            dims_decoder=dims_decoder,
            depths_decoder=tuple(int(d) for d in depths_decoder),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path_rate=drop_path_rate,
            scan_type=scan_type,
            num_direction=num_direction,
            official_ssm_required=official_ssm_required,
        )

        self.depths_decoder = tuple(int(d) for d in depths_decoder)
        self.scan_type = scan_type
        self.num_direction = int(num_direction)
        self.pixel_loss_weight = float(pixel_loss_weight)
        self.smooth_sigma = float(smooth_sigma)
        self.official_ssm_required = bool(official_ssm_required)
        self.uses_official_ssm = bool(_HAS_SELECTIVE_SCAN)

        if smooth_kernel_size is None:
            self.smooth_kernel_size = 2 * int(4.0 * self.smooth_sigma + 0.5) + 1
        else:
            self.smooth_kernel_size = int(smooth_kernel_size)

    def _extract_teacher_student(self, inputs: torch.Tensor) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        """Run the frozen teacher and official decoder."""
        with torch.no_grad():
            feats_t = self.encoder(inputs)
        feats_t = [feature.detach() for feature in feats_t]
        fused = self.mff_oce(feats_t)
        feats_s = self.decoder(fused)
        return feats_t, feats_s

    def _compute_loss(self, feats_t: Sequence[torch.Tensor], feats_s: Sequence[torch.Tensor]) -> torch.Tensor:
        """Official pixel-space L2 loss across all scales."""
        loss = torch.zeros((), device=feats_s[0].device)
        for feature_t, feature_s in zip(feats_t, feats_s):
            loss = loss + F.mse_loss(feature_s, feature_t)
        return loss * self.pixel_loss_weight

    def _compute_anomaly_map(
        self,
        feats_t: Sequence[torch.Tensor],
        feats_s: Sequence[torch.Tensor],
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """ADer-style add + normalize anomaly map."""
        layer_maps = []
        for feature_t, feature_s in zip(feats_t, feats_s):
            feature_t = F.normalize(feature_t, dim=1)
            feature_s = F.normalize(feature_s, dim=1)
            score = 1 - (feature_t * feature_s).sum(dim=1, keepdim=True)
            score = F.interpolate(score, size=image_size, mode='bilinear', align_corners=False)
            score = _minmax_normalize_per_sample(score)
            layer_maps.append(score)

        score_map = torch.stack(layer_maps, dim=0).sum(dim=0)
        score_map = _minmax_normalize_per_sample(score_map)
        if self.smooth_sigma > 0:
            score_map = _gaussian_blur_bchw(score_map, self.smooth_sigma, self.smooth_kernel_size)
        return score_map.squeeze(1)

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        """Forward interface used by MMEngine."""
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats_t, feats_s = self._extract_teacher_student(inputs)

        if mode == 'loss':
            return {'loss': self._compute_loss(feats_t, feats_s)}

        if mode == 'predict':
            score_map = self._compute_anomaly_map(feats_t, feats_s, tuple(inputs.shape[-2:]))
            img_scores = score_map.flatten(1).max(dim=1).values
            return build_predict_results(data_samples, img_scores, score_map)

        return list(feats_s)

    def train(self, mode: bool = True):
        """Keep the teacher frozen in eval mode during training."""
        super().train(mode)
        self.encoder.eval()
        return self
