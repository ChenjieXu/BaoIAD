"""PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow.

Best-effort strict alignment is currently guided by the CVPR 2023 paper,
supplementary material, and the local ADer proxy implementation.

Key components:
    - Pyramid normalizing flows with coupling blocks and volume normalization
    - Contrastive localization via latent template (Δz)
    - FFT loss on batch differences for training
"""
from abc import abstractmethod
import copy
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg as la
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.utils.data import DataLoader
from torchvision import models
from einops import rearrange

from mmengine.dataset import Compose

from baoiad.checkpoint import load_checkpoint as load_baoiad_checkpoint
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import FlowBasedADModel

logger = logging.getLogger(__name__)
_LEGACY_RESNET18_PATH = Path('pretrained') / 'resnet18-5c106cde.pth'


def kornia_filter2d(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Conv2d function with depthwise convolution (from kornia)."""
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output


def _build_pyramidflow_resnet18() -> nn.Module:
    """Build the frozen ResNet-18 stem with a deterministic legacy-weight fallback.

    ADer pins PyramidFlow to the classic torchvision `resnet18-5c106cde.pth`
    checkpoint. Current torchvision APIs expose the equivalent V1 weights
    through an enum, but keeping an explicit local-path preference avoids
    downloader drift and removes ambiguity from strict alignment runs.
    """
    if _LEGACY_RESNET18_PATH.is_file():
        resnet = models.resnet18(weights=None)
        state_dict = load_baoiad_checkpoint(
            _LEGACY_RESNET18_PATH, map_location='cpu')
        if isinstance(state_dict, dict):
            state_dict = state_dict.get('state_dict', state_dict)
        if isinstance(state_dict, dict) and state_dict:
            first_key = next(iter(state_dict))
            if first_key.startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
        incompatible = resnet.load_state_dict(state_dict, strict=False)
        missing_keys = [k for k in incompatible.missing_keys if not k.endswith('num_batches_tracked')]
        if missing_keys or incompatible.unexpected_keys:
            logger.warning(
                'PyramidFlow legacy ResNet-18 checkpoint load mismatch: missing=%s unexpected=%s',
                missing_keys[:10],
                incompatible.unexpected_keys[:10],
            )
        return resnet

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    from baoiad.runtime import require_torchvision_weights

    require_torchvision_weights(weights, action='load PyramidFlow ResNet-18 pretrained weights')
    return models.resnet18(weights=weights)


class InvertibleModule(nn.Module):
    """Base class for constructing normalizing flows.

    You should implement `forward` and `inverse` functions manually, which define
    a basic invertible module. Each function needs to implement the corresponding
    output tensor for a given input tensor, and also the Jacobian determinant
    of the output tensor relative to the input tensor.
    """

    def __init__(self):
        super(InvertibleModule, self).__init__()

    @abstractmethod
    def forward(self, inputs: Tuple[torch.Tensor, ...],
                logdets: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        raise NotImplementedError

    @abstractmethod
    def inverse(self, outputs: Tuple[torch.Tensor, ...],
                logdets: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        raise NotImplementedError

    def _forward(self, *inputs_logdets):
        """Hidden function for implementing SequentialNF."""
        assert len(inputs_logdets) % 2 == 0
        n = len(inputs_logdets) // 2
        inputs = inputs_logdets[:n]
        logdets = inputs_logdets[n:]
        outputs, logdets = self.forward(inputs, logdets)
        return outputs + logdets

    def _inverse(self, *outputs_logdets):
        """Hidden function for implementing SequentialNF."""
        assert len(outputs_logdets) % 2 == 0
        n = len(outputs_logdets) // 2
        outputs = outputs_logdets[:n]
        logdets = outputs_logdets[n:]
        inputs, logdets = self.inverse(outputs, logdets)
        return inputs + logdets


class AutoNFSequential(torch.autograd.Function):
    """Automatic implementation for sequential normalizing flows.

    Memory-saving backward pass using gradient checkpointing style computation.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, _forward_lst, _inverse_lst, inplogsRange, paramsRanges,
                *inplogs_and_params):
        assert inplogsRange[1] % 2 == 0
        inplogs = inplogs_and_params[inplogsRange[0]: inplogsRange[1]]

        with torch.no_grad():
            outlogs = tuple([inplog.detach() for inplog in inplogs])
            for _forward in _forward_lst:
                outlogs = _forward(*outlogs)
                for outlog in outlogs:
                    assert not outlog.isnan().any()

        ctx._forward_lst = _forward_lst
        ctx._inverse_lst = _inverse_lst
        ctx.outlogsRange = inplogsRange
        ctx.paramsRanges = paramsRanges

        outlogs = tuple([outlog.detach() for outlog in outlogs])
        params = inplogs_and_params[inplogsRange[1]:]
        ctx.save_for_backward(*outlogs, *params)

        return outlogs

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_outlogs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("This function is not compatible with .grad(), please use .backward() if possible")
        outlogs_params = ctx.saved_tensors
        outlogs = outlogs_params[ctx.outlogsRange[0]: ctx.outlogsRange[1]]
        params = [outlogs_params[r[0]: r[1]] for r in ctx.paramsRanges]
        _inverse_lst = ctx._inverse_lst
        _forward_lst = ctx._forward_lst
        grad_outlogs_loop = grad_outlogs

        grad_params = tuple()
        detached_outlogs_loop = tuple([outlog.detach() for outlog in outlogs])
        for _forward, _inverse, param in zip(reversed(_forward_lst), reversed(_inverse_lst), reversed(params)):
            with torch.no_grad():
                inplogs = _inverse(*detached_outlogs_loop)
            with torch.set_grad_enabled(True):
                inplogs_loop = tuple([inplog.detach().requires_grad_() for inplog in inplogs])
                outlogs_loop = _forward(*inplogs_loop)
            grad_inplogs_params = torch.autograd.grad(
                outputs=outlogs_loop, grad_outputs=grad_outlogs_loop,
                inputs=inplogs_loop + param
            )

            detached_outlogs_loop = tuple([inplog.detach() for inplog in inplogs])
            grad_outlogs_loop = grad_inplogs_params[:len(inplogs_loop)]
            grad_params = grad_inplogs_params[len(inplogs_loop):] + grad_params

        grad_inplogs = grad_outlogs_loop
        return (None, None, None, None,) + grad_inplogs + grad_params


class SequentialNF(InvertibleModule):
    """Memory-saving sequential normalizing flows.

    A constructor class to build memory saving normalizing flows by a tuple
    of `InvertibleModule`.
    """

    def __init__(self, modules: Tuple[InvertibleModule, ...]):
        super(SequentialNF, self).__init__()
        self.moduleslst = nn.ModuleList(modules)
        self._forward_lst = tuple([module._forward for module in modules])
        self._inverse_lst = tuple([module._inverse for module in modules])
        self.params = [[p for p in module.parameters() if p.requires_grad]
                       for module in self.moduleslst]

    def forward(self, inputs: Tuple[torch.Tensor, ...],
                logdets: Tuple[torch.Tensor, ...]):
        assert len(inputs) == len(logdets)
        inplogsRange = [0, len(inputs) + len(logdets)]
        paramsRange, lastIdx = [], inplogsRange[-1]
        for param in self.params:
            paramsRange.append([lastIdx, lastIdx + len(param)])
            lastIdx += len(param)

        outlogs = AutoNFSequential.apply(
            self._forward_lst, self._inverse_lst,
            inplogsRange, paramsRange,
            *inputs, *logdets,
            *[p for param in self.params for p in param]
        )
        mid = len(outlogs) // 2
        return outlogs[:mid], outlogs[mid:]

    def inverse(self, outputs: Tuple[torch.Tensor, ...],
                logdets: Tuple[torch.Tensor, ...]):
        assert len(outputs) == len(logdets)
        outlogsRange = [0, len(outputs) + len(logdets)]
        paramsRange, lastIdx = [], outlogsRange[-1]
        for param in self.params:
            paramsRange.append([lastIdx, lastIdx + len(param)])
            lastIdx += len(param)

        inplogs = AutoNFSequential.apply(
            list(reversed(self._inverse_lst)), list(reversed(self._forward_lst)),
            outlogsRange, paramsRange,
            *outputs, *logdets,
            *[p for param in self.params for p in param]
        )
        mid = len(inplogs) // 2
        return inplogs[:mid], inplogs[mid:]


class SequentialNet(nn.Module):
    """A constructor class to build pytorch-based normalizing flows (non-memory-saving)."""

    def __init__(self, modules: Tuple[nn.Module, ...]):
        super(SequentialNet, self).__init__()
        self.moduleslst = nn.ModuleList(modules)

    def forward(self, inputs: Tuple[torch.Tensor, ...],
                logdets: Tuple[torch.Tensor, ...]):
        outputs = tuple(inputs)
        for m in self.moduleslst:
            outputs, logdets = m(outputs, logdets)
        return outputs, logdets


class SemiInvertible1x1Conv(nn.Conv2d):
    """Semi-invertible 1x1 Conv used at the first stage of NF (when no backbone)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        assert out_channels >= in_channels
        super().__init__(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.weight.data)

    def inverse(self, output: torch.Tensor) -> torch.Tensor:
        b, c, h, w = output.shape
        A = self.weight[..., 0, 0]  # (outch, inch)
        B = output.permute([1, 0, 2, 3]).reshape(c, -1)  # (outch, bhw)
        X = torch.linalg.lstsq(A, B)
        return X.solution.reshape(-1, b, h, w).permute([1, 0, 2, 3])

    @property
    def logdet(self) -> torch.Tensor:
        w = self.weight.squeeze()  # (out, in)
        return 0.5 * torch.logdet(w.T @ w)


class LaplacianMaxPyramid(nn.Module):
    """Laplacian Pyramid with Max Pooling for multi-scale decomposition.

    Uses Gaussian blur + max pooling for downsampling, and nearest interpolation
    + Gaussian blur for upsampling.
    """

    def __init__(self, num_levels: int, downsample_mode: str = 'nearest') -> None:
        super().__init__()
        # Binomial kernel [1,4,6,4,1]^T @ [1,4,6,4,1] / 256
        self.kernel = torch.tensor(
            [
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ]
        ) / 256.0
        self.num_levels = num_levels - 1  # Total num_levels layers
        if downsample_mode not in {'nearest', 'maxpool'}:
            raise ValueError(
                "downsample_mode must be one of {'nearest', 'maxpool'}, "
                f'got {downsample_mode!r}.')
        self.downsample_mode = downsample_mode

    def _pyramid_down(self, input: torch.Tensor, pad_mode: str = 'constant') -> torch.Tensor:
        """Downsample: blur + configurable pyramid downsample.

        The CVPR 2023 supplementary material text suggests nearest-neighbor
        downsampling after Gaussian filtering, while the ADer proxy
        implementation uses max-pooling. Strict targeted A/B needs to compare
        both behaviors.
        """
        if not len(input.shape) == 4:
            raise ValueError(f'Invalid img shape, we expect BCHW, got: {input.shape}')
        img_pad = F.pad(input, (2, 2, 2, 2), mode=pad_mode)
        img_blur = kornia_filter2d(img_pad, kernel=self.kernel)
        if self.downsample_mode == 'nearest':
            out = F.interpolate(
                img_blur,
                size=(input.shape[2] // 2, input.shape[3] // 2),
                mode='nearest',
            )
        else:
            out = F.max_pool2d(img_blur, kernel_size=2, stride=2)
        return out

    def _pyramid_up(self, input: torch.Tensor, size: Tuple[int, int],
                    pad_mode: str = 'constant') -> torch.Tensor:
        """Upsample: nearest interpolate + blur."""
        if not len(input.shape) == 4:
            raise ValueError(f'Invalid img shape, we expect BCHW, got: {input.shape}')
        img_up = F.interpolate(input, size=size, mode='nearest')
        img_pad = F.pad(img_up, (2, 2, 2, 2), mode=pad_mode)
        img_blur = kornia_filter2d(img_pad, kernel=self.kernel)
        return img_blur

    def build_pyramid(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Build Laplacian pyramid from input tensor.

        Returns:
            Laplacian pyramid as tuple of tensors, from finest to coarsest level.
        """
        gp, lp = [input], []
        for _ in range(self.num_levels):
            gp.append(self._pyramid_down(gp[-1]))
        for layer in range(self.num_levels):
            curr_gp = gp[layer]
            next_gp = self._pyramid_up(gp[layer + 1], size=curr_gp.shape[2:])
            lp.append(curr_gp - next_gp)
        lp.append(gp[self.num_levels])  # Last layer is the residual
        return tuple(lp)

    def compose_pyramid(self, lp: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Reconstruct from Laplacian pyramid."""
        rs = lp[-1]
        for i in range(len(lp) - 2, -1, -1):
            rs = self._pyramid_up(rs, size=lp[i].shape[2:])
            rs = torch.add(rs, lp[i])
        return rs


class VolumeNorm(nn.Module):
    """Volume Normalization.

    CVN dims = (0, 1) for object images (normalize across batch and channel)
    SVN dims = (0, 2, 3) for texture images (normalize across batch and spatial dims)
    """

    def __init__(self, dims: Tuple[int, ...] = (0, 1)):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(1, 1, 1, 1))
        self.momentum = 0.1
        self.dims = dims

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'running_mean'
        if key in state_dict:
            loaded = state_dict[key]
            if self.running_mean.shape != loaded.shape:
                self.running_mean = loaded.clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            sample_mean = torch.mean(x, dim=self.dims, keepdim=True)
            if self.running_mean.shape != sample_mean.shape:
                self.running_mean = torch.zeros_like(sample_mean)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * sample_mean
            out = x - sample_mean
        else:
            out = x - self.running_mean
        return out


class AffineParamBlock(nn.Module):
    """Estimate scale (slog) and bias (t) for affine coupling.

    Uses soft clipping: clamp * 0.636 * atan(x / clamp) to bound the scale.
    """

    def __init__(self, in_ch: int, out_ch: Optional[int] = None,
                 hidden_ch: Optional[int] = None, ksize: int = 7,
                 clamp: float = 2.0, vn_dims: Tuple[int, ...] = (0, 1)):
        super().__init__()
        if out_ch is None:
            out_ch = 2 * in_ch
        if hidden_ch is None:
            hidden_ch = out_ch
        self.clamp = clamp
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=ksize, padding=ksize // 2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=ksize, padding=ksize // 2, bias=False),
        )
        nn.init.zeros_(self.conv[-1].weight.data)
        self.norm = VolumeNorm(vn_dims)

    def forward(self, input: torch.Tensor, forward_mode: bool) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        output = self.conv(input)
        _dlogdet, bias = output.chunk(2, 1)
        dlogdet = self.clamp * 0.636 * torch.atan(_dlogdet / self.clamp)  # soft clip
        dlogdet = self.norm(dlogdet)
        scale = torch.exp(dlogdet)
        return (scale, bias), dlogdet


class InvConv2dLU(nn.Module):
    """Invertible 1x1 Conv with PLU decomposition and volume normalization."""

    def __init__(self, in_channel: int, volumeNorm: bool = True):
        super().__init__()
        self.volumeNorm = volumeNorm
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(w_s.abs().log())
        self.w_u = nn.Parameter(w_u)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        return out

    def inverse(self, output: torch.Tensor) -> torch.Tensor:
        weight = self.calc_weight()
        inv_weight = torch.inverse(weight.squeeze().double()).float()
        input = F.conv2d(output, inv_weight.unsqueeze(2).unsqueeze(3))
        return input

    def calc_weight(self) -> torch.Tensor:
        if self.volumeNorm:
            w_s = self.w_s - self.w_s.mean()  # volume normalization
        else:
            w_s = self.w_s
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)


class FlowBlock(InvertibleModule):
    """Scale-wise pyramid coupling block (Paper Figure 3(c)).

    Processes two adjacent pyramid levels, either:
    - direct='up': interpolate finer level to coarser, apply affine to coarser
    - direct='down': apply affine to coarser, interpolate to finer
    """

    def __init__(self, channel: int, direct: str, start_level: int,
                 ksize: int, vn_dims: Tuple[int, ...]):
        super().__init__()
        assert direct in ['up', 'down']
        self.direct = direct
        self.start_idx = start_level
        self.affineParams = AffineParamBlock(channel, ksize=ksize, vn_dims=vn_dims)
        self.conv1x1 = InvConv2dLU(channel)

    def forward(self, inputs: Tuple[torch.Tensor, ...],
                logdets: Tuple[torch.Tensor, ...]):
        assert self.start_idx + 1 < len(inputs)
        x0, x1 = inputs[self.start_idx: self.start_idx + 2]
        logdet0, logdet1 = logdets[self.start_idx: self.start_idx + 2]

        if self.direct == 'up':
            y10 = F.interpolate(x1, size=x0.shape[2:], mode='nearest')
            (scale0, bias0), dlogdet0 = self.affineParams(y10, forward_mode=True)
            z0, z1 = scale0 * x0 + bias0, x1
            z0 = self.conv1x1(z0)
            dlogdet1 = 0
        else:
            (scale10, bias10), dlogdet10 = self.affineParams(x0, forward_mode=True)
            scale1 = F.interpolate(scale10, size=x1.shape[2:], mode='nearest')
            bias1 = F.interpolate(bias10, size=x1.shape[2:], mode='nearest')
            dlogdet1 = F.interpolate(dlogdet10, size=x1.shape[2:], mode='nearest')
            z0, z1 = x0, scale1 * x1 + bias1
            z1 = self.conv1x1(z1)
            dlogdet0 = 0

        outputs = inputs[:self.start_idx] + (z0, z1) + inputs[self.start_idx + 2:]
        out_logdets = logdets[:self.start_idx] + (logdet0 + dlogdet0, logdet1 + dlogdet1) + logdets[self.start_idx + 2:]
        return outputs, out_logdets

    def inverse(self, outputs: Tuple[torch.Tensor, ...],
                logdets: Tuple[torch.Tensor, ...]):
        assert self.start_idx + 1 < len(outputs)
        z0, z1 = outputs[self.start_idx: self.start_idx + 2]
        logdet0, logdet1 = logdets[self.start_idx: self.start_idx + 2]

        if self.direct == 'up':
            z0 = self.conv1x1.inverse(z0)
            z10 = F.interpolate(z1, size=z0.shape[2:], mode='nearest')
            (scale0, bias0), dlogdet0 = self.affineParams(z10, forward_mode=False)
            x0, x1 = (z0 - bias0) / scale0, z1
            dlogdet1 = 0
        else:
            z1 = self.conv1x1.inverse(z1)
            (scale01, bias01), dlogdet01 = self.affineParams(z0, forward_mode=False)
            scale1 = F.interpolate(scale01, size=z1.shape[2:], mode='nearest')
            bias1 = F.interpolate(bias01, size=z1.shape[2:], mode='nearest')
            dlogdet1 = F.interpolate(dlogdet01, size=z1.shape[2:], mode='nearest')
            x0, x1 = z0, (z1 - bias1) / scale1
            dlogdet0 = 0

        inputs = outputs[:self.start_idx] + (x0, x1) + outputs[self.start_idx + 2:]
        in_logdets = logdets[:self.start_idx] + (logdet0 - dlogdet0, logdet1 - dlogdet1) + logdets[self.start_idx + 2:]
        return inputs, in_logdets


class FlowBlock2(InvertibleModule):
    """Reverse parallel pyramid coupling block (Paper Figure 3(d)).

    Uses three-scale context: upsampled x0 and x2 concatenated for affine input,
    applied to middle scale x1.
    """

    def __init__(self, channel: int, start_level: int,
                 ksize: int, vn_dims: Tuple[int, ...]):
        super().__init__()
        self.start_idx = start_level
        self.affineParams = AffineParamBlock(in_ch=2 * channel, out_ch=2 * channel,
                                             ksize=ksize, vn_dims=vn_dims)
        self.conv1x1 = InvConv2dLU(channel)

    def forward(self, inputs: Tuple[torch.Tensor, ...],
                logdets: Tuple[torch.Tensor, ...]):
        x0, x1, x2 = inputs[self.start_idx: self.start_idx + 3]
        logdet0, logdet1, logdet2 = logdets[self.start_idx: self.start_idx + 3]

        y01 = F.interpolate(x0, size=x1.shape[2:], mode='nearest')
        y21 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
        affine_input = torch.concat([y01, y21], dim=1)  # (B, 2*ch, H, W)
        (scale1, bias1), dlogdet1 = self.affineParams(affine_input, forward_mode=True)
        z0, z1, z2 = x0, scale1 * x1 + bias1, x2
        z1 = self.conv1x1(z1)

        outputs = inputs[:self.start_idx] + (z0, z1, z2) + inputs[self.start_idx + 3:]
        out_logdets = logdets[:self.start_idx] + (logdet0, logdet1 + dlogdet1, logdet2) + logdets[self.start_idx + 3:]
        return outputs, out_logdets

    def inverse(self, outputs: Tuple[torch.Tensor, ...],
                logdets: Tuple[torch.Tensor, ...]):
        z0, z1, z2 = outputs[self.start_idx: self.start_idx + 3]
        logdet0, logdet1, logdet2 = logdets[self.start_idx: self.start_idx + 3]

        z1 = self.conv1x1.inverse(z1)
        z01 = F.interpolate(z0, size=z1.shape[2:], mode='nearest')
        z21 = F.interpolate(z2, size=z1.shape[2:], mode='nearest')
        affine_input = torch.concat([z01, z21], dim=1)
        (scale1, bias1), dlogdet1 = self.affineParams(affine_input, forward_mode=False)
        x0, x1, x2 = z0, (z1 - bias1) / scale1, z2

        inputs = outputs[:self.start_idx] + (x0, x1, x2) + outputs[self.start_idx + 3:]
        in_logdets = logdets[:self.start_idx] + (logdet0, logdet1 - dlogdet1, logdet2) + logdets[self.start_idx + 3:]
        return inputs, in_logdets


class BatchDiffLoss(nn.Module):
    """Difference Loss within a batch for pair training.

    Computes L_p norm of differences between all pairs in a batch.
    """

    def __init__(self, batchsize: int = 2, p: int = 2) -> None:
        super().__init__()
        self.batchsize = batchsize
        self.p = p

    def forward(self, pyramid: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        diffes = []
        for inp in pyramid:
            b = inp.shape[0]
            if b < 2:
                # Single sample: return zeros (no pairs to compare)
                diffes.append(torch.zeros_like(inp))
                continue
            idx0, idx1 = np.triu_indices(n=b, k=1)
            diff = (inp[idx0] - inp[idx1]).abs() ** self.p
            diffes.append(diff)
        return tuple(diffes)


class PyramidFlowCore(nn.Module):
    """Core PyramidFlow model (from ADer reference).

    Args:
        encoder: 'resnet18', 'tv_resnet18', or None for 1x1 conv
        channel: Feature channels (default 64)
        num_level: Number of pyramid levels (default 4)
        num_stack: Number of flow blocks per pattern (default 4)
        ksize: Conv kernel size (default 7)
        vn_dims: Volume normalization dims (0,1) for CVN, (0,2,3) for SVN
        batch_size: Batch size for pair training (default 2)
        save_memory: Use memory-saving gradient checkpointing
    """

    def __init__(self,
                 encoder: Optional[str] = 'resnet18',
                 channel: int = 64,
                 num_level: int = 4,
                 num_stack: int = 4,
                 ksize: int = 7,
                 vn_dims: Tuple[int, ...] = (0, 1),
                 batch_size: int = 2,
                 save_memory: bool = False,
                 pyramid_downsample_mode: str = 'nearest'):
        super().__init__()
        assert num_level >= 2
        self.channel = channel
        self.num_level = num_level

        # Build encoder
        if encoder in ['resnet18', 'tv_resnet18']:
            resnet = _build_pyramidflow_resnet18()
            self.inconv = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1
            )  # 1024 -> 256 for 256x256 input, outputs 64x64
        else:
            self.inconv = SemiInvertible1x1Conv(3, self.channel)

        # Build flow modules
        modules = []
        for _ in range(num_stack):
            for range_start in [0, 1]:
                if range_start == 1:
                    modules.append(FlowBlock(self.channel, direct='up', start_level=0,
                                            ksize=ksize, vn_dims=vn_dims))
                for start_idx in range(range_start, num_level, 2):
                    if start_idx + 2 < num_level:
                        modules.append(FlowBlock2(self.channel, start_level=start_idx,
                                                 ksize=ksize, vn_dims=vn_dims))
                    elif start_idx + 1 < num_level:
                        modules.append(FlowBlock(self.channel, direct='down', start_level=start_idx,
                                                ksize=ksize, vn_dims=vn_dims))

        self.nf = SequentialNF(modules) if save_memory else SequentialNet(modules)
        self.pyramid = LaplacianMaxPyramid(num_level, downsample_mode=pyramid_downsample_mode)
        self.loss = BatchDiffLoss(batchsize=batch_size, p=2)

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.inconv.parameters():
            param.requires_grad = False
        self.inconv.eval()

    def forward_train(self, imgs: torch.Tensor) -> torch.Tensor:
        """Training forward: compute batch difference for FFT loss."""
        b, c, h, w = imgs.shape
        assert h % (2 ** (self.num_level - 1)) == 0 and w % (2 ** (self.num_level - 1)) == 0

        with torch.no_grad():
            feat1 = self.inconv(imgs)  # encoder is frozen

        pyramid = self.pyramid.build_pyramid(feat1)
        logdets = tuple(torch.zeros_like(p) for p in pyramid)
        pyramid_out, _ = self.nf.forward(pyramid, logdets)
        diffes = self.loss(pyramid_out)
        diff_pixel = self.pyramid.compose_pyramid(diffes).mean(1)
        return diff_pixel

    def encode_to_latent(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Encode images to latent pyramid (for template building and inference)."""
        b, c, h, w = imgs.shape
        assert h % (2 ** (self.num_level - 1)) == 0 and w % (2 ** (self.num_level - 1)) == 0

        with torch.no_grad():
            feat1 = self.inconv(imgs)

        pyramid = self.pyramid.build_pyramid(feat1)
        logdets = tuple(torch.zeros_like(p) for p in pyramid)
        pyramid_out, _ = self.nf.forward(pyramid, logdets)
        return pyramid_out

    def predict(self, imgs: torch.Tensor,
                template: Tuple[torch.Tensor, ...],
                multi_view: int = 1,
                drop_levels: Optional[Sequence[int]] = None,
                level_weights: Optional[Sequence[float]] = None) -> torch.Tensor:
        """Inference: contrastive localization against template.

        Args:
            imgs: Input images (B, 3, H, W)
            template: Latent template tuple (mean of training latents)
            multi_view: Number of views per sample (for multi-view inference)

        Returns:
            Anomaly map (B, 1, H, W)
        """
        pyramid_out = self.encode_to_latent(imgs)

        if template[0].size(0) == 1 or multi_view == 1:
            pyramid_diff = [(z - t).abs() for z, t in zip(pyramid_out, template)]
        else:
            pyramid_diff = []
            for z, t in zip(pyramid_out, template):
                feat = rearrange(z, '(b v) c h w -> b v c h w', v=multi_view)
                pyramid_diffs = []
                for i in range(feat.size(0)):
                    pyramid_diff_s = (feat[i] - t).abs()
                    pyramid_diffs.append(pyramid_diff_s.unsqueeze(dim=0))
                pyramid_diffs = torch.cat(pyramid_diffs, dim=0)
                pyramid_diffs = rearrange(pyramid_diffs, 'b v c h w -> (b v) c h w')
                pyramid_diff.append(pyramid_diffs)

        if drop_levels:
            drop_levels = {int(level) for level in drop_levels}
            pyramid_diff = [
                torch.zeros_like(level_diff) if idx in drop_levels else level_diff
                for idx, level_diff in enumerate(pyramid_diff)
            ]
        if level_weights:
            pyramid_diff = [
                level_diff * float(level_weights[idx])
                for idx, level_diff in enumerate(pyramid_diff)
            ]

        diff = self.pyramid.compose_pyramid(pyramid_diff).mean(1, keepdim=True)
        return diff

    def inverse(self, pyramid_out: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Inverse flow to reconstruct features from latent."""
        logdets_out = tuple(torch.zeros_like(p) for p in pyramid_out)
        pyramid_in, _ = self.nf.inverse(pyramid_out, logdets_out)
        feat1 = self.pyramid.compose_pyramid(pyramid_in)
        if self.channel != 64:
            input = self.inconv.inverse(feat1)
            return input
        return feat1


@MODELS.register_module(force=True)
class PyramidFlowDetector(FlowBasedADModel):
    """PyramidFlow: High-Resolution Defect Contrastive Localization.

    Faithful implementation of CVPR 2023 paper.

    Args:
        encoder: Encoder type ('resnet18', 'tv_resnet18', or None)
        channel: Feature channels (default 64)
        num_level: Number of pyramid levels (default 4)
        num_stack: Number of flow blocks per pattern (default 4)
        ksize: Conv kernel size (default 7)
        vn_dims: Volume normalization dims (0,1) for CVN, (0,2,3) for SVN
        save_memory: Use memory-saving gradient checkpointing
        data_preprocessor: Data preprocessor config
    """

    def __init__(self,
                 encoder: str = 'resnet18',
                 channel: int = 64,
                 num_level: int = 4,
                 num_stack: int = 4,
                 ksize: int = 7,
                 vn_dims: Tuple[int, ...] = (0, 1),
                 save_memory: bool = False,
                 pyramid_downsample_mode: str = 'nearest',
                 predict_resize_to_input: bool = True,
                 predict_drop_levels: Optional[Sequence[int]] = None,
                 predict_level_weights: Optional[Sequence[float]] = None,
                 template_pipeline: Optional[Sequence[dict]] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.core = PyramidFlowCore(
            encoder=encoder,
            channel=channel,
            num_level=num_level,
            num_stack=num_stack,
            ksize=ksize,
            vn_dims=vn_dims,
            batch_size=2,  # Fixed for pair training
            save_memory=save_memory,
            pyramid_downsample_mode=pyramid_downsample_mode,
        )

        # Freeze encoder
        self.core.freeze_encoder()

        # Template storage (built from training data during build_memory_bank)
        self.template = None
        self._template_built = False
        # The published protocol estimates the latent template from training normals.
        self.template_dataloader_split = 'train'
        # ADer's PyramidFlow trainer rebuilds the template with a dedicated
        # single-sample loader before test-time prediction.
        self.template_batch_size = 1
        self.template_drop_last = False
        self.template_pipeline = copy.deepcopy(list(template_pipeline)) if template_pipeline is not None else None
        self.predict_resize_to_input = bool(predict_resize_to_input)
        self.predict_drop_levels = self._normalize_predict_drop_levels(predict_drop_levels, num_level)
        self.predict_level_weights = self._normalize_predict_level_weights(predict_level_weights, num_level)

    @staticmethod
    def _normalize_predict_drop_levels(
        predict_drop_levels: Optional[Sequence[int]],
        num_level: int,
    ) -> tuple[int, ...]:
        if predict_drop_levels is None:
            return ()
        normalized = tuple(sorted({int(level) for level in predict_drop_levels}))
        invalid = [level for level in normalized if level < 0 or level >= num_level]
        if invalid:
            raise ValueError(
                f'predict_drop_levels contains invalid levels {invalid}; '
                f'valid range is [0, {num_level - 1}].'
            )
        return normalized

    @staticmethod
    def _normalize_predict_level_weights(
        predict_level_weights: Optional[Sequence[float]],
        num_level: int,
    ) -> tuple[float, ...]:
        if predict_level_weights is None:
            return ()
        if len(predict_level_weights) != num_level:
            raise ValueError(
                'predict_level_weights must provide one weight per pyramid level; '
                f'got {len(predict_level_weights)} weights for num_level={num_level}.'
            )
        return tuple(float(weight) for weight in predict_level_weights)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[Sequence] = None,
                mode: str = 'tensor'):
        """Forward pass.

        Args:
            inputs: Input images
            data_samples: ADDataSample list (for predict mode)
            mode: 'loss', 'predict', or 'tensor'

        Returns:
            - mode='loss': dict with 'loss' key
            - mode='predict': list of ADDataSample with predictions
            - mode='tensor': raw encoder features
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            return self._forward_train(inputs)
        elif mode == 'predict':
            return self._forward_predict(inputs, data_samples)
        else:
            # tensor mode - return encoder features
            with torch.no_grad():
                return self.core.inconv(inputs)

    def _forward_train(self, imgs: torch.Tensor) -> dict:
        """Training forward: compute FFT loss on batch differences."""
        diff_pixel = self.core.forward_train(imgs)
        # FFT loss: L1 on FFT magnitude
        loss = torch.fft.fft2(diff_pixel).abs().mean()
        return {'loss': loss}

    def _forward_predict(self,
                         imgs: torch.Tensor,
                         data_samples: Optional[Sequence]) -> list:
        """Inference forward: contrastive localization against template."""
        if self.template is None:
            raise RuntimeError("Template not built. Call build_memory_bank() first or ensure MemoryBankHook is active.")

        with torch.no_grad():
            anomaly_map = self.core.predict(
                imgs,
                self.template,
                drop_levels=self.predict_drop_levels,
                level_weights=self.predict_level_weights,
            )
            if self.predict_resize_to_input:
                anomaly_map = F.interpolate(
                    anomaly_map,
                    size=imgs.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )

        flat_maps = anomaly_map.view(imgs.shape[0], -1)
        img_scores_mean = flat_maps.mean(dim=1)
        img_scores_max = flat_maps.max(dim=1).values

        # Keep the historical default as spatial mean while exposing alternative
        # score fields for strict diagnostics.
        return build_predict_results(
            data_samples,
            img_scores_mean,
            anomaly_map,
            extra_scores=dict(
                pred_score_mean=img_scores_mean,
                pred_score_max=img_scores_max,
            ),
        )

    def build_memory_bank(self) -> None:
        """Build latent template from training data.

        This method is called by MemoryBankHook after training.
        It requires access to the training dataloader, which we get from
        the runner through the hook context.

        Note: This implementation uses a workaround - the actual template
        building happens in after_train_epoch when the runner is available.
        """
        # This is a placeholder - actual template building requires
        # access to the dataloader which is handled by the hook
        if self._template_built:
            return
        # Template will be built via build_template_from_dataloader

    def _build_template_dataloader_view(self, dataloader) -> DataLoader:
        """Clone a deterministic single-sample view for template estimation.

        ADer's PyramidFlow trainer reconstructs a new loader with
        ``batch_size_per_gpu = 1`` before template prediction. Reusing the
        training loader directly keeps BaoIAD at pair-training batch semantics
        and can skew the latent mean for strict alignment.
        """
        source_dataset = getattr(dataloader, 'dataset', None)
        if source_dataset is None:
            raise TypeError('Template dataloader must expose a dataset.')
        dataset = source_dataset
        if self.template_pipeline is not None:
            dataset = copy.copy(source_dataset)
            dataset.pipeline = Compose(copy.deepcopy(self.template_pipeline))

        num_workers = int(getattr(dataloader, 'num_workers', 0) or 0)
        persistent_workers = bool(getattr(dataloader, 'persistent_workers', False)) if num_workers > 0 else False
        collate_fn = getattr(dataloader, 'collate_fn', None)

        loader_kwargs = dict(
            dataset=dataset,
            batch_size=self.template_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(getattr(dataloader, 'pin_memory', False)),
            drop_last=self.template_drop_last,
            timeout=getattr(dataloader, 'timeout', 0),
            worker_init_fn=getattr(dataloader, 'worker_init_fn', None),
            persistent_workers=persistent_workers,
        )

        generator = getattr(dataloader, 'generator', None)
        if generator is not None:
            loader_kwargs['generator'] = generator

        multiprocessing_context = getattr(dataloader, 'multiprocessing_context', None)
        if multiprocessing_context is not None:
            loader_kwargs['multiprocessing_context'] = multiprocessing_context

        pin_memory_device = getattr(dataloader, 'pin_memory_device', '')
        if pin_memory_device:
            loader_kwargs['pin_memory_device'] = pin_memory_device

        if num_workers > 0:
            prefetch_factor = getattr(dataloader, 'prefetch_factor', None)
            if prefetch_factor is not None:
                loader_kwargs['prefetch_factor'] = prefetch_factor

        return DataLoader(**loader_kwargs)

    @staticmethod
    def _extract_template_inputs(data_batch: Any, device: torch.device) -> torch.Tensor:
        """Extract image tensors from a template dataloader batch."""
        if isinstance(data_batch, dict):
            if 'inputs' in data_batch:
                imgs = data_batch['inputs']
            elif 'img' in data_batch:
                imgs = data_batch['img']
            else:
                for key in ['inputs', 'img', 'images', 'pixel_values']:
                    if key in data_batch and isinstance(data_batch[key], torch.Tensor):
                        imgs = data_batch[key]
                        break
                else:
                    raise KeyError(f'Could not find image tensor in batch. Keys: {data_batch.keys()}')
        else:
            imgs = data_batch[0] if isinstance(data_batch, (list, tuple)) else data_batch

        if isinstance(imgs, (list, tuple)):
            imgs = torch.stack(imgs).to(device)
        else:
            imgs = imgs.to(device)
        if imgs.dim() == 5:
            imgs = imgs.squeeze(1)
        return imgs

    def build_template_from_dataloader(self, dataloader, device: torch.device) -> None:
        """Build template by running a single-sample train-normal loader.

        Args:
            dataloader: Train-normal dataloader
            device: Device to use
        """
        self.eval()
        self.to(device)

        template_loader = self._build_template_dataloader_view(dataloader)
        feat_sums = None
        num_batches = 0

        for data_batch in template_loader:
            imgs = self._extract_template_inputs(data_batch, device)

            with torch.no_grad():
                pyramid_out = self.core.encode_to_latent(imgs)

            if feat_sums is None:
                feat_sums = [torch.zeros_like(p[:1]) for p in pyramid_out]

            for i, p in enumerate(pyramid_out):
                # Match ADer's "sum batches then divide by val_length" semantics.
                feat_sums[i] = feat_sums[i] + p.detach().mean(dim=0, keepdim=True)
            num_batches += 1

        if not num_batches or feat_sums is None:
            raise RuntimeError('Cannot build PyramidFlow template from an empty dataloader.')

        template = tuple(feat_sum / num_batches for feat_sum in feat_sums)
        self.template = template
        self._template_built = True

    def train(self, mode: bool = True):
        """Override train to keep encoder frozen."""
        super().train(mode)
        self.core.freeze_encoder()
        return self


# For backward compatibility, keep the simplified config parameters working
# by providing alternative constructor
def create_pyramidflow_from_simplified_config(
        backbone: str = 'wide_resnet50_2',
        n_coupling_blocks: int = 4,
        clamp: float = 2.0,
        scales: tuple = (0, 1, 2),
        **kwargs):
    """Factory for backward compatibility with simplified config.

    WARNING: This creates a simplified variant, not the faithful PyramidFlow.
    """
    raise NotImplementedError(
        "The simplified PyramidFlow variant has been removed. "
        "Please use the faithful implementation with encoder='resnet18'."
    )
