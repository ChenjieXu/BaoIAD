"""CS-Flow: Fully Convolutional Cross-Scale-Flows (WACV 2022).

Faithful reimplementation with:
- EfficientNet-B5 multi-scale feature extraction (3 scales via features.6.8)
- Cross-scale coupling layers with parallel GLOW-style affine transforms
- ParallelPermute for channel shuffling across scales
- Anomaly map via product of per-scale mean z^2
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from typing import Union
from FrEIA.framework import GraphINN, InputNode, Node, OutputNode
from FrEIA.modules import InvertibleModule
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import FlowBasedADModel


# ─── Cross-scale convolutions ──────────────────────────────────────────────

class CrossConvolutions(nn.Module):
    """Cross convolution across three scales with up/down connections."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        channels_hidden: int = 512,
        kernel_size: int = 3,
        leaky_slope: float = 0.1,
        batch_norm: bool = False,
        use_gamma: bool = True,
    ):
        super().__init__()
        pad = kernel_size // 2
        pad_mode = "zeros"
        self.use_gamma = use_gamma
        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden, kernel_size, padding=pad,
                                         bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden, kernel_size, padding=pad,
                                         bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden, kernel_size, padding=pad,
                                         bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale0_1 = nn.Conv2d(channels_hidden, channels, kernel_size, padding=pad,
                                         bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden, channels, kernel_size, padding=pad,
                                         bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden, channels, kernel_size, padding=pad,
                                         bias=not batch_norm, padding_mode=pad_mode)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv10 = nn.Conv2d(channels_hidden, channels, kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.up_conv21 = nn.Conv2d(channels_hidden, channels, kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.down_conv01 = nn.Conv2d(channels_hidden, channels, kernel_size, padding=pad, bias=not batch_norm,
                                       stride=2, padding_mode=pad_mode)
        self.down_conv12 = nn.Conv2d(channels_hidden, channels, kernel_size, padding=pad, bias=not batch_norm,
                                       stride=2, padding_mode=pad_mode)
        self.leaky_relu = nn.LeakyReLU(leaky_slope)

    def forward(self, scale0, scale1, scale2):
        out0 = self.conv_scale0_0(scale0)
        out1 = self.conv_scale1_0(scale1)
        out2 = self.conv_scale2_0(scale2)

        lr0 = self.leaky_relu(out0)
        lr1 = self.leaky_relu(out1)
        lr2 = self.leaky_relu(out2)

        out0 = self.conv_scale0_1(lr0)
        out1 = self.conv_scale1_1(lr1)
        out2 = self.conv_scale2_1(lr2)

        y1_up = self.up_conv10(self.upsample(lr1))
        y2_up = self.up_conv21(self.upsample(lr2))
        y0_down = self.down_conv01(lr0)
        y1_down = self.down_conv12(lr1)

        out0 = out0 + y1_up
        out1 = out1 + y0_down + y2_up
        out2 = out2 + y1_down

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2

        return out0, out1, out2


# ─── Parallel Permute ──────────────────────────────────────────────────────

class ParallelPermute(InvertibleModule):
    """Fixed random permutation applied in parallel to multiple inputs."""

    def __init__(self, dims_in, seed=None):
        super().__init__(dims_in)
        self.n_inputs = len(dims_in)
        self.in_channels = [dims_in[i][0] for i in range(self.n_inputs)]
        self.perm = []
        self.perm_inv = []
        for i in range(self.n_inputs):
            rng = np.random.default_rng(seed)
            p = rng.permutation(self.in_channels[i])
            p_inv = np.zeros_like(p)
            for idx, val in enumerate(p):
                p_inv[val] = idx
            self.perm.append(torch.LongTensor(p))
            self.perm_inv.append(torch.LongTensor(p_inv))

    def forward(self, input_tensor, rev=False, jac=True):
        if not rev:
            return [input_tensor[i][:, self.perm[i]] for i in range(self.n_inputs)], 0.0
        return [input_tensor[i][:, self.perm_inv[i]] for i in range(self.n_inputs)], 0.0

    @staticmethod
    def output_dims(input_dims):
        return input_dims


# ─── Parallel GLOW Coupling Layer ──────────────────────────────────────────

class ParallelGlowCouplingLayer(InvertibleModule):
    """GLOW-style coupling applied in parallel across 3 scales."""

    def __init__(self, dims_in, subnet_args, clamp=5.0):
        super().__init__(dims_in)
        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.clamp = clamp

        self.cross_convolution1 = CrossConvolutions(self.split_len1, self.split_len2 * 2, **subnet_args)
        self.cross_convolution2 = CrossConvolutions(self.split_len2, self.split_len1 * 2, **subnet_args)

    def _exp(self, s):
        return torch.exp(self._log_e(s)) if self.clamp > 0 else torch.exp(s)

    def _log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp) if self.clamp > 0 else s

    def forward(self, input_tensor, rev=False, jac=True):
        x01, x02 = input_tensor[0].narrow(1, 0, self.split_len1), input_tensor[0].narrow(1, self.split_len1, self.split_len2)
        x11, x12 = input_tensor[1].narrow(1, 0, self.split_len1), input_tensor[1].narrow(1, self.split_len1, self.split_len2)
        x21, x22 = input_tensor[2].narrow(1, 0, self.split_len1), input_tensor[2].narrow(1, self.split_len1, self.split_len2)

        if not rev:
            r02, r12, r22 = self.cross_convolution2(x02, x12, x22)
            s02, t02 = r02[:, :self.split_len1], r02[:, self.split_len1:]
            s12, t12 = r12[:, :self.split_len1], r12[:, self.split_len1:]
            s22, t22 = r22[:, :self.split_len1], r22[:, self.split_len1:]

            y01 = self._exp(s02) * x01 + t02
            y11 = self._exp(s12) * x11 + t12
            y21 = self._exp(s22) * x21 + t22

            r01, r11, r21 = self.cross_convolution1(y01, y11, y21)
            s01, t01 = r01[:, :self.split_len2], r01[:, self.split_len2:]
            s11, t11 = r11[:, :self.split_len2], r11[:, self.split_len2:]
            s21, t21 = r21[:, :self.split_len2], r21[:, self.split_len2:]

            y02 = self._exp(s01) * x02 + t01
            y12 = self._exp(s11) * x12 + t11
            y22 = self._exp(s21) * x22 + t21
        else:
            r01, r11, r21 = self.cross_convolution1(x01, x11, x21)
            s01, t01 = r01[:, :self.split_len2], r01[:, self.split_len2:]
            s11, t11 = r11[:, :self.split_len2], r11[:, self.split_len2:]
            s21, t21 = r21[:, :self.split_len2], r21[:, self.split_len2:]

            y02 = (x02 - t01) / self._exp(s01)
            y12 = (x12 - t11) / self._exp(s11)
            y22 = (x22 - t21) / self._exp(s21)

            r02, r12, r22 = self.cross_convolution2(y02, y12, y22)
            s02, t02 = r02[:, :self.split_len2], r01[:, self.split_len2:]
            s12, t12 = r12[:, :self.split_len2], r11[:, self.split_len2:]
            s22, t22 = r22[:, :self.split_len2], r21[:, self.split_len2:]

            y01 = (x01 - t02) / self._exp(s02)
            y11 = (x11 - t12) / self._exp(s12)
            y21 = (x21 - t22) / self._exp(s22)

        z0 = torch.clamp(torch.cat((y01, y02), 1), -1e6, 1e6)
        z1 = torch.clamp(torch.cat((y11, y12), 1), -1e6, 1e6)
        z2 = torch.clamp(torch.cat((y21, y22), 1), -1e6, 1e6)

        jac0 = torch.sum(self._log_e(s01), dim=(1, 2, 3)) + torch.sum(self._log_e(s02), dim=(1, 2, 3))
        jac1 = torch.sum(self._log_e(s11), dim=(1, 2, 3)) + torch.sum(self._log_e(s12), dim=(1, 2, 3))
        jac2 = torch.sum(self._log_e(s21), dim=(1, 2, 3)) + torch.sum(self._log_e(s22), dim=(1, 2, 3))

        return [z0, z1, z2], torch.stack([jac0, jac1, jac2], dim=1).sum()

    @staticmethod
    def output_dims(input_dims):
        return input_dims


# ─── Cross-Scale Flow graph ────────────────────────────────────────────────

class CrossScaleFlow(nn.Module):
    """Cross-scale normalizing flow graph using FrEIA."""

    def __init__(self, input_dims, n_coupling_blocks, clamp, cross_conv_hidden_channels):
        super().__init__()
        self.input_dims = input_dims
        self.n_coupling_blocks = n_coupling_blocks
        self.kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]
        self.clamp = clamp
        self.cross_conv_hidden_channels = cross_conv_hidden_channels
        self.graph = self._create_graph()

    def _create_graph(self):
        nodes = []
        # 304 channels from EfficientNet-B5 features.6
        input_nodes = [
            InputNode(304, self.input_dims[1] // 32, self.input_dims[2] // 32, name="input"),
            InputNode(304, self.input_dims[1] // 64, self.input_dims[2] // 64, name="input2"),
            InputNode(304, self.input_dims[1] // 128, self.input_dims[2] // 128, name="input3"),
        ]
        nodes.extend(input_nodes)

        for cb in range(self.n_coupling_blocks):
            if cb == 0:
                node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
            else:
                node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

            permute_node = Node(
                inputs=node_to_permute,
                module_type=ParallelPermute,
                module_args={"seed": cb},
                name=f"permute_{cb}",
            )
            nodes.append(permute_node)

            coupling_node = Node(
                inputs=[nodes[-1].out0, nodes[-1].out1, nodes[-1].out2],
                module_type=ParallelGlowCouplingLayer,
                module_args={
                    "clamp": self.clamp,
                    "subnet_args": {
                        "channels_hidden": self.cross_conv_hidden_channels,
                        "kernel_size": self.kernel_sizes[cb],
                    },
                },
                name=f"coupling_{cb}",
            )
            nodes.append(coupling_node)

        output_nodes = [
            OutputNode([nodes[-1].out0], name="output_end0"),
            OutputNode([nodes[-1].out1], name="output_end1"),
            OutputNode([nodes[-1].out2], name="output_end2"),
        ]
        nodes.extend(output_nodes)
        return GraphINN(nodes)

    def forward(self, inputs):
        return self.graph(inputs)


# ─── CS-Flow Detector ──────────────────────────────────────────────────────

@MODELS.register_module(force=True)
class CSFlowDetector(FlowBasedADModel):
    """CS-Flow: Cross-Scale Flow for anomaly detection.

    Uses EfficientNet-B5 features at 3 scales with cross-scale
    normalizing flows for density estimation.

    Args:
        input_size (tuple): Input image size (H, W). Default (256, 256).
        n_coupling_blocks (int): Number of coupling blocks. Default 4.
        cross_conv_hidden_channels (int): Hidden channels in cross convolutions. Default 1024.
        clamp (float): Clamping value for affine layers. Default 3.
    """

    def __init__(
        self,
        input_size=(256, 256),
        n_coupling_blocks=4,
        cross_conv_hidden_channels=1024,
        clamp=3,
        image_score_mode='all_scale_mean',
        backbone: Union[str, dict] = None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.input_size = input_size
        self.image_score_mode = image_score_mode
        input_dims = (3, *input_size)

        # Build backbone via registry
        if backbone is None:
            backbone = dict(type='CSFlowFeatureExtractor',
                            n_scales=3, input_size=input_size, frozen=True)
        elif isinstance(backbone, dict):
            backbone = copy.deepcopy(backbone)
        else:
            raise ValueError(f"backbone must be None or dict, got {type(backbone)}")
        self.feature_extractor = MODELS.build(backbone)
        self.feature_extractor.eval()

        self.graph = CrossScaleFlow(
            input_dims=input_dims,
            n_coupling_blocks=n_coupling_blocks,
            clamp=clamp,
            cross_conv_hidden_channels=cross_conv_hidden_channels,
        )

    def _scale_maps(self, z_dist):
        return [
            F.interpolate(
                (z ** 2).mean(dim=1, keepdim=True),
                size=self.input_size,
                mode='bilinear',
                align_corners=False,
            )
            for z in z_dist
        ]

    def _compute_image_scores(self, z_dist):
        if self.image_score_mode == 'all_scale_mean':
            flat = torch.cat([z.reshape(z.shape[0], -1) for z in z_dist], dim=1)
            return torch.mean(flat ** 2 / 2, dim=1)

        if self.image_score_mode.startswith('scale') and self.image_score_mode.endswith('_mean'):
            scale_idx = int(self.image_score_mode[len('scale')])
            z = z_dist[scale_idx]
            flat = z.reshape(z.shape[0], -1)
            return torch.mean(flat ** 2 / 2, dim=1)

        if self.image_score_mode.startswith('scale') and self.image_score_mode.endswith('_map_max'):
            scale_idx = int(self.image_score_mode[len('scale')])
            scale_map = self._scale_maps(z_dist)[scale_idx]
            return scale_map.flatten(1).max(dim=1).values

        raise ValueError(f'Unsupported image_score_mode: {self.image_score_mode}')

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        features = self.feature_extractor(inputs)

        if mode == 'loss':
            z_dist, jacobians = self.graph(features)
            # CS-Flow loss: mean of (0.5 * sum(z^2) - jacobians) / D
            concatenated = torch.cat(
                [z.reshape(z.shape[0], -1) for z in z_dist], dim=1
            )
            loss = torch.mean(
                0.5 * torch.sum(concatenated ** 2, dim=1) - jacobians
            ) / concatenated.shape[1]
            return {'loss': loss}

        elif mode == 'predict':
            z_dist, _ = self.graph(features)
            B = inputs.shape[0]

            img_scores = self._compute_image_scores(z_dist)

            # Anomaly map: product of per-scale mean z^2, upsampled
            anomaly_map = torch.ones(B, 1, *self.input_size, device=inputs.device)
            for scale_map in self._scale_maps(z_dist):
                anomaly_map *= scale_map

            return build_predict_results(data_samples, img_scores, anomaly_map)

        return features

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self
