"""FastFlow anomaly detector — aligned with anomalib using FrEIA flows."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework
import FrEIA.modules

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import FlowBasedADModel


@MODELS.register_module(force=True)
class FastFlowDetector(FlowBasedADModel):
    """FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.

    Uses FrEIA SequenceINN + AllInOneBlock with LayerNorm on backbone features,
    aligned with the anomalib implementation.
    """

    def __init__(self, backbone=None, flow_steps=8, conv3x3_only=False,
                 hidden_ratio=1.0, clamp=2.0, input_size=(256, 256),
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build backbone
        if backbone is None:
            backbone = dict(
                type='TIMMBackbone',
                model_name='wide_resnet50_2',
                features_only=True,
                out_indices=(1, 2, 3),
                frozen=True,
            )
        if isinstance(backbone, dict):
            backbone.setdefault('frozen', True)
            self.backbone = MODELS.build(backbone)
        else:
            self.backbone = MODELS.build(dict(
                type='TIMMBackbone',
                model_name=backbone,
                features_only=True,
                out_indices=(1, 2, 3),
                frozen=True,
            ))

        # Determine channel dimensions and spatial scales from backbone
        channels = self.backbone.out_channels
        reductions = self.backbone.reduction

        # LayerNorm on each feature level (trainable)
        self.norms = nn.ModuleList()
        for ch, red in zip(channels, reductions):
            h = input_size[0] // red
            w = input_size[1] // red
            self.norms.append(nn.LayerNorm([ch, h, w], elementwise_affine=True))

        # Build flow blocks using FrEIA SequenceINN + AllInOneBlock
        self.flows = nn.ModuleList()
        for ch, red in zip(channels, reductions):
            h = input_size[0] // red
            w = input_size[1] // red
            self.flows.append(
                self._create_flow_block(ch, h, w, flow_steps, conv3x3_only, hidden_ratio, clamp)
            )

    @staticmethod
    def _subnet_conv_func(kernel_size, hidden_ratio):
        def subnet_conv(in_channels, out_channels):
            hidden = int(in_channels * hidden_ratio)
            # Match anomalib: use ZeroPad2d + Conv2d without padding
            # This ensures consistent boundary handling
            padding_dims = (kernel_size // 2 - ((1 + kernel_size) % 2), kernel_size // 2)
            padding = (*padding_dims, *padding_dims)
            return nn.Sequential(
                nn.ZeroPad2d(padding),
                nn.Conv2d(in_channels, hidden, kernel_size),
                nn.ReLU(),
                nn.ZeroPad2d(padding),
                nn.Conv2d(hidden, out_channels, kernel_size),
            )
        return subnet_conv

    def _create_flow_block(self, channels, h, w, flow_steps, conv3x3_only, hidden_ratio, clamp):
        nodes = FrEIA.framework.SequenceINN(channels, h, w)
        for i in range(flow_steps):
            # Match anomalib: use 1x1 conv on odd steps when not conv3x3_only, otherwise 3x3
            kernel_size = 1 if (i % 2 == 1 and not conv3x3_only) else 3
            nodes.append(
                FrEIA.modules.AllInOneBlock,
                subnet_constructor=self._subnet_conv_func(kernel_size, hidden_ratio),
                affine_clamping=clamp,
                permute_soft=False,
            )
        return nodes

    @torch.no_grad()
    def extract_features(self, x):
        features = self.backbone(x)
        return [self.norms[i](feat) for i, feat in enumerate(features)]

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats = self.extract_features(inputs)
        B = inputs.shape[0]

        if mode == 'loss':
            loss = 0
            for flow, feat in zip(self.flows, feats):
                z, log_jac_det = flow(feat)
                loss += torch.mean(0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) - log_jac_det)
            return {'loss': loss}

        elif mode == 'predict':
            input_size = inputs.shape[-2:]
            flow_maps = []
            for flow, feat in zip(self.flows, feats):
                z, _ = flow(feat)
                log_prob = -torch.mean(z ** 2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                flow_map = F.interpolate(-prob, size=input_size, mode='bilinear', align_corners=False)
                flow_maps.append(flow_map)
            anomaly_map = torch.mean(torch.stack(flow_maps, dim=-1), dim=-1).squeeze(1)
            img_scores = anomaly_map.view(B, -1).max(dim=1).values

            return build_predict_results(data_samples, img_scores, anomaly_map)

        return feats

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
