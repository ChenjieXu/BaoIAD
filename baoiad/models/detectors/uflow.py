"""U-Flow: A U-shaped Normalizing Flow for Anomaly Detection (WACV 2023).

Faithful reimplementation with:
- ResNet-18 backbone with LayerNorm feature extraction (layer1, layer2, layer3)
- U-shaped normalizing flow: split → output + upsample → concat with next scale
- AllInOneBlock from anomalib (GLOW-style coupling with global affine + permutation)
- Alternating 3x3 / 1x1 convolution subnets in coupling layers
- Anomaly map: 1 - mean(exp(-0.5 * mean(z^2, dim=C)))
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
from FrEIA import framework as ff
from FrEIA import modules as fm
from FrEIA.modules import InvertibleModule

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.utils.uflow_nfa import compute_nfa_anomaly_score_tree
from baoiad.models.base_ad_model import FlowBasedADModel


# ─── AllInOneBlock (from anomalib, self-contained) ─────────────────────────

def _global_scale_softplus_activation(s):
    return 0.1 * nn.Softplus(beta=0.5)(s)


class AllInOneBlock(InvertibleModule):
    """GLOW-style coupling block with global affine transform and permutation.

    Faithfully ported from anomalib's AllInOneBlock.
    """

    def __init__(
        self,
        dims_in,
        dims_c=None,
        subnet_constructor=None,
        affine_clamping=2.0,
        gin_block=False,
        global_affine_init=1.0,
        global_affine_type='SOFTPLUS',
        permute_soft=False,
        learned_householder_permutation=0,
        reverse_permutation=False,
    ):
        if dims_c is None:
            dims_c = []
        super().__init__(dims_in, dims_c)

        channels = dims_in[0][0]
        self.input_rank = len(dims_in[0]) - 1
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        self.permute_function = {0: F.linear, 1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[self.input_rank]
        self.in_channels = channels
        self.clamp = affine_clamping
        self.GIN = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder = learned_householder_permutation

        # Global affine
        global_scale = 2.0 * torch.log(torch.exp(torch.tensor(0.5 * 10.0 * global_affine_init)) - 1)
        self.global_scale_activation = _global_scale_softplus_activation
        self.global_scale = nn.Parameter(torch.ones(1, channels, *([1] * self.input_rank)) * global_scale)
        self.global_offset = nn.Parameter(torch.zeros(1, channels, *([1] * self.input_rank)))

        # Permutation
        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            indices = torch.randperm(channels)
            w = torch.zeros((channels, channels))
            w[torch.arange(channels), indices] = 1.0

        if self.householder:
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
        else:
            self.w_perm = nn.Parameter(
                torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)),
                requires_grad=False,
            )
            self.w_perm_inv = nn.Parameter(
                torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)),
                requires_grad=False,
            )

        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])

    def _construct_householder_permutation(self):
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).to(w.device) - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))
        for _ in range(self.input_rank):
            w = w.unsqueeze(-1)
        return w

    def _permute(self, x, rev=False):
        if self.GIN:
            scale = 1.0
            perm_log_jac = 0.0
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = torch.sum(torch.log(scale))

        if rev:
            return (self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale, perm_log_jac
        return self.permute_function(x * scale + self.global_offset, self.w_perm), perm_log_jac

    def _pre_permute(self, x, rev=False):
        if rev:
            return self.permute_function(x, self.w_perm)
        return self.permute_function(x, self.w_perm_inv)

    def _affine(self, x, a, rev=False):
        a *= 0.1
        ch = x.shape[1]
        sub_jac = self.clamp * torch.tanh(a[:, :ch])
        if self.GIN:
            sub_jac -= torch.mean(sub_jac, dim=self.sum_dims, keepdim=True)
        if not rev:
            return x * torch.exp(sub_jac) + a[:, ch:], torch.sum(sub_jac, dim=self.sum_dims)
        return (x - a[:, ch:]) * torch.exp(-sub_jac), -torch.sum(sub_jac, dim=self.sum_dims)

    def forward(self, x, c=None, rev=False, jac=True):
        if c is None:
            c = []
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)

        x1, x2 = torch.split(x[0], self.splits, dim=1)
        x1c = torch.cat([x1, *c], 1) if self.conditional else x1

        if not rev:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1)
        else:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1) ** rev * n_pixels * global_scaling_jac

        return (x_out,), log_jac_det

    @staticmethod
    def output_dims(input_dims):
        return input_dims


# ─── Affine Coupling Subnet ───────────────────────────────────────────────

class AffineCouplingSubnet:
    """Subnet constructor for affine coupling layers."""

    def __init__(self, kernel_size, subnet_channels_ratio=1.0):
        self.kernel_size = kernel_size
        self.subnet_channels_ratio = subnet_channels_ratio

    def __call__(self, in_channels, out_channels):
        mid_channels = int(in_channels * self.subnet_channels_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, self.kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, self.kernel_size, padding="same"),
        )


# ─── LayerNorm Feature Extractor ──────────────────────────────────────────

class LayerNormFeatureExtractor(nn.Module):
    """ResNet feature extractor with LayerNorm on each scale."""

    def __init__(self, backbone_name, input_size, layers=('layer1', 'layer2', 'layer3')):
        super().__init__()
        self.backbone = MODELS.build(dict(
            type='TIMMBackbone', model_name=backbone_name,
            pretrained=True, features_only=True, out_indices=(1, 2, 3), frozen=True,
        ))
        self.channels = self.backbone.out_channels
        self.scale_factors = self.backbone.reduction
        self.scales = range(len(self.scale_factors))

        self.feature_normalizations = nn.ModuleList()
        for ch, sf in zip(self.channels, self.scale_factors):
            self.feature_normalizations.append(
                nn.LayerNorm([ch, input_size[0] // sf, input_size[1] // sf], elementwise_affine=True)
            )

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, img):
        self.backbone.eval()
        features = self.backbone(img)
        return [self.feature_normalizations[i](f) for i, f in enumerate(features)]


class CaitFeatureExtractor(nn.Module):
    """MCait feature extractor: dual CaiT models at two scales (anomalib-compatible).

    Uses CaiT-M48 at 448x448 and CaiT-S24 at 224x224 to produce 2-scale features.
    Ported from anomalib's UFlow implementation.
    """

    def __init__(self, input_size=448):
        super().__init__()
        import timm
        self.input_size = input_size
        overlays = resolve_cait_pretrained_overlays()
        self.extractor1 = timm.create_model(
            'cait_m48_448',
            pretrained=True,
            pretrained_cfg_overlay=overlays.get('cait_m48_448'),
        )
        self.extractor2 = timm.create_model(
            'cait_s24_224',
            pretrained=True,
            pretrained_cfg_overlay=overlays.get('cait_s24_224'),
        )
        self.channels = [768, 384]
        self.scale_factors = [16, 32]
        self.scales = range(len(self.scale_factors))

        for param in self.extractor1.parameters():
            param.requires_grad = False
        for param in self.extractor2.parameters():
            param.requires_grad = False

    def forward(self, img):
        features = self._extract_features(img)
        return self._normalize_features(features)

    def _extract_features(self, img):
        self.extractor1.eval()
        self.extractor2.eval()

        # Scale 1: CaiT-M48 at 448x448, block index 40
        x1 = self.extractor1.patch_embed(img)
        x1 = x1 + self.extractor1.pos_embed
        x1 = self.extractor1.pos_drop(x1)
        for i in range(41):
            x1 = self.extractor1.blocks[i](x1)

        # Scale 2: CaiT-S24 at 224x224, block index 20
        img_sub = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=True)
        x2 = self.extractor2.patch_embed(img_sub)
        x2 = x2 + self.extractor2.pos_embed
        x2 = self.extractor2.pos_drop(x2)
        for i in range(21):
            x2 = self.extractor2.blocks[i](x2)

        return (x1, x2)

    def _normalize_features(self, features):
        normalized = []
        for i, extractor in enumerate([self.extractor1, self.extractor2]):
            batch, _, channels = features[i].shape
            sf = self.scale_factors[i]
            x = extractor.norm(features[i].contiguous())
            x = x.permute(0, 2, 1)
            x = x.reshape(batch, channels, self.input_size // sf, self.input_size // sf)
            normalized.append(x)
        return normalized


def resolve_cait_pretrained_overlays(cache_dir: str | None = None) -> dict[str, dict[str, str | None]]:
    """Prefer local CaiT checkpoints over hf-hub when they are cached.

    timm's default source resolution prefers ``hf_hub_id`` over ``url`` when
    both are present. Under ``HF_HUB_OFFLINE=1`` that fails even if the
    original `.pth` files are already cached in the torch hub checkpoints
    directory. UFlow strict alignment uses the original Facebook `.pth`
    weights, so when those files exist locally we force timm to load from file.
    """
    if cache_dir is None:
        cache_dir = os.path.join(torch.hub.get_dir(), 'checkpoints')

    local_files = {
        'cait_m48_448': 'M48_448.pth',
        'cait_s24_224': 'S24_224.pth',
    }
    overlays = {}
    for model_name, filename in local_files.items():
        cached_path = os.path.join(cache_dir, filename)
        if not os.path.exists(cached_path):
            continue
        overlays[model_name] = {
            'file': cached_path,
            'hf_hub_id': None,
            'source': 'file',
        }
    return overlays


# ─── U-Flow Detector ──────────────────────────────────────────────────────

@MODELS.register_module()
class UFlowDetector(FlowBasedADModel):
    """U-Flow: U-shaped Normalizing Flow for anomaly detection.

    Uses a ResNet-18 backbone with LayerNorm feature extraction and
    a U-shaped normalizing flow architecture (split + upsample + concat).

    Args:
        input_size (tuple): Input image size (H, W). Default (448, 448).
        flow_steps (int): Number of flow steps per stage. Default 4.
        backbone (str): Backbone name. Default 'resnet18'.
        affine_clamp (float): Clamping for affine coupling. Default 2.0.
        affine_subnet_channels_ratio (float): Channel ratio in subnets. Default 1.0.
        permute_soft (bool): Use soft permutation. Default False.
    """

    def __init__(
        self,
        input_size=(256, 256),
        flow_steps=4,
        backbone='resnet18',
        affine_clamp=2.0,
        affine_subnet_channels_ratio=1.0,
        permute_soft=False,
        compute_nfa_in_predict=False,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.input_size = input_size
        self.affine_clamp = affine_clamp
        self.affine_subnet_channels_ratio = affine_subnet_channels_ratio
        self.permute_soft = permute_soft
        self.compute_nfa_in_predict = bool(compute_nfa_in_predict)

        # backbone may be a ConfigDict from config inheritance; extract name string
        if isinstance(backbone, dict):
            backbone = backbone.get('backbone_name', backbone.get('model_name', 'resnet18'))
        if backbone == 'mcait':
            self.feature_extractor = CaitFeatureExtractor(input_size=input_size[0])
        else:
            self.feature_extractor = LayerNormFeatureExtractor(backbone, input_size)
        self.flow = self._build_flow(flow_steps)

    def _build_flow(self, flow_steps):
        """Build U-shaped normalizing flow."""
        channels = self.feature_extractor.channels
        scale_factors = self.feature_extractor.scale_factors

        input_nodes = []
        for ch, sf in zip(channels, scale_factors):
            input_nodes.append(
                ff.InputNode(ch, self.input_size[0] // sf, self.input_size[1] // sf, name=f"cond_{ch}")
            )

        nodes, output_nodes = [], []
        last_node = input_nodes[-1]

        for i in reversed(range(1, len(input_nodes))):
            flows = self._build_flow_stage(last_node, flow_steps)
            volume_size = flows[-1].output_dims[0][0]
            split = ff.Node(
                flows[-1],
                fm.Split,
                {"section_sizes": (volume_size // 8 * 4, volume_size - volume_size // 8 * 4), "dim": 0},
                name=f"split_{i + 1}",
            )
            output = ff.OutputNode(split.out1, name=f"output_scale_{i + 1}")
            up = ff.Node(split.out0, fm.IRevNetUpsampling, {}, name=f"up_{i + 1}")
            last_node = ff.Node(
                [input_nodes[i - 1].out0, up.out0], fm.Concat, {"dim": 0}, name=f"cat_{i}"
            )
            output_nodes.append(output)
            nodes.extend([*flows, split, up, last_node])

        flows = self._build_flow_stage(last_node, flow_steps)
        output = ff.OutputNode(flows[-1], name="output_scale_1")
        output_nodes.append(output)
        nodes.extend(flows)

        return ff.GraphINN(input_nodes + nodes + output_nodes[::-1])

    def _build_flow_stage(self, in_node, flow_steps):
        """Build a single flow stage with alternating kernel sizes."""
        flow_size = in_node.output_dims[0][-1]
        nodes = []
        for step in range(flow_steps):
            nodes.append(
                ff.Node(
                    in_node,
                    AllInOneBlock,
                    module_args={
                        "subnet_constructor": AffineCouplingSubnet(
                            3 if step % 2 == 0 else 1,
                            self.affine_subnet_channels_ratio,
                        ),
                        "affine_clamping": self.affine_clamp,
                        "permute_soft": self.permute_soft,
                    },
                    name=f"flow{flow_size}_step{step}",
                )
            )
            in_node = nodes[-1]
        return nodes

    @staticmethod
    def _compute_loss(hidden_variables, jacobians):
        lpz = torch.sum(
            torch.stack([0.5 * torch.sum(z_i ** 2, dim=(1, 2, 3)) for z_i in hidden_variables], dim=0),
            dim=0,
        )
        return torch.mean(lpz - jacobians)

    def get_probability(self, outputs, resize_size=None):
        """Return the mean likelihood map used by the official UFlow code."""
        if resize_size is None:
            resize_size = self.input_size

        probabilities = []
        for output in outputs:
            log_prob = -torch.mean(output ** 2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            probabilities.append(
                F.interpolate(
                    prob,
                    size=resize_size,
                    mode='bilinear',
                    align_corners=False,
                ))
        return torch.mean(torch.stack(probabilities, dim=-1), dim=-1)

    def compute_likelihood_anomaly_map(self, outputs, resize_size=None):
        return 1 - self.get_probability(outputs, resize_size)

    def compute_nfa_anomaly_map(self, outputs, resize_size=None):
        if resize_size is None:
            resize_size = self.input_size
        return compute_nfa_anomaly_score_tree(outputs, target_size=resize_size)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        features = self.feature_extractor(inputs)
        z, ljd = self.flow(features, rev=False)
        if not isinstance(z, (list, tuple)):
            z = [z]

        if mode == 'loss':
            return {'loss': self._compute_loss(z, ljd)}

        elif mode == 'predict':
            anomaly_map = self.compute_likelihood_anomaly_map(z, self.input_size)
            img_scores = torch.amax(anomaly_map, dim=(-2, -1))
            results = build_predict_results(data_samples, img_scores, anomaly_map)
            if self.compute_nfa_in_predict:
                nfa_map = self.compute_nfa_anomaly_map(z, self.input_size)
                nfa_scores = torch.amax(nfa_map, dim=(-2, -1))
                for index, sample in enumerate(results):
                    sample.pred_nfa_score = float(nfa_scores[index].item())
                    sample.pred_nfa_anomaly_map = nfa_map[index].detach().cpu()
            return results

        return z, ljd

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self.feature_extractor, 'backbone'):
            self.feature_extractor.backbone.eval()
        else:
            self.feature_extractor.eval()
        return self
