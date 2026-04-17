"""CFlow anomaly detector (WACV 2022).

Faithful reimplementation using FrEIA conditional normalizing flows with:
- 3-scale features: layer2 + layer3 + layer4 from WideResNet-50-2
- 8 coupling blocks (AllInOneBlock with SOFTPLUS global affine)
- condition_dim=128 positional encoding
- Fiber batching for memory efficiency
- Per-fiber-batch gradient updates (matching anomalib)
"""
import os
from pathlib import Path
from typing import Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import FlowBasedADModel

try:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
    HAS_FREIA = True
except ImportError:
    Ff = None
    Fm = None
    HAS_FREIA = False


_GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))
_SOFT_PERM_CACHE = {}


def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5 * torch.sum(z ** 2, 1) + logdet_J
    return logp


def positionalencoding2d(D, H, W):
    """2D sinusoidal positional encoding. D must be divisible by 4."""
    if D % 4 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with dim={D}")
    P = torch.zeros(D, H, W)
    D2 = D // 2
    div_term = torch.exp(torch.arange(0.0, D2, 2) * -(math.log(1e4) / D2))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D2:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, H, 1)
    P[1:D2:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, H, 1)
    P[D2::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, W)
    P[D2 + 1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 2 * dims_in),
        nn.ReLU(),
        nn.Linear(2 * dims_in, dims_out),
    )


def _cflow_soft_perm_cache_dir() -> Path:
    cache_root = os.environ.get(
        'BAOIAD_CFLOW_PERM_CACHE_DIR',
        os.path.join(Path.home(), '.cache', 'baoiad', 'cflow_soft_permutations'),
    )
    path = Path(cache_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _soft_perm_seed(n_feat: int, block_idx: int) -> int:
    return 1729 + 10007 * int(n_feat) + 97 * int(block_idx)


def _generate_soft_permutation_matrix(n_feat: int, block_idx: int) -> torch.Tensor:
    key = (int(n_feat), int(block_idx))
    cached = _SOFT_PERM_CACHE.get(key)
    if cached is not None:
        return cached

    cache_path = _cflow_soft_perm_cache_dir() / f'{n_feat}_{block_idx}.pt'
    if cache_path.is_file():
        matrix = torch.load(cache_path, map_location='cpu')
        _SOFT_PERM_CACHE[key] = matrix
        return matrix

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = torch.Generator(device=device)
    generator.manual_seed(_soft_perm_seed(n_feat, block_idx))

    with torch.no_grad():
        random_matrix = torch.randn(
            n_feat,
            n_feat,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        q, r = torch.linalg.qr(random_matrix, mode='reduced')
        signs = torch.sign(torch.diag(r))
        signs[signs == 0] = 1
        q = q * signs.unsqueeze(0)
        matrix = q.to(device='cpu', dtype=torch.float32).contiguous()

    tmp_path = cache_path.with_suffix('.tmp')
    torch.save(matrix, tmp_path)
    os.replace(tmp_path, cache_path)
    _SOFT_PERM_CACHE[key] = matrix
    return matrix


def build_cflow_head(n_feat, condition_dim, coupling_blocks, clamp_alpha,
                     permute_soft=False):
    """Build a conditional normalizing flow head using FrEIA."""
    if not HAS_FREIA:
        return _FallbackCFlowHead(n_feat, condition_dim)
    coder = Ff.SequenceINN(n_feat)
    for block_idx in range(coupling_blocks):
        coder.append(
            Fm.AllInOneBlock,
            cond=0,
            cond_shape=(condition_dim,),
            subnet_constructor=subnet_fc,
            affine_clamping=clamp_alpha,
            global_affine_type='SOFTPLUS',
            permute_soft=False,
        )
        if permute_soft:
            block = coder[-1]
            matrix = _generate_soft_permutation_matrix(n_feat, block_idx)
            block.w_perm.data.copy_(matrix.view_as(block.w_perm))
            block.w_perm_inv.data.copy_(matrix.t().contiguous().view_as(block.w_perm_inv))
    return coder


class _FallbackCFlowHead(nn.Module):
    """Lightweight fallback used when FrEIA is unavailable.

    This preserves the CFlow interface for tests and basic execution paths.
    The real FrEIA implementation is still used whenever the dependency exists.
    """

    def __init__(self, n_feat, condition_dim):
        super().__init__()
        hidden = max(n_feat, condition_dim)
        self.net = nn.Sequential(
            nn.Linear(n_feat + condition_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_feat),
        )

    def forward(self, x, conds):
        cond = conds[0]
        z = self.net(torch.cat([x, cond], dim=1))
        log_jac_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return z, log_jac_det


@MODELS.register_module()
class CFlowDetector(FlowBasedADModel):
    """CFlow: Conditional Normalizing Flow for anomaly detection.

    Uses layer2 + layer3 + layer4 of WideResNet-50-2 as multi-scale features,
    each with a conditional normalizing flow decoder (FrEIA).

    Training uses per-fiber-batch gradient updates matching anomalib.
    """

    def __init__(
        self,
        backbone: Union[str, dict] = 'wide_resnet50_2',
        condition_dim=128,
        coupling_blocks=8,
        clamp_alpha=1.9,
        permute_soft=False,
        fiber_batch_size=64,
        reference_repo: str = '',
        require_official_reference: bool = False,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.reference_repo = reference_repo
        self.require_official_reference = require_official_reference
        if self.require_official_reference:
            if not self.reference_repo:
                raise FileNotFoundError('CFlow strict mode requires `reference_repo` to be set.')
            if not os.path.isdir(self.reference_repo):
                raise FileNotFoundError(
                    f'CFlow official reference repo not found: {self.reference_repo}'
                )

        if isinstance(backbone, dict) and backbone.get('type') == 'TIMMBackbone':
            backbone_cfg = backbone.copy()
            backbone_cfg.setdefault('out_indices', (1, 2, 3))
            backbone_cfg.setdefault('pretrained', True)
            backbone_cfg.setdefault('frozen', True)
            self.backbone = MODELS.build(backbone_cfg)
        else:
            from baoiad.models.backbone_utils import build_feature_extractor
            self.backbone = build_feature_extractor(
                backbone, default_out_indices=(2, 3, 4), default_frozen=True)

        ch = self.backbone.out_channels
        pool_dims = list(ch)
        self.flows = nn.ModuleList([
            build_cflow_head(dim, condition_dim, coupling_blocks, clamp_alpha,
                             permute_soft)
            for dim in pool_dims
        ])

        self.condition_dim = condition_dim
        self.fiber_batch_size = fiber_batch_size
        self._pe_cache = {}

    @torch.no_grad()
    def extract_features(self, x):
        feats = self.backbone(x)
        return list(feats)

    def _get_positional_encoding(self, H, W, device, dtype):
        key = (H, W, str(device), str(dtype))
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = positionalencoding2d(self.condition_dim, H, W).to(device=device, dtype=dtype)
            self._pe_cache[key] = pe
        return pe

    def _prepare_fiber_data(self, feat, B):
        """Prepare positional encoding and flatten features for fiber batching."""
        _, C, H, W = feat.shape
        S = H * W
        E = B * S

        pe = self._get_positional_encoding(H, W, feat.device, feat.dtype)
        pe = pe.unsqueeze(0).expand(B, -1, -1, -1)
        c_r = pe.reshape(B, self.condition_dim, S).permute(0, 2, 1).reshape(E, self.condition_dim)
        e_r = feat.reshape(B, C, S).permute(0, 2, 1).reshape(E, C)
        return c_r, e_r, C, H, W, E

    def train_step(self, data, optim_wrapper):
        """Per-fiber-batch training matching anomalib's manual optimization.

        Each fiber batch gets its own zero_grad -> backward -> step cycle,
        rather than accumulating loss across all fibers.
        """
        data = self.data_preprocessor(data, True)
        inputs = data['inputs']
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats = self.extract_features(inputs)
        B = inputs.shape[0]

        total_loss = 0.0
        num_fibers = 0

        for feat, flow in zip(feats, self.flows):
            c_r, e_r, C, H, W, E = self._prepare_fiber_data(feat, B)

            perm = torch.randperm(E, device=feat.device)
            N = self.fiber_batch_size
            FIB = E // N  # drop remainder like anomalib

            for f in range(FIB):
                idx = perm[f * N:(f + 1) * N]
                with optim_wrapper.optim_context(self):
                    z, log_jac_det = flow(e_r[idx], [c_r[idx]])
                    log_prob = get_logp(C, z, log_jac_det)
                    loss = -F.logsigmoid(log_prob / C).mean()
                optim_wrapper.update_params(loss)
                total_loss += float(loss.detach().item())
                num_fibers += 1

        avg_loss = total_loss / max(num_fibers, 1)
        return {'loss': torch.tensor(avg_loss, device=inputs.device)}

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        feats = self.extract_features(inputs)
        B = inputs.shape[0]

        if mode == 'loss':
            # Fallback loss (not used during training due to train_step override)
            total_loss = 0.0
            for feat, flow in zip(feats, self.flows):
                c_r, e_r, C, H, W, E = self._prepare_fiber_data(feat, B)
                perm = torch.randperm(E, device=feat.device)
                N = self.fiber_batch_size
                FIB = E // N
                for f in range(FIB):
                    idx = perm[f * N:(f + 1) * N]
                    z, log_jac_det = flow(e_r[idx], [c_r[idx]])
                    log_prob = get_logp(C, z, log_jac_det)
                    loss = -F.logsigmoid(log_prob / C).mean()
                    total_loss = total_loss + loss
            return {'loss': total_loss}

        elif mode == 'predict':
            layer_maps = []
            for feat, flow in zip(feats, self.flows):
                c_r, e_r, C, H, W, E = self._prepare_fiber_data(feat, B)

                N = self.fiber_batch_size
                FIB = E // N + int(E % N > 0)
                log_probs = []
                for f in range(FIB):
                    start = f * N
                    end = min((f + 1) * N, E)
                    z, log_jac_det = flow(e_r[start:end], [c_r[start:end]])
                    log_probs.append(get_logp(C, z, log_jac_det))

                log_prob = torch.cat(log_probs)
                log_prob = log_prob / C
                # Match anomalib: convert likelihoods to relative probabilities
                # before upsampling and invert the aggregated score map.
                layer_prob = torch.exp(log_prob - log_prob.max())
                layer_map = layer_prob.reshape(B, H, W)
                layer_map = F.interpolate(
                    layer_map.unsqueeze(1), size=inputs.shape[-2:],
                    mode='bilinear', align_corners=True,
                ).squeeze(1)
                layer_maps.append(layer_map)

            score_map = sum(layer_maps)
            score_map = score_map.max() - score_map
            img_scores = score_map.view(B, -1).max(dim=1).values

            return build_predict_results(data_samples, img_scores, score_map)

        return feats

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
