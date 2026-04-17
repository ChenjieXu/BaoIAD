"""AnoVL detector — zero-shot anomaly localization via V-V attention surgery on CLIP.

Reference: Deng et al., "AnoVL: Adapting Vision-Language Models for Unified
Zero-shot Anomaly Localization", AAAI 2024.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import VisionLanguageADModel

# ---------------------------------------------------------------------------
# Domain-aware contrastive state prompting
# ---------------------------------------------------------------------------

NORMAL_STATES = [
    "normal {}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]

ABNORMAL_STATES = [
    "damaged {}",
    "abnormal {}",
    "imperfect {}",
    "blemished {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

# Industrial templates (for all categories)
INDUSTRIAL_TEMPLATES = [
    "a cropped industrial photo of the {}",
    "a cropped industrial photo of a {}",
    "a close-up industrial photo of a {}",
    "a close-up industrial photo of the {}",
    "a bright industrial photo of a {}",
    "a bright industrial photo of the {}",
    "a dark industrial photo of the {}",
    "a dark industrial photo of a {}",
    "a jpeg corrupted industrial photo of a {}",
    "a jpeg corrupted industrial photo of the {}",
    "a blurry industrial photo of the {}",
    "a blurry industrial photo of a {}",
    "an industrial photo of a {}",
    "an industrial photo of the {}",
    "an industrial photo of a small {}",
    "an industrial photo of the small {}",
    "an industrial photo of a large {}",
    "an industrial photo of the large {}",
    "an industrial photo of the {} for visual inspection",
    "an industrial photo of a {} for visual inspection",
    "an industrial photo of the {} for anomaly detection",
    "an industrial photo of a {} for anomaly detection",
]

# Object-specific templates (image content)
OBJECT_IMAGE_TEMPLATES = [
    "a cropped industrial image of the {}",
    "a cropped industrial image of a {}",
    "a close-up industrial image of a {}",
    "a close-up industrial image of the {}",
    "a bright industrial image of a {}",
    "a bright industrial image of the {}",
    "a dark industrial image of the {}",
    "a dark industrial image of a {}",
    "a jpeg corrupted industrial image of a {}",
    "a jpeg corrupted industrial image of the {}",
    "a blurry industrial image of the {}",
    "a blurry industrial image of a {}",
    "an industrial image of a {}",
    "an industrial image of the {}",
    "an industrial image of a small {}",
    "an industrial image of the small {}",
    "an industrial image of a large {}",
    "an industrial image of the large {}",
    "an industrial image of the {} for visual inspection",
    "an industrial image of a {} for visual inspection",
    "an industrial image of the {} for anomaly detection",
    "an industrial image of a {} for anomaly detection",
]

# Object manufacturing templates
OBJECT_MANUFACTURING_TEMPLATES = [
    "a cropped manufacturing image of the {}",
    "a cropped manufacturing image of a {}",
    "a close-up manufacturing image of a {}",
    "a close-up manufacturing image of the {}",
    "a bright manufacturing image of a {}",
    "a bright manufacturing image of the {}",
    "a dark manufacturing image of the {}",
    "a dark manufacturing image of a {}",
    "a jpeg corrupted manufacturing image of a {}",
    "a jpeg corrupted manufacturing image of the {}",
    "a blurry manufacturing image of the {}",
    "a blurry manufacturing image of a {}",
    "a manufacturing image of a {}",
    "a manufacturing image of the {}",
    "a manufacturing image of a small {}",
    "a manufacturing image of the small {}",
    "a manufacturing image of a large {}",
    "a manufacturing image of the large {}",
    "a manufacturing image of the {} for visual inspection",
    "a manufacturing image of a {} for visual inspection",
    "a manufacturing image of the {} for anomaly detection",
    "a manufacturing image of a {} for anomaly detection",
]

# Texture-specific templates (surface pictures)
TEXTURE_SURFACE_TEMPLATES = [
    "a cropped surface picture of the {}",
    "a cropped surface picture of a {}",
    "a close-up surface picture of a {}",
    "a close-up surface picture of the {}",
    "a bright surface picture of a {}",
    "a bright surface picture of the {}",
    "a dark surface picture of the {}",
    "a dark surface picture of a {}",
    "a jpeg corrupted surface picture of a {}",
    "a jpeg corrupted surface picture of the {}",
    "a blurry surface picture of the {}",
    "a blurry surface picture of a {}",
    "a surface picture of a {}",
    "a surface picture of the {}",
    "a surface picture of a small {}",
    "a surface picture of the small {}",
    "a surface picture of a large {}",
    "a surface picture of the large {}",
    "a surface picture of the {} for visual inspection",
    "a surface picture of a {} for visual inspection",
    "a surface picture of the {} for anomaly detection",
    "a surface picture of a {} for anomaly detection",
]

# Texture textural photo templates
TEXTURE_TEXT_TEMPLATES = [
    "a cropped textural photo of the {}",
    "a cropped textural photo of a {}",
    "a close-up textural photo of a {}",
    "a close-up textural photo of the {}",
    "a bright textural photo of a {}",
    "a bright textural photo of the {}",
    "a dark textural photo of the {}",
    "a dark textural photo of a {}",
    "a jpeg corrupted textural photo of a {}",
    "a jpeg corrupted textural photo of the {}",
    "a blurry textural photo of the {}",
    "a blurry textural photo of a {}",
    "a textural photo of a {}",
    "a textural photo of the {}",
    "a textural photo of a small {}",
    "a textural photo of the small {}",
    "a textural photo of a large {}",
    "a textural photo of the large {}",
    "a textural photo of the {} for visual inspection",
    "a textural photo of a {} for visual inspection",
    "a textural photo of the {} for anomaly detection",
    "a textural photo of a {} for anomaly detection",
]

# MVTec texture categories (used for template selection)
TEXTURE_CATEGORIES = frozenset({
    "carpet", "grid", "leather", "tile", "wood",
})


from baoiad.utils.score_utils import normalize_class_name as _normalize_class_name


def _is_texture(class_name):
    return class_name.lower().replace(" ", "_") in TEXTURE_CATEGORIES


def _build_prompts_for_class(class_name):
    """Build domain-aware prompts: 3 template groups x 22 templates x 7 states."""
    class_name = _normalize_class_name(class_name)
    is_texture = _is_texture(class_name)

    if is_texture:
        template_groups = [
            INDUSTRIAL_TEMPLATES,
            TEXTURE_TEXT_TEMPLATES,
            TEXTURE_SURFACE_TEMPLATES,
        ]
    else:
        template_groups = [
            INDUSTRIAL_TEMPLATES,
            OBJECT_IMAGE_TEMPLATES,
            OBJECT_MANUFACTURING_TEMPLATES,
        ]

    all_normal = []
    all_abnormal = []
    for templates in template_groups:
        for state in NORMAL_STATES:
            for template in templates:
                all_normal.append(template.format(state.format(class_name)))
        for state in ABNORMAL_STATES:
            for template in templates:
                all_abnormal.append(template.format(state.format(class_name)))

    return all_normal, all_abnormal


# ---------------------------------------------------------------------------
# V-V Attention Surgery modules
# ---------------------------------------------------------------------------

class VVAttention(nn.Module):
    """Multi-head attention with dual-path: standard Q-K and V-V surgery.

    Replaces Q and K with V in the attention computation to extract
    local-aware patch tokens.
    """

    def __init__(self, original_attn):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.embed_dim = original_attn.embed_dim
        self.head_dim = self.embed_dim // self.num_heads

        # Copy QKV weights from original attention
        self.in_proj_weight = nn.Parameter(
            original_attn.in_proj_weight.clone())
        self.in_proj_bias = nn.Parameter(
            original_attn.in_proj_bias.clone()) if original_attn.in_proj_bias is not None else None
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj.weight = nn.Parameter(
            original_attn.out_proj.weight.clone())
        self.out_proj.bias = nn.Parameter(
            original_attn.out_proj.bias.clone())

    def _split_qkv(self, x):
        """Split input into Q, K, V using in_proj weights."""
        w = self.in_proj_weight
        b = self.in_proj_bias
        d = self.embed_dim
        q = F.linear(x, w[:d], b[:d] if b is not None else None)
        k = F.linear(x, w[d:2*d], b[d:2*d] if b is not None else None)
        v = F.linear(x, w[2*d:], b[2*d:] if b is not None else None)
        return q, k, v

    def _mha(self, q, k, v):
        """Standard multi-head attention."""
        L, B, _ = q.shape
        q = q.reshape(L, B * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(L, B * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(L, B * self.num_heads, self.head_dim).transpose(0, 1)
        scale = self.head_dim ** -0.5
        attn = torch.bmm(q * scale, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v)
        out = out.transpose(0, 1).reshape(L, B, self.embed_dim)
        return self.out_proj(out)

    def forward(self, x):
        """Compute both V-V surgery and standard attention from same input.

        Both paths share the same QKV decomposition (from x_ori in practice).
        The V-V path replaces Q and K with V for local-aware features.

        Args:
            x: Input tensor (L, B, D) — typically ln_1(x_ori).

        Returns:
            (out_vv, out_ori): V-V surgery output and standard output.
        """
        q, k, v = self._split_qkv(x)

        # Original path: standard Q-K attention
        out_ori = self._mha(q, k, v)

        # V-V path: replace Q and K with V
        out_vv = self._mha(v, v, v)

        return out_vv, out_ori


class VVResidualBlock(nn.Module):
    """Wraps an original ResidualAttentionBlock for dual-path V-V surgery.

    The V-V path skips the FFN (MLP); the original path uses FFN normally.
    """

    def __init__(self, original_block):
        super().__init__()
        self.vv_attn = VVAttention(original_block.attn)
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2
        self.mlp = original_block.mlp

    def forward(self, x):
        """Forward with dual-path processing.

        Matches the original AnoVL: both VV and original paths compute
        QKV from x_ori only.  The VV path is an accumulator that adds
        VV attention residuals without FFN.

        Args:
            x: Either a single tensor (L, B, D) for the first surgery block,
               or a list [x_vv, x_ori] each (L, B, D) for subsequent blocks.

        Returns:
            List [x_vv, x_ori], each (L, B, D).
        """
        if isinstance(x, list):
            # Continuing dual path
            x_vv, x_ori = x
        else:
            # First surgery block: start dual path from single input
            x_vv = x
            x_ori = x

        # BOTH paths compute attention from x_ori features (key insight)
        attn_vv, attn_ori = self.vv_attn(self.ln_1(x_ori))

        # V-V path: residual attention only, no FFN
        x_vv = x_vv + attn_vv

        # Original path: residual attention + FFN
        x_ori = x_ori + attn_ori
        x_ori = x_ori + self.mlp(self.ln_2(x_ori))

        return [x_vv, x_ori]


def apply_vv_surgery(visual_encoder):
    """Replace all transformer blocks with VVResidualBlock for V-V surgery."""
    transformer = visual_encoder.transformer
    new_blocks = nn.ModuleList()
    for block in transformer.resblocks:
        new_blocks.append(VVResidualBlock(block))
    transformer.resblocks = new_blocks
    return visual_encoder


# ---------------------------------------------------------------------------
# Token projection
# ---------------------------------------------------------------------------

class LinearLayer(nn.Module):
    """Project intermediate layer tokens through ln_post and visual.proj."""

    def __init__(self, ln_post, proj):
        super().__init__()
        self.ln_post = ln_post
        self.proj = proj

    def forward(self, x):
        """Project patch tokens (no CLS) through ln_post + proj.

        Args:
            x: (B, N, D) patch tokens.

        Returns:
            (B, N, D_out) projected tokens.
        """
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x


# ---------------------------------------------------------------------------
# TTA adapter
# ---------------------------------------------------------------------------

class TextAdapter(nn.Module):
    """Feature-space residual adapter for test-time adaptation.

    Matches the original AnoVL implementation: maps features to an affinity
    space via a linear layer, applies tanh, then projects back to the feature
    space via text embeddings.

    Forward: ``0.5 * x + 0.5 * adapter(x)``  (output is D-dim features).
    """

    def __init__(self, text_embeddings, noise_level=1):
        """
        Args:
            text_embeddings: (2*K, D) concatenated normal+abnormal prompt
                embeddings, where K = num prompts per state.
        """
        super().__init__()
        # ad: D -> 2*K (affinity space)
        self.ad = nn.Linear(text_embeddings.shape[1], text_embeddings.shape[0])
        self.text_embeddings = text_embeddings  # (2*K, D), for projection back
        self.noise_level = noise_level

    def _adapter(self, img):
        """Project to affinity space and back to feature space."""
        img = img / img.norm(dim=-1, keepdim=True)
        affinity = self.ad(img)              # (..., 2*K)
        affinity = torch.tanh(affinity)
        output = F.linear(affinity, self.text_embeddings.t())  # (..., D)
        return output

    def _aug(self, x):
        """Add Gaussian noise augmentation (for TTA training pass)."""
        feat_list = [x]
        for n in range(self.noise_level):
            noise = torch.normal(
                0, 0.05 * 1.1 ** (n + 1), x.shape, device=x.device,
            )
            feat_list.append(x + noise)
        return torch.cat(feat_list, dim=0)  # ((1+noise_level)*B, ...)

    def forward(self, x, is_test=False):
        """
        Args:
            x: (B, H, W, D) or (B, N, D) features.
            is_test: If False, augment with noise before adapting.

        Returns:
            Adapted features, same shape as input (or expanded batch if training).
        """
        if not is_test:
            x = self._aug(x)
        if len(x.shape) == 4:
            N, H, W, C = x.shape
            x = 0.5 * x.view(N, H * W, C) + 0.5 * self._adapter(
                x.view(N, H * W, C)
            )
            x = x.view(N, H, W, C)
        else:
            x = 0.5 * x + 0.5 * self._adapter(x)
        return x


def _tta_loss(pred):
    """TTA loss from original AnoVL.

    Args:
        pred: (1+noise_level, H, W, 2) softmax probabilities.
            pred[0] = clean sample, pred[1:] = noisy samples.

    Returns:
        Scalar loss.
    """
    # Entropy minimization on clean sample
    soft_loss = -(pred[0] * pred[0].log()).sum(-1).mean()
    # Cross-entropy: noisy→abnormal, clean→normal
    mask = torch.zeros(pred[1:].shape, device=pred.device)
    mask[..., 1] = 1  # target abnormal for noisy
    hard_loss = (-mask * pred[1:].log() - (1 - mask) * pred[0].log())
    hard_loss = hard_loss.sum(-1).mean()
    return soft_loss + 0.5 * hard_loss


def _weight_reset(m):
    """Reset Linear/Conv2d parameters."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()


# ---------------------------------------------------------------------------
# Official image augmentation for robust image-level scoring
# ---------------------------------------------------------------------------

def _affine_transform(images, matrix):
    """Apply an affine transform with the official reflection padding."""
    grid = F.affine_grid(matrix, images.size(), align_corners=True)
    return F.grid_sample(
        images,
        grid,
        padding_mode="reflection",
        align_corners=True,
    )


def _rotation_matrix(theta, batch_size, device, dtype):
    theta = torch.as_tensor(theta, device=device, dtype=dtype)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    matrix = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    matrix[:, 0, 0] = cos_theta
    matrix[:, 0, 1] = -sin_theta
    matrix[:, 1, 0] = sin_theta
    matrix[:, 1, 1] = cos_theta
    return matrix


def _translation_matrix(tx, ty, batch_size, device, dtype):
    matrix = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    matrix[:, 0, 0] = 1
    matrix[:, 1, 1] = 1
    matrix[:, 0, 2] = tx
    matrix[:, 1, 2] = ty
    return matrix


def _rotate_images(images, theta):
    matrix = _rotation_matrix(
        theta,
        batch_size=images.shape[0],
        device=images.device,
        dtype=images.dtype,
    )
    return _affine_transform(images, matrix)


def _translate_images(images, tx, ty):
    matrix = _translation_matrix(
        tx,
        ty,
        batch_size=images.shape[0],
        device=images.device,
        dtype=images.dtype,
    )
    return _affine_transform(images, matrix)


def _grayscale_images(images):
    gray = (
        0.299 * images[:, 0:1]
        + 0.587 * images[:, 1:2]
        + 0.114 * images[:, 2:3]
    )
    return gray.repeat(1, 3, 1, 1)


def _augment_views(images):
    """Generate the official AnoVL 22-view augmentation set.

    Args:
        images: (B, C, H, W) tensor.

    Returns:
        List of 22 tensors, each with shape (B, C, H, W).
    """
    views = [images]

    for angle in (
        -math.pi / 4,
        -3 * math.pi / 16,
        -math.pi / 8,
        -math.pi / 16,
        math.pi / 16,
        math.pi / 8,
        3 * math.pi / 16,
        math.pi / 4,
    ):
        views.append(_rotate_images(images, angle))

    for tx, ty in (
        (0.2, 0.2),
        (-0.2, 0.2),
        (-0.2, -0.2),
        (0.2, -0.2),
        (0.1, 0.1),
        (-0.1, 0.1),
        (-0.1, -0.1),
        (0.1, -0.1),
    ):
        views.append(_translate_images(images, tx, ty))

    views.append(torch.flip(images, [3]))
    views.append(_grayscale_images(images))

    for k in (1, 2, 3):
        views.append(torch.rot90(images, k, [2, 3]))

    return views


# ---------------------------------------------------------------------------
# AnoVL Detector
# ---------------------------------------------------------------------------

@MODELS.register_module()
class AnoVLDetector(VisionLanguageADModel):
    """AnoVL zero-shot anomaly localization via V-V attention surgery on CLIP.

    Args:
        clip_model: OpenCLIP model name.
        pretrained: Pretrained weights source.
        class_name: Default object class name.
        image_size: Input image size (square).
        features_list: Transformer layer indices to extract features from.
        tta_enabled: Enable test-time adaptation.
        tta_epochs: Number of TTA optimization epochs per image.
        tta_lr: Learning rate for TTA optimizer.
        smoothing_kernel: Kernel size for anomaly map smoothing.
    """

    def __init__(
        self,
        clip_model="ViT-B-16-plus-240",
        pretrained="laion400m_e32",
        class_name="object",
        image_size=240,
        features_list=None,
        tta_enabled=True,
        tta_epochs=5,
        tta_lr=1e-3,
        smoothing_kernel=3,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.class_name = _normalize_class_name(class_name)
        self.image_size = image_size
        self.features_list = [int(layer) for layer in (features_list or [3, 6, 9, 12])]
        self.tta_enabled = tta_enabled
        self.tta_epochs = tta_epochs
        self.tta_lr = tta_lr
        self.smoothing_kernel = smoothing_kernel

        # Build CLIP via OpenCLIPBackbone
        clip_backbone = MODELS.build(
            dict(
                type="OpenCLIPBackbone",
                model_name=clip_model,
                pretrained=pretrained,
                frozen=True,
            )
        )
        self.clip = clip_backbone.model
        self._tokenize = clip_backbone.tokenize

        # Enable output_tokens if available
        if hasattr(self.clip.visual, "output_tokens"):
            self.clip.visual.output_tokens = True

        # Adapt positional embedding for different image sizes
        self._adapt_positional_embedding(image_size)

        # Apply V-V surgery to visual encoder
        apply_vv_surgery(self.clip.visual)

        # ImageNet→CLIP normalization buffers
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

        # Token projection layer (reuses CLIP's ln_post + proj)
        self.linear_layer = LinearLayer(
            self.clip.visual.ln_post, self.clip.visual.proj
        )

        # Get shared embedding dimension
        self.shared_dim = self._get_shared_dim()

        # Text prompts are cached lazily on the runtime device.  The official
        # reference moves the model to CUDA before building prompts; eager
        # CPU-side initialization here would make probe/smoke much slower.
        self._text_cache = {}

        # TTA adapter (lazy init per class)
        self._tta_adapter = None

    def _get_shared_dim(self):
        if (
            hasattr(self.clip, "text_projection")
            and self.clip.text_projection is not None
        ):
            return self.clip.text_projection.shape[1]
        if hasattr(self.clip, "transformer"):
            return self.clip.transformer.width
        return self.clip.text.transformer.width

    def _adapt_positional_embedding(self, input_size):
        """Interpolate positional embedding to match input resolution."""
        visual = self.clip.visual
        if not hasattr(visual, "positional_embedding"):
            return
        if not hasattr(visual, "grid_size"):
            return

        patch_size = (
            visual.conv1.kernel_size[0] if hasattr(visual, "conv1") else None
        )
        if patch_size is None:
            return

        old_grid = visual.grid_size
        new_grid = input_size // patch_size
        if old_grid[0] == new_grid and old_grid[1] == new_grid:
            return

        pe = visual.positional_embedding
        cls_pe = pe[:1]
        patch_pe = pe[1:]
        dim = pe.shape[1]
        patch_pe = patch_pe.reshape(
            1, old_grid[0], old_grid[1], dim
        ).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe,
            size=(new_grid, new_grid),
            mode="bicubic",
            align_corners=False,
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(
            new_grid * new_grid, dim
        )
        visual.positional_embedding = nn.Parameter(
            torch.cat([cls_pe, patch_pe], dim=0)
        )
        visual.grid_size = (new_grid, new_grid)

    def _normalize_for_clip(self, x):
        """Re-normalize from ImageNet normalization to CLIP normalization."""
        x = x * self._imagenet_std + self._imagenet_mean
        return (x - self._clip_mean) / self._clip_std

    @torch.no_grad()
    def _build_text_features(self, class_name):
        """Build text features via domain-aware contrastive state prompting.

        Returns:
            (text_normal, text_abnormal, prompt_embeddings):
                text_normal: (D,) normalized mean normal feature.
                text_abnormal: (D,) normalized mean abnormal feature.
                prompt_embeddings: (2*K, D) all individual prompt embeddings
                    (first K normal, then K abnormal) for TextAdapter.
        """
        device = next(self.clip.parameters()).device
        normal_prompts, abnormal_prompts = _build_prompts_for_class(class_name)

        normal_tokens = self._tokenize(normal_prompts).to(device)
        abnormal_tokens = self._tokenize(abnormal_prompts).to(device)

        # Encode in chunks to avoid OOM
        chunk_size = 128
        normal_embs_raw = []
        for i in range(0, len(normal_tokens), chunk_size):
            emb = self.clip.encode_text(normal_tokens[i : i + chunk_size])
            normal_embs_raw.append(emb)
        all_normal_raw = torch.cat(normal_embs_raw, dim=0)  # (K_n, D)
        # Original: mean of RAW embeddings, then normalize
        normal_feat = F.normalize(all_normal_raw.mean(dim=0), dim=-1)
        # Individual embeddings normalized for TextAdapter
        all_normal = F.normalize(all_normal_raw, dim=-1)

        abnormal_embs_raw = []
        for i in range(0, len(abnormal_tokens), chunk_size):
            emb = self.clip.encode_text(abnormal_tokens[i : i + chunk_size])
            abnormal_embs_raw.append(emb)
        all_abnormal_raw = torch.cat(abnormal_embs_raw, dim=0)  # (K_a, D)
        abnormal_feat = F.normalize(all_abnormal_raw.mean(dim=0), dim=-1)
        all_abnormal = F.normalize(all_abnormal_raw, dim=-1)

        # Concatenate all per-prompt embeddings for adapter: (K_n+K_a, D)
        prompt_embeddings = torch.cat([all_normal, all_abnormal], dim=0)

        return normal_feat, abnormal_feat, prompt_embeddings

    @torch.no_grad()
    def _get_text_features(self, class_name, device, dtype):
        """Get text features for a given class, with caching.

        Returns:
            (text_normal, text_abnormal, prompt_embeddings).
        """
        class_name = _normalize_class_name(class_name)
        if class_name in self._text_cache:
            n, a, p = self._text_cache[class_name]
            return (
                n.to(device=device, dtype=dtype),
                a.to(device=device, dtype=dtype),
                p.to(device=device, dtype=dtype),
            )
        normal_feat, abnormal_feat, prompt_embs = self._build_text_features(
            class_name
        )
        self._text_cache[class_name] = (
            normal_feat.detach().cpu(),
            abnormal_feat.detach().cpu(),
            prompt_embs.detach().cpu(),
        )
        return (
            normal_feat.to(device=device, dtype=dtype),
            abnormal_feat.to(device=device, dtype=dtype),
            prompt_embs.to(device=device, dtype=dtype),
        )

    def _resolve_batch_classes(self, data_samples, batch_size):
        """Resolve per-sample class names from data_samples."""
        if data_samples is None:
            return [self.class_name] * batch_size
        classes = []
        for sample in data_samples:
            cls_name = getattr(sample, "cls_name", None)
            if cls_name is None:
                cls_name = getattr(sample, "class_name", None)
            classes.append(
                _normalize_class_name(cls_name)
                if cls_name is not None
                else self.class_name
            )
        if len(classes) < batch_size:
            classes.extend([self.class_name] * (batch_size - len(classes)))
        return classes[:batch_size]

    @torch.no_grad()
    def _encode_image_vv(self, x, out_layers=None):
        """Encode image through the V-V surgered visual encoder.

        Returns:
            (cls_token, patch_tokens): cls from original path, requested
                V-V patch tokens before projection.
                cls_token: (B, D)
                patch_tokens: list[(B, N, D)] raw patch tokens before projection.
        """
        visual = self.clip.visual
        out_layers = list(self.features_list if out_layers is None else out_layers)
        out_layer_set = set(out_layers)

        # Patch embedding
        x = visual.conv1(x)
        B, C, _, _ = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        # Add CLS token and positional embedding
        cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(
            B, -1, -1
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.unsqueeze(0)

        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        if hasattr(visual, "ln_pre"):
            x = visual.ln_pre(x)

        # Transformer expects (L, B, D) for the surgered blocks
        x = x.permute(1, 0, 2)  # (L, B, D)

        # Pass through surgered transformer blocks
        # First block receives single tensor, creates [x_vv, x_ori] dual path
        patch_layers = []
        for block_idx, block in enumerate(visual.transformer.resblocks, start=1):
            x = block(x)
            if block_idx in out_layer_set:
                patch_layers.append(x[0].permute(1, 0, 2)[:, 1:])

        # x is now [x_vv, x_ori], each (L, B, D)
        x_ori = x[1].permute(1, 0, 2)  # (B, L, D)

        # CLS from original path
        cls_ori = x_ori[:, 0]  # (B, D)
        cls_ori = visual.ln_post(cls_ori)
        if visual.proj is not None:
            cls_ori = cls_ori @ visual.proj

        return cls_ori, patch_layers

    def _compute_anomaly_map(self, patch_tokens, text_normal, text_abnormal):
        """Compute anomaly map from projected patch tokens and text features.

        Args:
            patch_tokens: (B, N, D) projected patch tokens (unnormalized).
            text_normal: (D,) normal text feature.
            text_abnormal: (D,) abnormal text feature.

        Returns:
            anomaly_map: (B, 1, H, W) anomaly map.
        """
        B, N, D = patch_tokens.shape

        # Normalize tokens (matching original: normalize after projection)
        patch_tokens = F.normalize(patch_tokens, dim=-1)

        # Stack text features: (2, D)
        text_features = torch.stack(
            [text_normal, text_abnormal], dim=0
        )  # (2, D)

        # Compute similarity and softmax
        # (B, N, D) @ (D, 2) -> (B, N, 2)
        logits = patch_tokens @ text_features.t()
        probs = logits.softmax(dim=-1)
        anomaly_scores = probs[..., 1]  # (B, N)

        # Reshape to spatial map
        h = w = int(math.sqrt(N))
        if h * w != N:
            # Fallback for non-square
            h = int(math.sqrt(N))
            w = N // h
        anomaly_map = anomaly_scores.reshape(B, 1, h, w)

        return anomaly_map

    def _smooth_anomaly_map(self, anomaly_map):
        """Smooth anomaly map with replicate padding + average pooling."""
        k = self.smoothing_kernel
        if k <= 1:
            return anomaly_map
        pad = k // 2
        anomaly_map = F.pad(anomaly_map, [pad] * 4, mode="replicate")
        anomaly_map = F.avg_pool2d(anomaly_map, kernel_size=k, stride=1)
        return anomaly_map

    @torch.no_grad()
    def _compute_image_score(self, images_clip, text_features):
        """Compute image-level score using augmented views.

        Args:
            images_clip: (B, C, H, W) CLIP-normalized images.
            text_features: (B, 2, D) normal/abnormal text features.

        Returns:
            img_scores: (B,) anomaly scores.
        """
        views = _augment_views(images_clip)
        num_views = len(views)
        batch_size = images_clip.shape[0]
        stacked_views = torch.cat(views, dim=0)

        cls_token, _ = self._encode_image_vv(stacked_views, out_layers=[])
        cls_norm = F.normalize(cls_token, dim=-1).reshape(
            num_views, batch_size, -1
        )
        logits = torch.einsum("vbd,bkd->vbk", cls_norm, text_features)
        probs = logits.softmax(dim=-1)

        # Mean across the official 22-view augmentation set.
        img_scores = probs[..., 1].mean(dim=0)
        return img_scores

    def _run_tta(self, patch_tokens, text_normal, text_abnormal, prompt_embs):
        """Run test-time adaptation with TextAdapter (matches original AnoVL).

        The adapter works in feature space: it transforms D-dim tokens, then
        anomaly scoring is done via similarity with text features.

        Args:
            patch_tokens: (1, N, D) projected patch tokens for single image.
            text_normal: (D,) normal text feature.
            text_abnormal: (D,) abnormal text feature.
            prompt_embs: (2*K, D) all per-prompt embeddings for adapter.

        Returns:
            anomaly_map: (1, 1, H, W) adapted anomaly map.
        """
        device = patch_tokens.device
        _, N, D = patch_tokens.shape
        h = w = int(math.sqrt(N))

        # Enable grad: predict mode runs inside ValLoop's @torch.no_grad()
        with torch.inference_mode(False), torch.enable_grad():
            text_features = torch.stack(
                [text_normal, text_abnormal], dim=-1
            ).to(device=device, dtype=torch.float32).detach().clone()
            tokens_hw = patch_tokens.detach().float().clone().reshape(1, h, w, D)
            prompt_embs = prompt_embs.detach().float().clone()
            adapter = TextAdapter(prompt_embs).to(device=device)
            optimizer = torch.optim.AdamW(
                adapter.parameters(), lr=self.tta_lr, weight_decay=0.0
            )

            for _ in range(self.tta_epochs):
                optimizer.zero_grad()

                # Adapter forward (training mode: adds noise aug)
                # Input: (1, H, W, D) → Output: ((1+noise)*1, H, W, D)
                adapted = adapter(tokens_hw, is_test=False)
                adapted = adapted / adapted.norm(dim=-1, keepdim=True)

                # Compute anomaly map: (..., H, W, D) @ (D, 2) → (..., H, W, 2)
                pred = (adapted @ text_features).softmax(dim=-1)

                loss = _tta_loss(pred)
                loss.backward()
                optimizer.step()

        # Final inference with adapted adapter (no noise)
        adapter.eval()
        with torch.no_grad():
            adapted = adapter(tokens_hw, is_test=True)  # (1, H, W, D)
            adapted = adapted / adapted.norm(dim=-1, keepdim=True)
            anomaly_map = (adapted @ text_features).softmax(dim=-1)
            anomaly_map = anomaly_map[..., 1]  # (1, H, W)
            anomaly_map = anomaly_map.unsqueeze(1)  # (1, 1, H, W)

        return anomaly_map

    def forward(self, inputs, data_samples=None, mode="tensor"):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == "loss":
            return {
                "loss": torch.tensor(
                    0.0, device=inputs.device, requires_grad=True
                )
            }

        # Re-normalize ImageNet → CLIP
        images_clip = self._normalize_for_clip(inputs)
        if images_clip.shape[-2:] != (self.image_size, self.image_size):
            images_clip = F.interpolate(
                images_clip,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Encode image through V-V surgered encoder
        _cls_token, patch_token_layers = self._encode_image_vv(
            images_clip,
            out_layers=self.features_list,
        )
        if not patch_token_layers:
            raise RuntimeError("AnoVL requires at least one feature layer.")

        # Project patch tokens through ln_post + proj (NOT normalized yet —
        # normalization happens inside _compute_anomaly_map / _run_tta, matching
        # the original code where tokens are normalized AFTER the adapter).
        projected_patch_layers = [
            self.linear_layer(patch_tokens_raw)
            for patch_tokens_raw in patch_token_layers
        ]
        patch_tokens = projected_patch_layers[-1]

        if mode == "tensor":
            return F.normalize(patch_tokens, dim=-1)

        # mode == 'predict'
        B = inputs.shape[0]
        input_h, input_w = inputs.shape[-2:]
        class_names = self._resolve_batch_classes(data_samples, B)

        class_text_features = []
        for cls_name in class_names:
            class_text_features.append(
                self._get_text_features(
                    cls_name,
                    device=patch_tokens.device,
                    dtype=patch_tokens.dtype,
                )
            )

        all_results = []
        batch_text_pairs = []
        for i, (text_normal, text_abnormal, prompt_embs) in enumerate(class_text_features):
            batch_text_pairs.append(torch.stack([text_normal, text_abnormal], dim=0))

            tokens_i = patch_tokens[i : i + 1]  # (1, N, D)

            if self.tta_enabled:
                anomaly_map = self._run_tta(
                    tokens_i, text_normal, text_abnormal, prompt_embs
                )
            else:
                anomaly_map = self._compute_anomaly_map(
                    tokens_i, text_normal, text_abnormal
                )

            # Smooth
            anomaly_map = self._smooth_anomaly_map(anomaly_map)

            # Upsample to input size
            anomaly_map = F.interpolate(
                anomaly_map,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=True,
            )

            all_results.append(anomaly_map)

        score_map = torch.cat(all_results, dim=0)  # (B, 1, H, W)

        # Image-level score via augmented views
        with torch.no_grad():
            img_scores = self._compute_image_score(
                images_clip,
                torch.stack(batch_text_pairs, dim=0),
            )

        return build_predict_results(data_samples, img_scores, score_map)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self
