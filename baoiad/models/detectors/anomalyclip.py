"""AnomalyCLIP detector with DPAM (V-V attention surgery) for pixel scoring.

Reference: Zhou et al., "AnomalyCLIP: Object-agnostic Prompt Learning for
Zero-shot Anomaly Detection", ICLR 2024.

DPAM (Disentangled Patch-level Anomaly Matching) replaces Q,K with V in
the attention layers to preserve spatial locality in patch tokens.
"""

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import VisionLanguageADModel

# V-V attention modules reused from AnoVL
from baoiad.models.detectors.anovl import (
    VVAttention, VVResidualBlock, LinearLayer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-aware contrastive state prompting (matching AnoVL reference)
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

ANOMALOUS_STATES = [
    "damaged {}",
    "abnormal {}",
    "imperfect {}",
    "blemished {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

# Class-specific anomaly states for MVTec AD (from AnoVL reference)
CLASS_SPECIFIC_STATES = {
    'bottle': [],
    'toothbrush': [],
    'carpet': ['{} with hole'],
    'hazelnut': ['{} with crack', 'cracked {}', 'cut {}', '{} with hole', '{} with print'],
    'leather': ['color-varied {}', 'cut {}', 'folded {}', '{} with fold', '{} with glue', '{} with poke'],
    'cable': ['bent {}', '{} with bent wire', 'missing {}', '{} with missing wire'],
    'capsule': ['{} with crack'],
    'grid': ['broken {}', '{} with thread'],
    'pill': ['combined {}', '{} with contamination', 'cracked {}', '{} with crack', '{} with faulty imprint', '{} with scratch'],
    'transistor': ['{} with bent lead', 'bent {}', '{} with cut lead', 'damaged {}'],
    'metal_nut': ['bent {}', 'color-varied {}', 'flipped {}', '{} with scratch'],
    'screw': ['{} with manipulated front', '{} with scratch neck'],
    'zipper': ['{} with broken teeth', 'combined {}', 'rough {}', '{} with split teeth', '{} with squeezed teeth'],
    'tile': ['cracked {}', '{} with crack', '{} with glue strip', 'rough {}'],
    'wood': ['{} with hole', '{} with liquid'],
}

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

# MVTec texture categories
TEXTURE_CATEGORIES = frozenset({
    "carpet", "grid", "leather", "tile", "wood",
})


from baoiad.utils.score_utils import normalize_class_name as _normalize_class_name


def _is_texture(class_name):
    return class_name.lower().replace(' ', '_') in TEXTURE_CATEGORIES


def create_prompt_ensemble(class_name="object"):
    """Build domain-aware prompts with class-specific anomaly states.

    This matches the AnoVL reference implementation:
    - Uses industrial templates for all categories
    - Distinguishes texture vs object categories
    - Adds class-specific anomaly states for better detection
    """
    class_name = _normalize_class_name(class_name)
    is_texture = _is_texture(class_name)

    # Select template groups based on category type
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

    # Build normal prompts
    normal_prompts = []
    for templates in template_groups:
        for state in NORMAL_STATES:
            for template in templates:
                normal_prompts.append(template.format(state.format(class_name)))

    # Build anomalous prompts (including class-specific states)
    anomalous_states = list(ANOMALOUS_STATES)
    class_key = class_name.lower().replace(' ', '_')
    if class_key in CLASS_SPECIFIC_STATES:
        anomalous_states.extend(CLASS_SPECIFIC_STATES[class_key])

    anomalous_prompts = []
    for templates in template_groups:
        for state in anomalous_states:
            for template in templates:
                anomalous_prompts.append(template.format(state.format(class_name)))

    return normal_prompts, anomalous_prompts


def _gaussian_blur_bchw(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(int(round(4 * sigma)), 1)
    kernel_size = 2 * radius + 1
    coord = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coord ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


def _augment_views(images):
    """Generate augmented views for robust image-level scoring.

    Args:
        images: (B, C, H, W) tensor.

    Returns:
        List of (B, C, H, W) augmented tensors.
    """
    views = [images]
    # 90-degree rotations
    for k in (1, 2, 3):
        views.append(torch.rot90(images, k, [2, 3]))
    # Horizontal flip
    views.append(torch.flip(images, [3]))
    # Vertical flip
    views.append(torch.flip(images, [2]))
    return views


@MODELS.register_module(force=True)
class AnomalyCLIPDetector(VisionLanguageADModel):
    """AnomalyCLIP detector with DPAM (V-V attention) for pixel-level scoring.

    DPAM replaces Q,K with V in the visual encoder's attention layers to
    preserve spatial locality in patch tokens.

    Key improvements over baseline:
    1. Domain-aware industrial prompts (matching AnoVL)
    2. Texture vs object category distinction
    3. Class-specific anomaly states
    4. Multi-view augmentation for image-level scoring
    5. Support for loading official trained prompts from checkpoint
    """

    def __init__(
        self,
        clip_model='ViT-L-14-336',
        pretrained='openai',
        class_name='object',
        n_prompt_tokens=12,
        image_size=256,
        temperature=0.07,
        gaussian_sigma=4.0,
        dpam_enabled=True,
        dpam_layer=20,
        features_list=None,
        use_view_augmentation=True,
        # Official checkpoint support
        official_checkpoint=None,
        prompt_depth=9,
        prompt_text_length=4,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.dpam_enabled = dpam_enabled
        self.dpam_layer = dpam_layer
        self.use_view_augmentation = use_view_augmentation
        self.official_checkpoint = official_checkpoint
        self.prompt_depth = prompt_depth
        self.prompt_text_length = prompt_text_length

        clip_backbone = MODELS.build(
            dict(
                type='OpenCLIPBackbone',
                model_name=clip_model,
                pretrained=pretrained,
                frozen=True,
            ))
        self.clip = clip_backbone.model
        self._tokenize = clip_backbone.tokenize
        self.class_name = _normalize_class_name(class_name)
        self.n_prompt_tokens = n_prompt_tokens
        self.image_size = image_size
        self.temperature = temperature
        self.gaussian_sigma = gaussian_sigma

        if hasattr(self.clip.visual, 'output_tokens'):
            self.clip.visual.output_tokens = True
        self._adapt_positional_embedding(input_size=image_size)

        # Apply DPAM (V-V attention surgery) to visual encoder
        n_layers = len(self.clip.visual.transformer.resblocks)
        if features_list is not None:
            self.features_list = features_list
        else:
            # Default: extract every 6th layer for 24-layer, every 3rd for 12-layer
            step = max(n_layers // 4, 1)
            self.features_list = list(range(step, n_layers + 1, step))
            if n_layers not in self.features_list:
                self.features_list.append(n_layers)

        if dpam_enabled:
            self._apply_dpam_surgery(dpam_layer)

        # Projection layer: ln_post + visual.proj (same as AnoVL)
        self.linear_layer = LinearLayer(
            self.clip.visual.ln_post, self.clip.visual.proj)

        # BaoIAD inputs are ImageNet-normalized; CLIP expects its own normalization.
        self.register_buffer(
            '_imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

        shared_dim = self._get_shared_dim()
        self.shared_dim = shared_dim

        # Initialize official prompt learner if checkpoint provided
        self._official_prompt_learner = None
        if official_checkpoint is not None:
            self._init_official_prompt_learner(official_checkpoint)

        normal_base, anomalous_base = self._build_base_text_features(self.class_name)
        self.register_buffer('normal_base', normal_base)
        self.register_buffer('anomalous_base', anomalous_base)
        self._text_base_cache = {
            self.class_name: (normal_base.detach().cpu(), anomalous_base.detach().cpu())
        }

        # Learn residuals on top of semantic base prompts for stable convergence.
        self.normal_prompt = nn.Parameter(torch.zeros(n_prompt_tokens, shared_dim))
        self.anomalous_prompt = nn.Parameter(torch.zeros(n_prompt_tokens, shared_dim))
        nn.init.normal_(self.normal_prompt, std=0.02)
        nn.init.normal_(self.anomalous_prompt, std=0.02)

    def _apply_dpam_surgery(self, dpam_layer=20):
        """Apply V-V attention surgery to last dpam_layer visual transformer blocks.

        Official implementation applies DPAM to last 20 layers only, not all layers.
        """
        transformer = self.clip.visual.transformer
        n_blocks = len(transformer.resblocks)
        new_blocks = nn.ModuleList()
        for i, block in enumerate(transformer.resblocks):
            # Only apply V-V surgery to last dpam_layer blocks
            if i >= n_blocks - dpam_layer:
                new_blocks.append(VVResidualBlock(block))
            else:
                new_blocks.append(block)
        transformer.resblocks = new_blocks

    def _init_official_prompt_learner(self, checkpoint_path):
        """Initialize official AnomalyCLIP prompt learner from checkpoint.

        This loads the trained prompts from the official implementation,
        which uses deep prompt learning in the text encoder.

        Args:
            checkpoint_path: Path to official .pth checkpoint file
        """
        import sys
        # Add AnomalyCLIP to path for imports
        anomalyclip_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'AnomalyCLIP')
        anomalyclip_path = os.path.abspath(anomalyclip_path)
        if anomalyclip_path not in sys.path:
            sys.path.insert(0, anomalyclip_path)

        try:
            from prompt_ensemble import AnomalyCLIP_PromptLearner

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            prompt_learner_state = checkpoint.get('prompt_learner', checkpoint)

            # Create prompt learner with same architecture
            design_details = {
                'Prompt_length': self.n_prompt_tokens,
                'learnabel_text_embedding_depth': self.prompt_depth,
                'learnabel_text_embedding_length': self.prompt_text_length,
            }

            device = next(self.clip.parameters()).device

            # Create a CLIP adapter using open_clip
            # This avoids needing to download the official CLIP model
            official_clip = self._create_openclip_adapter()

            prompt_learner = AnomalyCLIP_PromptLearner(official_clip, design_details)
            prompt_learner.load_state_dict(prompt_learner_state)
            prompt_learner.to(device)
            prompt_learner.eval()

            self._official_prompt_learner = prompt_learner
            self._official_clip = official_clip
            logger.info(f"Loaded official prompt learner from {checkpoint_path}")

        except ImportError as e:
            logger.warning(f"Could not load official prompt learner: {e}")
            logger.warning("Falling back to zero-shot mode")
            self._official_prompt_learner = None
        except Exception as e:
            logger.warning(f"Error loading official prompt learner: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("Falling back to zero-shot mode")
            self._official_prompt_learner = None

    def _create_openclip_adapter(self):
        """Create an adapter for open_clip to match official CLIP interface.

        The official AnomalyCLIP prompt learner expects specific attributes
        from the official CLIP model. This adapter provides those interfaces
        using open_clip's model.
        """
        import open_clip

        # Load ViT-L-14-336 from open_clip (same as official)
        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14-336', pretrained='openai'
        )
        model.eval()

        # Create adapter class
        class OpenCLIPAdapter:
            def __init__(self, open_clip_model):
                self._model = open_clip_model
                # Get token embedding
                self.token_embedding = open_clip_model.token_embedding
                self.positional_embedding = open_clip_model.positional_embedding
                self.transformer = open_clip_model.transformer
                self.ln_final = open_clip_model.ln_final
                self.text_projection = open_clip_model.text_projection
                self.dtype = next(open_clip_model.parameters()).dtype

                # Get cast dtype
                self._cast_dtype = open_clip_model.transformer.get_cast_dtype()

            def encode_text_learn(self, prompts, tokenized_prompts, deep_compound_prompts_text=None, normalize=False):
                """Encode text with learned prompts.

                This matches the official encode_text_learn interface.
                """
                x = prompts + self.positional_embedding.to(self._cast_dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND

                if deep_compound_prompts_text is None:
                    x = self.transformer(x)
                else:
                    # Handle deep prompts
                    # The official implementation injects deep prompts at each layer
                    # For simplicity, we use the standard transformer
                    # This may not perfectly match official behavior
                    x = self.transformer(x)

                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.ln_final(x).type(self.dtype)

                # Take features from EOT token
                x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
                if self.text_projection is not None:
                    x = x @ self.text_projection

                return x

            def to(self, device):
                self._model = self._model.to(device)
                self.token_embedding = self._model.token_embedding
                self.positional_embedding = self._model.positional_embedding
                self.transformer = self._model.transformer
                self.ln_final = self._model.ln_final
                self.text_projection = self._model.text_projection
                return self

        return OpenCLIPAdapter(model)

    def _get_shared_dim(self):
        if hasattr(self.clip, 'text_projection') and self.clip.text_projection is not None:
            return self.clip.text_projection.shape[1]
        if hasattr(self.clip, 'transformer'):
            return self.clip.transformer.width
        return self.clip.text.transformer.width

    def _adapt_positional_embedding(self, input_size):
        visual = self.clip.visual
        if not hasattr(visual, 'positional_embedding'):
            return
        if not hasattr(visual, 'grid_size'):
            return

        patch_size = visual.conv1.kernel_size[0] if hasattr(visual, 'conv1') else None
        if patch_size is None:
            return

        old_grid = visual.grid_size
        new_grid_h = input_size // patch_size
        new_grid_w = input_size // patch_size
        if old_grid[0] == new_grid_h and old_grid[1] == new_grid_w:
            return

        pe = visual.positional_embedding
        cls_pe = pe[:1]
        patch_pe = pe[1:]
        dim = pe.shape[1]
        patch_pe = patch_pe.reshape(1, old_grid[0], old_grid[1], dim).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe, size=(new_grid_h, new_grid_w), mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(new_grid_h * new_grid_w, dim)
        visual.positional_embedding = nn.Parameter(torch.cat([cls_pe, patch_pe], dim=0))
        visual.grid_size = (new_grid_h, new_grid_w)

    @torch.no_grad()
    def _build_base_text_features(self, class_name, batch_size=64):
        """Build text features with batched encoding to avoid OOM.

        The prompt ensemble can be large (462+ prompts), so we encode
        them in smaller batches to avoid GPU memory issues.
        """
        device = next(self.clip.parameters()).device
        normal_prompts, anomalous_prompts = create_prompt_ensemble(class_name)

        # Encode in batches to avoid OOM
        normal_embeddings = []
        for i in range(0, len(normal_prompts), batch_size):
            batch_prompts = normal_prompts[i:i + batch_size]
            batch_tokens = self._tokenize(batch_prompts).to(device)
            batch_emb = self.clip.encode_text(batch_tokens)
            normal_embeddings.append(batch_emb)
        normal_emb = torch.cat(normal_embeddings, dim=0)
        normal_emb = F.normalize(normal_emb, dim=-1).mean(dim=0)

        anomalous_embeddings = []
        for i in range(0, len(anomalous_prompts), batch_size):
            batch_prompts = anomalous_prompts[i:i + batch_size]
            batch_tokens = self._tokenize(batch_prompts).to(device)
            batch_emb = self.clip.encode_text(batch_tokens)
            anomalous_embeddings.append(batch_emb)
        anomalous_emb = torch.cat(anomalous_embeddings, dim=0)
        anomalous_emb = F.normalize(anomalous_emb, dim=-1).mean(dim=0)

        return F.normalize(normal_emb, dim=-1), F.normalize(anomalous_emb, dim=-1)

    @torch.no_grad()
    def _get_class_base_features(self, class_name, device, dtype):
        class_name = _normalize_class_name(class_name)
        if class_name in self._text_base_cache:
            normal_base, anomalous_base = self._text_base_cache[class_name]
            return normal_base.to(device=device, dtype=dtype), anomalous_base.to(device=device, dtype=dtype)
        normal_base, anomalous_base = self._build_base_text_features(class_name)
        self._text_base_cache[class_name] = (
            normal_base.detach().cpu(),
            anomalous_base.detach().cpu(),
        )
        return normal_base.to(device=device, dtype=dtype), anomalous_base.to(device=device, dtype=dtype)

    def encode_text_prompts(self, class_name=None, device=None, dtype=None):
        """Encode text prompts for anomaly detection.

        If official checkpoint is loaded, uses trained prompts.
        Otherwise, uses zero-shot prompts with learned residuals.
        """
        # Use official prompt learner if available
        if self._official_prompt_learner is not None:
            return self._encode_official_prompts(device, dtype)

        if class_name is None:
            normal_base = self.normal_base
            anomalous_base = self.anomalous_base
        else:
            if device is None:
                device = self.normal_prompt.device
            if dtype is None:
                dtype = self.normal_prompt.dtype
            normal_base, anomalous_base = self._get_class_base_features(class_name, device, dtype)

        normal_delta = self.normal_prompt.mean(dim=0)
        anomalous_delta = self.anomalous_prompt.mean(dim=0)
        normal_feat = F.normalize(normal_base + normal_delta, dim=-1)
        anomalous_feat = F.normalize(anomalous_base + anomalous_delta, dim=-1)
        return normal_feat, anomalous_feat

    @torch.no_grad()
    def _encode_official_prompts(self, device=None, dtype=None):
        """Encode text features using official trained prompts.

        This uses the deep prompt learning approach from the official
        AnomalyCLIP implementation.
        """
        if device is None:
            device = next(self._official_prompt_learner.parameters()).device
        if dtype is None:
            dtype = next(self._official_prompt_learner.parameters()).dtype

        # Get prompts from official prompt learner
        prompts, tokenized_prompts, compound_prompts_text = self._official_prompt_learner(cls_id=None)

        # Move prompts to correct device
        prompts = prompts.to(device)
        tokenized_prompts = tokenized_prompts.to(device)
        if compound_prompts_text is not None:
            compound_prompts_text = [p.to(device) for p in compound_prompts_text]

        # Ensure official clip is on correct device
        if hasattr(self._official_clip, '_model'):
            self._official_clip._model = self._official_clip._model.to(device)
        self._official_clip = self._official_clip.to(device)

        # Encode using official CLIP's encode_text_learn
        text_features = self._official_clip.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text
        ).float()

        # Split into normal and anomalous (first half normal, second half anomalous)
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Return (normal, anomalous) for batch
        # Shape: text_features is [1, 2, D] after chunking and stacking
        normal_feat = text_features[0, 0]  # [D]
        anomalous_feat = text_features[0, 1]  # [D]

        return normal_feat, anomalous_feat

    def _resolve_batch_classes(self, data_samples, batch_size):
        if data_samples is None:
            return [self.class_name] * batch_size
        classes = []
        for sample in data_samples:
            cls_name = getattr(sample, 'cls_name', None)
            if cls_name is None:
                cls_name = getattr(sample, 'class_name', None)
            classes.append(_normalize_class_name(cls_name) if cls_name is not None else self.class_name)
        if len(classes) < batch_size:
            classes.extend([self.class_name] * (batch_size - len(classes)))
        return classes[:batch_size]

    def _get_class_conditioned_text_features(self, class_names, device, dtype):
        class_feature_cache = {}
        normal_list = []
        anomalous_list = []
        for cls_name in class_names:
            if cls_name not in class_feature_cache:
                class_feature_cache[cls_name] = self.encode_text_prompts(
                    class_name=cls_name, device=device, dtype=dtype)
            normal_feat, anomalous_feat = class_feature_cache[cls_name]
            normal_list.append(normal_feat)
            anomalous_list.append(anomalous_feat)
        return torch.stack(normal_list, dim=0), torch.stack(anomalous_list, dim=0)

    def _compute_raw_similarity(self, visual_tokens, normal_text, anomalous_text):
        """Compute raw dot-product similarity (for training loss).

        This is the original simple dot-product similarity used in training.
        Returns raw similarity scores without softmax.

        Args:
            visual_tokens: (B, N, D)
            normal_text: (B, D) or (D,)
            anomalous_text: (B, D) or (D,)

        Returns:
            sim_normal: (B, N) raw similarity to normal text
            sim_anomalous: (B, N) raw similarity to anomalous text
        """
        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens.unsqueeze(1)
        sim_normal = torch.einsum('bnd,bd->bn', visual_tokens, normal_text)
        sim_anomalous = torch.einsum('bnd,bd->bn', visual_tokens, anomalous_text)
        return sim_normal, sim_anomalous

    def _compute_similarity(self, visual_tokens, normal_text, anomalous_text):
        """Compute similarity using official AnomalyCLIP formula.

        Official implementation:
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        similarity = feats.sum(-1)
        return (similarity/0.07).softmax(-1)

        For our case with normal_text and anomalous_text:
        - visual_tokens: (B, N, D)
        - normal_text: (B, D) or (D,)
        - anomalous_text: (B, D) or (D,)

        Returns:
            sim_normal: (B, N) normalized similarity to normal text
            sim_anomalous: (B, N) normalized similarity to anomalous text
        """
        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens.unsqueeze(1)

        # Handle batch dimension for text features
        if normal_text.ndim == 1:
            normal_text = normal_text.unsqueeze(0)
        if anomalous_text.ndim == 1:
            anomalous_text = anomalous_text.unsqueeze(0)

        # Stack text features: (B, 2, D)
        text_features = torch.stack([normal_text, anomalous_text], dim=1)

        B, N, D = visual_tokens.shape

        # Element-wise product and sum (matching official compute_similarity)
        # visual_tokens: (B, N, D) -> (B, N, 1, D)
        # text_features: (B, 2, D) -> (B, 1, 2, D)
        feats = visual_tokens.unsqueeze(2) * text_features.unsqueeze(1)  # (B, N, 2, D)
        similarity = feats.sum(-1)  # (B, N, 2)

        # Apply temperature and softmax (official uses 0.07)
        similarity = (similarity / 0.07).softmax(-1)

        # Return separate normal and anomalous similarities
        sim_normal = similarity[..., 0]  # (B, N)
        sim_anomalous = similarity[..., 1]  # (B, N)

        return sim_normal, sim_anomalous

    def _normalize_for_clip(self, x):
        x = x * self._imagenet_std + self._imagenet_mean
        return (x - self._clip_mean) / self._clip_std

    @torch.no_grad()
    def _encode_image_vv(self, x):
        """Encode image through V-V surgered visual encoder.

        Returns:
            cls_token: (B, D) CLS from original path (for image scoring).
            patch_levels: list of (B, N, D) projected V-V patch tokens from
                selected layers (for pixel scoring).
        """
        visual = self.clip.visual

        # Patch embedding
        x = visual.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        # Add CLS + positional embedding
        cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.unsqueeze(0)

        if hasattr(visual, 'patch_dropout'):
            x = visual.patch_dropout(x)
        if hasattr(visual, 'ln_pre'):
            x = visual.ln_pre(x)

        # Transformer expects (L, B, D)
        x = x.permute(1, 0, 2)

        # Walk through transformer blocks, collecting multi-layer features
        layer_tokens = []
        for idx, block in enumerate(visual.transformer.resblocks):
            x = block(x)
            layer_idx = idx + 1  # 1-indexed
            if layer_idx in self.features_list:
                if isinstance(x, list):
                    # V-V path: x[0] = V-V, x[1] = original
                    vv_tokens = x[0].permute(1, 0, 2)[:, 1:]  # (B, N, D)
                else:
                    vv_tokens = x.permute(1, 0, 2)[:, 1:]  # (B, N, D)
                # Project through ln_post + proj
                projected = self.linear_layer(vv_tokens)
                layer_tokens.append(projected)

        # CLS from original path
        if isinstance(x, list):
            x_ori = x[1].permute(1, 0, 2)  # (B, L, D)
        else:
            x_ori = x.permute(1, 0, 2)
        cls_ori = x_ori[:, 0]
        cls_ori = visual.ln_post(cls_ori)
        if visual.proj is not None:
            cls_ori = cls_ori @ visual.proj

        return cls_ori, layer_tokens

    def _reshape_score_map(self, anomaly_scores):
        batch_size, num_patches = anomaly_scores.shape
        h = w = int(math.sqrt(num_patches))
        if h * w != num_patches:
            if hasattr(self.clip.visual, 'grid_size'):
                h, w = self.clip.visual.grid_size
            else:
                h = 1
                w = num_patches
        return anomaly_scores.reshape(batch_size, 1, h, w)

    def _compute_image_score_robust(self, images_clip, text_normal, text_anomalous):
        """Compute image-level score with multi-view augmentation.

        Uses rotations and flips for more robust scoring.
        """
        if not self.use_view_augmentation:
            # Single view (original behavior)
            cls_features, _ = self._encode_image_vv(images_clip)
            cls_features = F.normalize(cls_features, dim=-1)
            sim_normal = torch.einsum('bd,bd->b', cls_features, text_normal)
            sim_anomalous = torch.einsum('bd,bd->b', cls_features, text_anomalous)
            logits = torch.stack([sim_normal, sim_anomalous], dim=-1) / self.temperature
            return logits.softmax(dim=-1)[..., 1]

        # Multi-view augmentation for robust scoring
        views = _augment_views(images_clip)
        all_scores = []

        for view in views:
            cls_features, _ = self._encode_image_vv(view)
            cls_features = F.normalize(cls_features, dim=-1)
            sim_normal = torch.einsum('bd,bd->b', cls_features, text_normal)
            sim_anomalous = torch.einsum('bd,bd->b', cls_features, text_anomalous)
            logits = torch.stack([sim_normal, sim_anomalous], dim=-1) / self.temperature
            probs = logits.softmax(dim=-1)[..., 1]
            all_scores.append(probs)

        # Average scores across views
        return torch.stack(all_scores, dim=0).mean(dim=0)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        # Re-normalize ImageNet -> CLIP
        x = self._normalize_for_clip(inputs)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False)

        B = inputs.shape[0]
        class_names = self._resolve_batch_classes(data_samples, B)
        text_normal, text_anomalous = self._get_class_conditioned_text_features(
            class_names, device=x.device, dtype=x.dtype)

        if mode == 'loss':
            # Training: compute loss with single view for efficiency
            cls_features, patch_levels = self._encode_image_vv(x)
            cls_features = F.normalize(cls_features, dim=-1)
            patch_levels = [F.normalize(level, dim=-1) for level in patch_levels]

            sim_global_normal, sim_global_anomalous = self._compute_raw_similarity(
                cls_features, text_normal, text_anomalous)
            global_logits = torch.stack(
                [sim_global_normal, sim_global_anomalous], dim=-1) / self.temperature

            targets = torch.zeros_like(sim_global_normal, dtype=torch.long)
            loss_global = F.cross_entropy(
                global_logits.reshape(-1, 2), targets.reshape(-1))

            patch_losses = []
            for patch_tokens in patch_levels:
                sim_normal, sim_anomalous = self._compute_raw_similarity(
                    patch_tokens, text_normal, text_anomalous)
                logits = torch.stack(
                    [sim_normal, sim_anomalous], dim=-1) / self.temperature
                patch_targets = torch.zeros_like(sim_normal, dtype=torch.long)
                patch_losses.append(F.cross_entropy(
                    logits.reshape(-1, 2), patch_targets.reshape(-1)))
            if patch_losses:
                loss_patch = torch.stack(patch_losses).mean()
            else:
                loss_patch = loss_global * 0.0

            margin = 0.2
            loss_margin = F.relu(
                margin - (sim_global_normal - sim_global_anomalous)).mean()
            normal_global, anomalous_global = self.encode_text_prompts()
            loss_sep = F.relu(F.cosine_similarity(
                normal_global.unsqueeze(0),
                anomalous_global.unsqueeze(0)).mean() + 0.2)
            return {
                'loss': loss_global + loss_patch
                + 0.5 * loss_margin + 0.1 * loss_sep
            }

        if mode == 'predict':
            # Use robust image scoring with view augmentation
            img_scores = self._compute_image_score_robust(x, text_normal, text_anomalous)

            # Pixel-level scoring with V-V patch tokens
            _, patch_levels = self._encode_image_vv(x)
            patch_levels = [F.normalize(level, dim=-1) for level in patch_levels]

            score_maps = []
            for patch_tokens in patch_levels:
                sim_normal, sim_anomalous = self._compute_similarity(
                    patch_tokens, text_normal, text_anomalous)
                # Official formula: (anomaly_sim + 1 - normal_sim) / 2
                # sim_normal and sim_anomalous are already softmax probabilities
                anomaly_map = (sim_anomalous + 1 - sim_normal) / 2.0
                level_map = self._reshape_score_map(anomaly_map)
                level_map = F.interpolate(
                    level_map, size=inputs.shape[-2:],
                    mode='bilinear', align_corners=False)
                score_maps.append(level_map)

            if score_maps:
                # Mean of probabilities across layers
                score_map = torch.stack(score_maps, dim=0).mean(dim=0)
                if self.gaussian_sigma > 0:
                    score_map = _gaussian_blur_bchw(
                        score_map, float(self.gaussian_sigma))
            else:
                score_map = img_scores.view(-1, 1, 1, 1).expand(
                    -1, 1, inputs.shape[-2], inputs.shape[-1])
            return build_predict_results(data_samples, img_scores, score_map)

        # Tensor mode: return features
        cls_features, patch_levels = self._encode_image_vv(x)
        return patch_levels[-1] if patch_levels else cls_features.unsqueeze(1)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self
