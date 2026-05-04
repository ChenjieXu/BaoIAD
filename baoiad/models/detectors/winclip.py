"""WinCLIP anomaly detector — CLIP-based zero-/few-shot anomaly detection.

Faithful reimplementation of the WinCLIP algorithm from anomalib.
Reference: https://arxiv.org/abs/2303.14814
"""

from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToPILImage
from baoiad.models.predict_utils import build_predict_results
from baoiad.models.base_ad_model import VisionLanguageADModel

from baoiad.registry import MODELS

BACKBONE = "ViT-B-16-plus-240"
PRETRAINED = "laion400m_e31"
TEMPERATURE = 0.07
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ---------------------------------------------------------------------------
# Prompt ensemble (from anomalib winclip/prompting.py)
# ---------------------------------------------------------------------------
NORMAL_STATES = [
    "{}", "flawless {}", "perfect {}", "unblemished {}",
    "{} without flaw", "{} without defect", "{} without damage",
]

ANOMALOUS_STATES = [
    "damaged {}", "{} with flaw", "{} with defect", "{} with damage",
]

TEMPLATES = [
    "a cropped photo of the {}.", "a close-up photo of a {}.",
    "a close-up photo of the {}.", "a bright photo of a {}.",
    "a bright photo of the {}.", "a dark photo of the {}.",
    "a dark photo of a {}.", "a jpeg corrupted photo of the {}.",
    "a jpeg corrupted photo of the {}.", "a blurry photo of the {}.",
    "a blurry photo of a {}.", "a photo of a {}.",
    "a photo of the {}.", "a photo of a small {}.",
    "a photo of the small {}.", "a photo of a large {}.",
    "a photo of the large {}.", "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection.",
]


def create_prompt_ensemble(class_name="object"):
    normal_states = [s.format(class_name) for s in NORMAL_STATES]
    normal_ensemble = [t.format(s) for s in normal_states for t in TEMPLATES]
    anomalous_states = [s.format(class_name) for s in ANOMALOUS_STATES]
    anomalous_ensemble = [t.format(s) for s in anomalous_states for t in TEMPLATES]
    return normal_ensemble, anomalous_ensemble


# ---------------------------------------------------------------------------
# Utility functions (from anomalib winclip/utils.py)
# ---------------------------------------------------------------------------

def _cosine_similarity(input1, input2):
    """Pairwise cosine similarity between two tensors."""
    input1_ndim = input1.ndim
    input2_ndim = input2.ndim
    if input1_ndim == 2 and input2_ndim == 2:
        input1 = input1.unsqueeze(0)
        input2 = input2.unsqueeze(0)
    elif input1_ndim == 2 and input2_ndim == 3:
        input1 = input1.unsqueeze(1)
    elif input1_ndim == 3 and input2_ndim == 2:
        input2 = input2.repeat(input1.shape[0], 1, 1)
    input1_norm = F.normalize(input1, p=2, dim=-1)
    input2_norm = F.normalize(input2, p=2, dim=-1)
    similarity = torch.bmm(input1_norm, input2_norm.transpose(-2, -1))
    if input1_ndim == 2 and input2_ndim == 2:
        return similarity.squeeze(0)
    if input1_ndim == 2 and input2_ndim == 3:
        return similarity.squeeze(1)
    return similarity


def class_scores(image_embeddings, text_embeddings, temperature=1.0, target_class=None):
    scores = (_cosine_similarity(image_embeddings, text_embeddings) / temperature).softmax(dim=-1)
    if target_class is not None:
        return scores[..., target_class]
    return scores


def harmonic_aggregation(window_scores, output_size, masks):
    batch_size = window_scores.shape[0]
    height, width = output_size
    scores = []
    for idx in range(height * width):
        patch_mask = torch.any(masks == idx, dim=0)
        scores.append(sum(patch_mask) / (1 / window_scores.T[patch_mask]).sum(dim=0))
    return torch.stack(scores).T.reshape(batch_size, height, width).nan_to_num(posinf=0.0)


def visual_association_score(embeddings, reference_embeddings):
    reference_embeddings = reference_embeddings.reshape(-1, embeddings.shape[-1])
    scores = _cosine_similarity(embeddings, reference_embeddings)
    return (1 - scores).min(dim=-1)[0] / 2


def make_masks(grid_size, kernel_size, stride=1):
    height, width = grid_size
    grid = torch.arange(height * width).reshape(1, height, width)
    return F.unfold(grid.float(), kernel_size=kernel_size, stride=stride).int()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@MODELS.register_module(force=True)
class WinClipDetector(VisionLanguageADModel):
    """WinCLIP zero-/few-shot anomaly detection using CLIP.

    Args:
        class_name: Object class name for prompt ensemble (e.g. 'bottle').
        scales: Sliding window scales for multi-scale detection.
        k_shot: Number of reference normal images for few-shot mode.
        reference_images: Optional pre-loaded reference images (K, C, H, W).
        apply_transform: Whether to apply CLIP's native PIL transform pipeline.
            When False, inputs are assumed to come from the standard BaoIAD
            ImageNet-normalized pipeline and are re-normalized to CLIP stats.
    """

    def __init__(self, class_name="object", scales=(2, 3), k_shot=0,
                 reference_images=None, backbone=None,
                 apply_transform=False,
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.class_name = class_name
        self.scales = scales
        self.k_shot = k_shot
        self.temperature = TEMPERATURE
        self.apply_transform = apply_transform

        # Initialize CLIP model via registry
        if backbone is None:
            backbone = dict(type='OpenCLIPBackbone', model_name=BACKBONE,
                            pretrained=PRETRAINED, frozen=True)
        clip_backbone = MODELS.build(backbone)
        self.clip = clip_backbone.model
        self._tokenize = clip_backbone.tokenize
        self._clip_preprocess = clip_backbone.preprocess  # Store CLIP's native transform
        self.clip.visual.output_tokens = True

        # Interpolate positional embedding only when NOT using CLIP's native transform.
        # CLIP's transform resizes to native resolution (240), so PE is already correct.
        # Without transform, input is 256x256, so PE needs interpolation.
        if not self.apply_transform:
            self._adapt_positional_embedding()

        self.grid_size = self.clip.visual.grid_size

        # Generate masks
        self._masks = [make_masks(self.grid_size, scale, 1) for scale in self.scales]

        # Placeholders for embeddings (registered as buffers for device tracking)
        self.register_buffer('_text_embeddings', torch.empty(0), persistent=False)
        self.register_buffer('_patch_embeddings', torch.empty(0), persistent=False)
        self.register_buffer(
            '_imagenet_mean',
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_imagenet_std',
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_clip_mean',
            torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_clip_std',
            torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        # _visual_embeddings is a list of tensors, handled manually
        self._visual_embeddings = None
        self._text_embedding_cache = {}

        # Setup
        self._setup(class_name, reference_images)

    def _adapt_positional_embedding(self, input_size=256):
        """Interpolate CLIP positional embedding to match input resolution."""
        visual = self.clip.visual
        pe = visual.positional_embedding  # (N_old, D)
        patch_size = visual.conv1.kernel_size[0] if hasattr(visual, 'conv1') else visual.patch_embed.patch_size[0]
        old_grid = visual.grid_size  # (H_old, W_old)
        new_grid_h = input_size // patch_size
        new_grid_w = input_size // patch_size

        if old_grid[0] == new_grid_h and old_grid[1] == new_grid_w:
            return  # no resize needed

        cls_pe = pe[:1]  # (1, D)
        patch_pe = pe[1:]  # (H*W, D)
        D = pe.shape[1]
        patch_pe = patch_pe.reshape(1, old_grid[0], old_grid[1], D).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(new_grid_h, new_grid_w), mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(new_grid_h * new_grid_w, D)
        new_pe = torch.cat([cls_pe, patch_pe], dim=0)
        visual.positional_embedding = nn.Parameter(new_pe)
        visual.grid_size = (new_grid_h, new_grid_w)

    def _setup(self, class_name=None, reference_images=None):
        """Collect text and visual embeddings."""
        if class_name is not None:
            self.class_name = class_name
            self._text_embeddings = self._get_text_embeddings_for_class(
                class_name, device=self._clip_device()
            )
        if reference_images is not None:
            self.k_shot = reference_images.shape[0]
            self._collect_visual_embeddings(reference_images)

    def setup_reference_images(self, reference_images):
        """Set up few-shot reference images after init."""
        self.k_shot = reference_images.shape[0]
        self._collect_visual_embeddings(reference_images)

    def _clip_device(self):
        parameter = next(self.clip.parameters(), None)
        if parameter is not None:
            return parameter.device
        return self._imagenet_mean.device

    @torch.no_grad()
    def _collect_text_embeddings(self, class_name):
        device = self._clip_device()
        normal_prompts, anomalous_prompts = create_prompt_ensemble(class_name)
        normal_tokens = self._tokenize(normal_prompts)
        anomalous_tokens = self._tokenize(anomalous_prompts)
        normal_emb = self.clip.encode_text(normal_tokens.to(device))
        anomalous_emb = self.clip.encode_text(anomalous_tokens.to(device))
        normal_emb = torch.mean(normal_emb, dim=0, keepdim=True)
        anomalous_emb = torch.mean(anomalous_emb, dim=0, keepdim=True)
        text_embeddings = torch.cat((normal_emb, anomalous_emb)).detach().cpu()
        self._text_embedding_cache[class_name] = text_embeddings
        if class_name == self.class_name:
            self._text_embeddings = text_embeddings
        return text_embeddings

    def _get_text_embeddings_for_class(self, class_name, device):
        cached = self._text_embedding_cache.get(class_name)
        if cached is None:
            cached = self._collect_text_embeddings(class_name)
        return cached.to(device)

    def _resolve_batch_classes(self, data_samples, batch_size):
        if not data_samples:
            return [self.class_name] * batch_size

        classes = []
        for sample in data_samples[:batch_size]:
            cls_name = getattr(sample, 'cls_name', None)
            if cls_name is None:
                cls_name = getattr(sample, 'class_name', None)
            classes.append(str(cls_name) if cls_name else self.class_name)

        if len(classes) < batch_size:
            classes.extend([self.class_name] * (batch_size - len(classes)))
        return classes

    def _get_batch_text_embeddings(self, class_names, device):
        class_embeddings = {}
        for class_name in dict.fromkeys(class_names):
            class_embeddings[class_name] = self._get_text_embeddings_for_class(class_name, device)
        return torch.stack([class_embeddings[class_name] for class_name in class_names], dim=0)

    @torch.no_grad()
    def _collect_visual_embeddings(self, images):
        _, self._visual_embeddings, patch_emb = self._encode_image(images)
        self._patch_embeddings = patch_emb

    @torch.no_grad()
    def _encode_image(self, batch):
        """Encode batch of images → (image_emb, window_emb_list, patch_emb)."""
        if self.apply_transform:
            device = batch.device
            batch = torch.stack([self.transform(img) for img in batch]).to(device)
        else:
            batch = self._normalize_for_clip(batch)

        outputs = {}

        def get_feature_map(name):
            def hook(_model, inputs, _outputs):
                outputs[name] = inputs[0].detach()
            return hook

        handle = self.clip.visual.patch_dropout.register_forward_hook(
            get_feature_map("feature_map")
        )

        image_embeddings, patch_embeddings = self.clip.encode_image(batch)

        handle.remove()

        feature_map = outputs["feature_map"]
        window_embeddings = [
            self._get_window_embeddings(feature_map, masks)
            for masks in self._masks
        ]
        return image_embeddings, window_embeddings, patch_embeddings

    def _normalize_for_clip(self, batch):
        batch = batch.float()
        imagenet_mean = self._imagenet_mean.to(batch.device)
        imagenet_std = self._imagenet_std.to(batch.device)
        clip_mean = self._clip_mean.to(batch.device)
        clip_std = self._clip_std.to(batch.device)
        batch = batch * imagenet_std + imagenet_mean
        return (batch - clip_mean) / clip_std

    @property
    def transform(self) -> Compose:
        """Get CLIP's transform pipeline.

        Retrieves transforms from CLIP backbone and prepends ToPILImage transform
        since original transforms expect PIL images.

        Returns:
            Compose: Transform pipeline for preprocessing images.
        """
        transforms = copy(self._clip_preprocess.transforms)
        transforms.insert(0, ToPILImage())
        return Compose(transforms)

    def _get_window_embeddings(self, feature_map, masks):
        batch_size = feature_map.shape[0]
        n_masks = masks.shape[1]
        device = feature_map.device

        class_index = torch.zeros(1, n_masks, dtype=int, device=device)
        masks_dev = masks.to(device)
        masks_full = torch.cat((class_index, masks_dev + 1)).T

        masked = torch.cat([torch.index_select(feature_map, 1, mask) for mask in masks_full])
        masked = self.clip.visual.patch_dropout(masked)
        masked = self.clip.visual.ln_pre(masked)
        masked = masked.permute(1, 0, 2)
        masked = self.clip.visual.transformer(masked)
        masked = masked.permute(1, 0, 2)
        masked = self.clip.visual.ln_post(masked)
        pooled, _ = self.clip.visual._global_pool(masked)
        if self.clip.visual.proj is not None:
            pooled = pooled @ self.clip.visual.proj
        return pooled.reshape((n_masks, batch_size, -1)).permute(1, 0, 2)

    def _compute_zero_shot_scores(self, image_scores, window_embeddings, text_embeddings):
        multi_scale_scores = [
            image_scores.view(-1, 1, 1).repeat(1, self.grid_size[0], self.grid_size[1])
        ]
        for window_emb, mask in zip(window_embeddings, self._masks):
            scores = class_scores(window_emb, text_embeddings, self.temperature, target_class=1)
            multi_scale_scores.append(harmonic_aggregation(scores, self.grid_size, mask.to(scores.device)))
        return (len(self.scales) + 1) / (1 / torch.stack(multi_scale_scores)).sum(dim=0)

    def _compute_few_shot_scores(self, patch_embeddings, window_embeddings):
        multi_scale_scores = [
            visual_association_score(patch_embeddings, self._patch_embeddings)
            .reshape((-1, *self.grid_size))
        ]
        for window_emb, ref_emb, mask in zip(window_embeddings, self._visual_embeddings, self._masks):
            scores = visual_association_score(window_emb, ref_emb)
            multi_scale_scores.append(harmonic_aggregation(scores, self.grid_size, mask.to(scores.device)))
        return torch.stack(multi_scale_scores).mean(dim=0)

    @torch.no_grad()
    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        if mode == 'loss':
            # WinCLIP is zero-/few-shot, no training loss
            return {'loss': torch.tensor(0.0, device=inputs.device, requires_grad=True)}

        image_embeddings, window_embeddings, patch_embeddings = self._encode_image(inputs)
        batch_size = inputs.shape[0]
        class_names = self._resolve_batch_classes(data_samples, batch_size)
        text_embeddings = self._get_batch_text_embeddings(class_names, inputs.device)

        # Zero-shot scores
        image_scores = class_scores(image_embeddings, text_embeddings,
                                    self.temperature, target_class=1)
        multi_scale_scores = self._compute_zero_shot_scores(
            image_scores, window_embeddings, text_embeddings
        )

        # Few-shot scores
        if self.k_shot and self._patch_embeddings.numel() > 0:
            few_shot_scores = self._compute_few_shot_scores(patch_embeddings, window_embeddings)
            multi_scale_scores = (multi_scale_scores + few_shot_scores) / 2
            image_scores = (image_scores + few_shot_scores.amax(dim=(-2, -1))) / 2

        # Upsample to input size
        pixel_scores = F.interpolate(
            multi_scale_scores.unsqueeze(1),
            size=inputs.shape[-2:],
            mode="bilinear",
        )

        if mode == 'predict':
            return build_predict_results(data_samples, image_scores, pixel_scores)

        return image_scores, pixel_scores

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self
