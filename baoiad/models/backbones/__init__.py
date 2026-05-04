# Backbones are typically reused from mmpretrain/timm.
from .ast_backbone import (EfficientNetFeatureExtractor,  # noqa: F401
                           EfficientNetLayerExtractor,  # noqa: F401
                           GenericFeatureExtractor)  # noqa: F401
from .clip_backbone import OpenCLIPBackbone  # noqa: F401
from .dinomaly_backbone import DinomalyEncoder  # noqa: F401
from .dinov2_backbone import DINOv2Backbone  # noqa: F401
from .feature_extractor import FeatureExtractor  # noqa: F401
from .musc_clip_backbone import MuScCLIPBackbone, MuScDINOv2Backbone  # noqa: F401
from .raw_backbone import RawBackbone  # noqa: F401
from .saa_saliency_backbone import SAASaliencyBackbone  # noqa: F401
from .timm_backbone import TIMMBackbone  # noqa: F401
from .vitad_backbone import (DistilledVisionTransformerBackbone,  # noqa: F401
                             ViTEncoderBackbone)  # noqa: F401

__all__ = [
    'FeatureExtractor', 'TIMMBackbone', 'OpenCLIPBackbone', 'RawBackbone',
    'DINOv2Backbone', 'DistilledVisionTransformerBackbone',
    'ViTEncoderBackbone', 'DinomalyEncoder', 'EfficientNetFeatureExtractor',
    'EfficientNetLayerExtractor', 'GenericFeatureExtractor', 'MuScCLIPBackbone',
    'MuScDINOv2Backbone', 'SAASaliencyBackbone',
]
