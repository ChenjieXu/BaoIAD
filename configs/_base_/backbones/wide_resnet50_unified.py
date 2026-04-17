# Unified backbone for fair ablation: WRN-50-2 with FeatureExtractor
backbone = dict(
    type='FeatureExtractor',
    backbone_name='wide_resnet50_2',
    pretrained=True,
    out_indices=(1, 2, 3),
    frozen=True,
)

# RawBackbone variant for methods that need the full model object
backbone_raw = dict(
    type='RawBackbone',
    backbone_name='wide_resnet50_2',
    pretrained=True,
    frozen=True,
)
