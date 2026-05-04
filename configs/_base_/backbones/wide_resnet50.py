backbone = dict(
    type='FeatureExtractor',
    backbone_name='wide_resnet50_2',
    pretrained=True,
    out_indices=(1, 2, 3),
    frozen=True,
)
