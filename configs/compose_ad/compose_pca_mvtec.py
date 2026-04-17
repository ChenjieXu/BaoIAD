# ComposeAD + PCAScoringHead ≈ DFM baseline
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/schedules/schedule_100e.py',
]

model = dict(
    type='ComposeAD',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(3,),
        frozen=True,
    ),
    neck=None,
    scoring_head=dict(
        type='PCAScoringHead',
        feature_selection='full',
        dim_reduction='none',
        pca_level=0.97,
        scoring='fre',
        pooling_kernel_size=4,
        input_size=(256, 256),
        blur_sigma=0.0,
    ),
    freeze_backbone=True,
)
