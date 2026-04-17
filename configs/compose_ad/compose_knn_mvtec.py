# ComposeAD + KNNScoringHead ≈ PatchCore baseline
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
        out_indices=(2, 3),
        frozen=True,
    ),
    neck=dict(
        type='MultiScalePooling',
        output_size=28,
    ),
    scoring_head=dict(
        type='KNNScoringHead',
        feature_selection='coreset',
        dim_reduction='none',
        coreset_ratio=0.1,
        num_neighbors=9,
        patchsize=3,
        input_size=(256, 256),
        blur_sigma=4.0,
    ),
    freeze_backbone=True,
)
