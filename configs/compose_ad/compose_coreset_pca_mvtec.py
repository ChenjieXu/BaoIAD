# NEW COMBINATION: Coreset feature selection + PCA reconstruction error scoring
# Trick from PatchCore (coreset) + Trick from DFM (PCA subspace)
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
        type='PCAScoringHead',
        feature_selection='coreset',  # PatchCore trick: select representative subset
        coreset_ratio=0.1,
        dim_reduction='none',
        pca_level=0.97,
        scoring='fre',
        pooling_kernel_size=1,  # features already pooled by neck
        input_size=(256, 256),
        blur_sigma=4.0,
    ),
    freeze_backbone=True,
)
