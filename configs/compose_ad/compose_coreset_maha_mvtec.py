# NEW COMBINATION: Coreset feature selection + Mahalanobis distance scoring
# Trick from PatchCore (coreset) + Trick from PaDiM (Gaussian + Mahalanobis)
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
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    neck=None,
    scoring_head=dict(
        type='GaussianScoringHead',
        feature_selection='coreset',  # PatchCore trick: select representative subset
        coreset_ratio=0.1,
        dim_reduction='random',       # PaDiM trick: random dim selection
        d_reduced=550,
        eps=0.01,
        input_size=(256, 256),
        blur_sigma=4.0,
    ),
    freeze_backbone=True,
)
