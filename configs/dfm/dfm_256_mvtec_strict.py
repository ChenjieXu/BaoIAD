_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

model = dict(
    type='DFMDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2.tv_in1k',
        pretrained=True,
        features_only=True,
        frozen=True,
    ),
    layer='layer3',
    pooling_kernel_size=4,
    pca_level=0.97,
    score_type='fre',
)

# Strict published-reference tracking config.
# Freeze the WRN-50 weight tag explicitly so future timm defaults cannot drift.
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-5),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
