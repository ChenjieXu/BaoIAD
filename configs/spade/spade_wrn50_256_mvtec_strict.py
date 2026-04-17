_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/mvtec_ad.py']

# Use TIMMBackbone for alignment with anomalib (timm weights)
model = dict(
    type='SPADEDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    k=5,
)

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

# SPADE is memory-bank based, no real training needed
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-5),
)
