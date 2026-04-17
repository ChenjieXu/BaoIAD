_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/mvtec_ad.py']

model = dict(
    type='MambaADDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnet34',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3),
        frozen=True,
    ),
    num_blocks=2,
    d_state=16,
    d_conv=3,
    expand=2,
    smooth_sigma=4.0,
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5),
)
param_scheduler = []
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=20)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
