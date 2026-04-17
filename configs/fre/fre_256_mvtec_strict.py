_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

model = dict(
    type='FREDetector',
    backbone=dict(
        type='TIMMBackbone',
        model_name='wide_resnet50_2',
        pretrained=True,
        features_only=True,
        frozen=True,
        allow_legacy_fallback=False,
    ),
    layer='layer3',
    pooling_kernel_size=4,  # Match anomalib: 4 (not 2)
    input_dim=16384,  # WRN50 layer3: 1024ch * (16/4)^2 = 16384
    latent_dim=220,
    loss=dict(type='MSELoss'),
)

# Anomalib: Adam lr=1e-3, no weight_decay, no scheduler
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=0),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=220, val_interval=10)  # Align with anomalib: 220 epochs
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

# anomalib FRE reports the final validation snapshot after 220 epochs.
benchmark_result_selector = dict(mode='last')
