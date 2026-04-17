# InvAD strict MUAD config: official ADer multi-class MVTec protocol.
_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/mvtec_ad.py']

data_root = 'data/mvtec_ad'
benchmark_multi_class = True
benchmark_keep_dataloader_workers = True
benchmark_preserve_checkpoint_hooks = True
benchmark_resume_existing = True
benchmark_timeout = 108000
# ADer trainer tracks `metric (Max)` per metric; strict benchmark should use
# per-metric best snapshots when rebuilding report numbers from logs.
benchmark_result_selector = dict(
    mode='best_per_metric',
    metrics=['image_auroc', 'pixel_auroc'],
)

# Official encoder: timm wide_resnet50_2, ImageNet pretrained (matching ADer pretrained=True).
backbone = dict(
    type='TIMMBackbone',
    model_name='wide_resnet50_2',
    pretrained=False,
    checkpoint_path='pretrained/wide_resnet50_racm-8234f177.pth',
    strict=False,
    features_only=True,
    out_indices=(1, 2, 3),
    frozen=True,
)

# multi_class=True loads all 15 categories together (~3629 training images)
train_dataloader = dict(batch_size=32, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='train',
                 multi_class=True, shuffle_train_data=True,
                 pipeline={{_base_.train_pipeline}}))
test_dataloader = dict(batch_size=32, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='test',
                 multi_class=True, pipeline={{_base_.test_pipeline}}))
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(type='InvADDetector',
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
    backbone=backbone,
    out_cha=64,
    latent_channel_size=16,
    gaussian_sigma=4.0,
)

# ADer reference: Adam lr=0.004 (0.001*32/8), betas=(0.0, 0.99), NO weight_decay
# Note: ADer cfg sets weight_decay=1e-4 but does NOT pass it to optim.kwargs, so effective wd=0
# StepLR decay at 80% of epochs (epoch 240), decay_rate=0.1
# ADer default clip_grad=5.0 (from cfg_common.py, not overridden by InvAD config)
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=4e-3, betas=(0.0, 0.99)),
    clip_grad=dict(max_norm=5.0),
)
param_scheduler = [
    dict(type='StepLR', step_size=240, gamma=0.1, by_epoch=True),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3, save_last=True),
)
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=30)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
