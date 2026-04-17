# Strict SimpleNet alignment against the official DonaldRR/SimpleNet mainline.
#
# Official run.sh freezes:
# - wideresnet50 backbone, layer2/layer3
# - Resize(329) -> CenterCrop(288)
# - batch_size=8
# - seed=0
# - meta_epochs=40, gan_epochs=4
# - pre_proj=1, noise_std=0.015, dsc_margin=0.5
#
# MMEngine mapping:
# - one train epoch = one full dataloader pass
# - official 40 meta-epochs x 4 gan epochs => 160 passes
# - validation every 4 passes

_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/mvtec_ad.py']

benchmark_result_selector = dict(mode='best', metric='image_auroc')
benchmark_keep_dataloader_workers = True

data_root = 'data/mvtec_ad'
resize_size = 329
img_size = 288

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=resize_size),
    dict(type='CenterCrop', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=resize_size),
    dict(type='CenterCrop', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        cls_names=['bottle'],
        multi_class=False,
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        cls_names=['bottle'],
        multi_class=False,
        pipeline=test_pipeline,
    ),
)

val_dataloader = test_dataloader

test_evaluator = dict(
    type='AnomalyDetectionMetric',
    normalize_image_scores=True,
    normalize_pred_maps='batch_broadcast',
)
val_evaluator = test_evaluator

model = dict(
    type='SimpleNetDetector',
    strict=True,
    image_size=img_size,
    gaussian_sigma=4.0,
    backbone=dict(
        type='FeatureExtractor',
        backbone_name='wide_resnet50_2',
        pretrained=True,
        out_indices=(2, 3),
        frozen=True,
    ),
    target_dim=1536,
    pretrain_embed_dim=1536,
    noise_std=0.015,
    mix_noise=1,
    dsc_margin=0.5,
    patchsize=3,
    patchstride=1,
    dsc_layers=2,
    dsc_hidden=1024,
    pre_proj=1,
    proj_layer_type=0,
)

optim_wrapper = dict(
    constructor='SimpleNetOptimWrapperConstructor',
    projection=dict(
        optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-2),
    ),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=2e-4, weight_decay=1e-5),
    ),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=160, val_interval=4)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')

randomness = dict(seed=0, deterministic=False)
