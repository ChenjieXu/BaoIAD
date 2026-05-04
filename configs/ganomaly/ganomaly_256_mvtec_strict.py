_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

# GANomaly strict image-only alignment:
# - original `samet-akcay/ganomaly` uses Adam(lr=2e-4, beta1=0.5), batch_size=64,
#   workers=8 and a 15-epoch budget.
# - official folder preprocessing is `Resize(isize) -> CenterCrop(isize) ->
#   ToTensor -> Normalize(0.5, 0.5, 0.5)`.
# - MVTec proxy evaluation keeps the image-only protocol and applies dataset-level
#   min-max normalization to image scores before computing image metrics.
benchmark_multi_class = False
benchmark_keep_dataloader_workers = True
benchmark_preserve_checkpoint_hooks = True
benchmark_result_selector = dict(mode='best', metric='image_auroc')

data_root = 'data/mvtec_ad'
img_size = 256

train_pipeline = [
    dict(type='LoadImage', backend='pil'),
    dict(type='LoadMask', backend='pil'),
    dict(type='ResizeAD', size=img_size, keep_ratio=True, backend='pillow', official_pil=True),
    dict(type='CenterCrop', size=img_size),
    dict(type='NormalizeAD', mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage', backend='pil'),
    dict(type='LoadMask', backend='pil'),
    dict(type='ResizeAD', size=img_size, keep_ratio=True, backend='pillow', official_pil=True),
    dict(type='CenterCrop', size=img_size),
    dict(type='NormalizeAD', mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)),
    dict(type='PackADInputs'),
]

model = dict(
    type='GanomalyDetector',
    strict=True,
    input_size=(256, 256),
    num_input_channels=3,
    n_features=64,
    latent_vec_size=100,
    extra_layers=0,
    add_final_conv_layer=True,
    wadv=1,
    wcon=50,
    wenc=1,
)

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        multi_class=True,
        pipeline=train_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        pipeline=test_pipeline,
    ),
)
val_dataloader = test_dataloader

image_only_metric = dict(
    type='AnomalyDetectionMetric',
    metrics=['image_auroc', 'image_f1max', 'image_ap', 'image_fpr@95tpr'],
    normalize_image_scores=True,
)

val_evaluator = dict(_delete_=True, metrics=[image_only_metric])
test_evaluator = dict(_delete_=True, metrics=[image_only_metric])

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
        save_best='ad/image_auroc',
        rule='greater',
    ),
)

optim_wrapper = dict(
    constructor='GanomalyOptimWrapperConstructor',
    generator=dict(
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999), weight_decay=0),
    ),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999), weight_decay=0),
    ),
)

param_scheduler = []

train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
