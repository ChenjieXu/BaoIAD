_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

img_size = 448

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

model = dict(
    type='UniVADDetector',
    clip_model='ViT-L-14-336',
    clip_pretrained='openai',
    dinov2_model='dinov2_vitg14',
    clip_layers=(6, 12, 18, 24),
    k_shot=1,
    image_size=448,
    mask_dir='',  # Set to pre-computed mask path for full C3 support
    clip_weight=1.0,
    dinov2_weight=1.0,
    vl_weight=1.0,
    gaussian_sigma=0.0,
)

optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3, weight_decay=1e-5))
param_scheduler = []

# Few-shot: 1 epoch to collect reference images, then fit()
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)

train_dataloader = dict(
    batch_size=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(pipeline=train_pipeline),
)
val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
