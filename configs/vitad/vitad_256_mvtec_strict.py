# ViTAD MUAD config: multi-class unified anomaly detection (all 15 MVTec categories)
# ADer reference: 98.3% image AUROC (MUAD 100ep)
_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/mvtec_ad.py']

data_root = 'data/mvtec_ad'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
benchmark_multi_class = True
benchmark_keep_dataloader_workers = True
train_pipeline = [
    dict(type='LoadImage', backend='pil'),
    dict(type='LoadMask', backend='pil'),
    dict(type='ResizeAD', size=256, backend='pillow', official_pil=True),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]
test_pipeline = [
    dict(type='LoadImage', backend='pil'),
    dict(type='LoadMask', backend='pil'),
    dict(type='ResizeAD', size=256, backend='pillow', official_pil=True),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]
# multi_class=True loads all 15 categories together (~3629 training images)
train_dataloader = dict(batch_size=8, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='train',
                 multi_class=True, shuffle_train_data=True,
                 pipeline=train_pipeline))
test_dataloader = dict(batch_size=8, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='MVTecADDataset', data_root=data_root, split='test',
                 multi_class=True, pipeline=test_pipeline))
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(type='ViTADDetector',
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
    encoder_name='vit_small_patch16_224_dino',
    teachers=(3, 6, 9),
    neck=(12,),
    students=(3, 6, 9),
    decoder_depth=9,
    fusion_mul=1,
    gaussian_sigma=4.0,
)

# ADer: AdamW lr=1e-4, wd=1e-4, clip_grad=5.0, StepLR(step=80, gamma=0.1)
optim_wrapper = dict(
    constructor='baoiad.ViTADOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=5.0),
)
param_scheduler = [
    dict(type='StepLR', step_size=80, gamma=0.1, by_epoch=True),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
