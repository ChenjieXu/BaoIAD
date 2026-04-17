# DeSTSeg strict config aligned with the official ResNet-18 implementation.
_base_ = ['../_base_/default_runtime.py']

data_root = 'data/mvtec_ad'
img_size = 256
dtd_path = 'auto'

train_pipeline = [
    dict(type='LoadImage', backend='pil'),
    dict(type='ResizeAD', size=img_size),
    dict(type='DeSTSegAugment', dtd_path=dtd_path),
    dict(type='PackDeSTSegInputs'),
]

test_pipeline = [
    dict(type='LoadImage', backend='pil'),
    dict(type='LoadMask', backend='pil', to_binary=False),
    dict(type='ResizeAD', size=img_size, backend='pillow', mask_interpolation='bilinear'),
    dict(type='ThresholdMask', threshold=0.5),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='PersistentShuffleSampler', shuffle=True, seed=42, round_up=False),
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
    batch_size=32,
    num_workers=16,
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
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='DeSTSegDetector',
    backbone='resnet18',
    teacher_pretrained=True,
    de_st_steps=1000,
    phase_ratio=0.2,
    top_k_score=100,
    gamma=4.0,
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

optim_wrapper = dict(
    constructor='baoiad.DeSTSegOptimWrapperConstructor',
    student=dict(
        optimizer=dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=1e-4),
    ),
    segmentation=dict(
        optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
        res_lr=0.1,
        head_lr=0.01,
    ),
)

train_cfg = dict(by_epoch=False, max_iters=5000, val_interval=5000)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')

randomness = dict(seed=42, deterministic=False)
train_disable_compile = True
benchmark_disable_compile = True
benchmark_keep_dataloader_workers = True
