"""PyramidFlow strict official-alignment config for the published FNF baseline.

Reference freeze:
- CVPR 2023 paper + supplementary material
- Official GitHub repository URL currently unavailable (404 as of 2026-03-27)
- Code-path proxy for implementation audit: `.refs/ader`
"""
_base_ = [
    '../_base_/default_runtime.py',
]

data_root = 'data/mvtec_ad'
img_size = 256
benchmark_multi_class = False
benchmark_keep_dataloader_workers = True
benchmark_result_selector = dict(mode='last')
benchmark_summary_categories = [
    'bottle', 'cable', 'capsule', 'carpet', 'hazelnut', 'leather',
    'pill', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]
benchmark_category_cfg_options = dict(
    carpet=['model.vn_dims=[0,2,3]'],
    grid=['model.vn_dims=[0,2,3]'],
    leather=['model.vn_dims=[0,2,3]'],
    tile=['model.vn_dims=[0,2,3]'],
    wood=['model.vn_dims=[0,2,3]'],
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=img_size),
    dict(type='PyramidFlowStrictTrainTransform', flip_p=0.5, rotation_p=0.5, rotation_degrees=180.0),
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

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='train',
        cls_names=['bottle'],
        multi_class=False,
        shuffle_train_data=True,
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
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
    image_score_field='pred_score_max',
)
val_evaluator = test_evaluator

model = dict(
    type='PyramidFlowDetector',
    encoder=None,
    channel=64,
    num_level=4,
    num_stack=4,
    ksize=7,
    vn_dims=(0, 1),
    save_memory=False,
    pyramid_downsample_mode='maxpool',
    template_pipeline=test_pipeline,
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4, eps=1e-4, betas=(0.5, 0.9), weight_decay=1e-5),
    clip_grad=dict(max_norm=1.0),
)

param_scheduler = []
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=100)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')
