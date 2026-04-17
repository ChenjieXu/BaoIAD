_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

# Strict UFlow alignment target:
# - original repo `mtailanian/uflow`
# - mcait backbone at 448x448
# - 200 training epochs
# - category-specific train batch size and learning rate
# - best checkpoint selected by pixel AUROC
benchmark_keep_dataloader_workers = True
benchmark_preserve_checkpoint_hooks = True
benchmark_result_selector = dict(mode='best', metric='pixel_auroc')

benchmark_category_cfg_options = dict(
    bottle=[
        'train_dataloader.batch_size=23',
        'optim_wrapper.optimizer.lr=0.00011289990475381853',
    ],
    cable=[
        'train_dataloader.batch_size=14',
        'optim_wrapper.optimizer.lr=0.001616039111268198',
    ],
    capsule=[
        'train_dataloader.batch_size=14',
        'optim_wrapper.optimizer.lr=0.0012118892498142058',
    ],
    carpet=[
        'train_dataloader.batch_size=13',
        'optim_wrapper.optimizer.lr=3.163419554150971e-05',
    ],
    grid=[
        'train_dataloader.batch_size=12',
        'optim_wrapper.optimizer.lr=3.6224837438852e-05',
    ],
    hazelnut=[
        'train_dataloader.batch_size=21',
        'optim_wrapper.optimizer.lr=0.0013268898640895934',
    ],
    leather=[
        'train_dataloader.batch_size=15',
        'optim_wrapper.optimizer.lr=0.0006124723659154866',
    ],
    metal_nut=[
        'train_dataloader.batch_size=11',
        'optim_wrapper.optimizer.lr=0.0008148858311570219',
    ],
    pill=[
        'train_dataloader.batch_size=11',
        'optim_wrapper.optimizer.lr=0.0010756100354690954',
    ],
    screw=[
        'train_dataloader.batch_size=11',
        'optim_wrapper.optimizer.lr=0.0004155987052838602',
    ],
    tile=[
        'train_dataloader.batch_size=30',
        'optim_wrapper.optimizer.lr=0.006045754779719109',
    ],
    toothbrush=[
        'train_dataloader.batch_size=21',
        'optim_wrapper.optimizer.lr=0.0001287312814913699',
    ],
    transistor=[
        'train_dataloader.batch_size=20',
        'optim_wrapper.optimizer.lr=0.00112129043027424',
    ],
    wood=[
        'train_dataloader.batch_size=22',
        'optim_wrapper.optimizer.lr=0.0002466546120460351',
    ],
    zipper=[
        'train_dataloader.batch_size=24',
        'optim_wrapper.optimizer.lr=4.55246818236177e-05',
    ],
)

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

custom_hooks = [
    dict(type='UFlowStrictTrainHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='ad/pixel_auroc',
        rule='greater',
        patience=20,
        min_delta=0.0,
        strict=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
        save_best='ad/pixel_auroc',
        rule='greater',
    ),
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(pipeline=train_pipeline),
)
test_dataloader = dict(
    batch_size=5,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(pipeline=test_pipeline),
)
val_dataloader = test_dataloader

model = dict(
    type='UFlowDetector',
    input_size=(448, 448),
    flow_steps=4,
    backbone='mcait',
    affine_clamp=2.0,
    affine_subnet_channels_ratio=1.0,
    permute_soft=False,
    compute_nfa_in_predict=False,
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-5),
)

param_scheduler = [
    # Placeholder end; UFlowStrictTrainHook rewrites this to the official
    # `len(train_dataloader) * epochs` iteration budget before training.
    dict(type='LinearLR', start_factor=1.0, end_factor=0.4, begin=0, end=2, by_epoch=False),
]

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
