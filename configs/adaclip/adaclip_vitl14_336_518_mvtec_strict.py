"""AdaCLIP strict official-alignment config.

Reference freeze:
- Official repository: `caoyunkang/AdaCLIP`
- Commit: `b762ac40c3f33c77e7e513e48cb436f059d456da`
- Runtime authority: `train.py` / `train.sh`
- Official MVTec protocol: train on `VisA + ClinicDB`, evaluate on `MVTec AD`
- Checkpoint-only eval remains available in `adaclip_vitl14_336_256_mvtec.py`
"""

_base_ = [
    '../_base_/default_runtime.py',
]

benchmark_multi_class = True
benchmark_keep_train_data_root = True
benchmark_preserve_checkpoint_hooks = True
benchmark_result_selector = dict(
    mode='best',
    metric='pixel_f1max',
    tie_break_metric='image_auroc',
)

img_size = 518
randomness = dict(seed=111, deterministic=False)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=(img_size, img_size)),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='LoadMask'),
    dict(type='ResizeAD', size=(img_size, img_size)),
    dict(type='NormalizeAD'),
    dict(type='PackADInputs'),
]

model = dict(
    type='AdaCLIPDetector',
    clip_model='ViT-L-14-336',
    pretrained='openai',
    image_size=518,
    features_list=[6, 12, 18, 24],
    prompting_depth=4,
    prompting_length=5,
    prompting_branch='VL',
    prompting_type='SD',
    use_hsf=True,
    k_clusters=20,
    temperature=0.07,
    gaussian_sigma=4.0,
    official_checkpoint=None,
    require_official_checkpoint=False,
    # Official MVTec protocol trains prompts on auxiliary labeled data.
    enable_train_loss=True,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='AdaCLIPVisADataset',
                data_root='data/visa',
                split='test',
                multi_class=True,
                pipeline=train_pipeline,
            ),
            dict(
                type='AdaCLIPClinicDBDataset',
                data_root='data/clinicdb',
                split='test',
                multi_class=True,
                pipeline=train_pipeline,
            ),
        ],
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root='data/mvtec_ad',
        split='test',
        multi_class=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader
val_evaluator = [
    dict(
        type='AnomalyDetectionMetric',
        metrics=[
            'image_auroc',
            'pixel_auroc',
            'image_f1max',
            'pixel_f1max',
            'image_ap',
            'pixel_ap',
            'aupro',
        ],
    ),
]
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.01, betas=(0.5, 0.999), weight_decay=0.0),
)

# Official AdaCLIP train.py does not define an LR scheduler.
param_scheduler = []

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_last=True,
        save_best='ad/pixel_f1max',
        rule='greater',
    ),
)

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
