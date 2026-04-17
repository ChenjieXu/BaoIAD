_base_ = [
    '../_base_/default_runtime.py',
]

img_size = 518
benchmark_keep_train_data_root = True
benchmark_preserve_checkpoint_hooks = True
randomness = dict(seed=111, deterministic=False)
custom_hooks = []

model = dict(
    type='AACLIPDetector',
    clip_model='ViT-L-14-336',
    pretrained='openai',
    image_size=img_size,
    training_stage='text',
    reference_root='.refs/AA-CLIP',
    text_norm_weight=0.1,
    text_adapt_weight=0.1,
    image_adapt_weight=0.1,
    text_adapt_until=3,
    image_adapt_until=6,
    surgery_until_layer=20,
    levels=[6, 12, 18, 24],
    relu=False,
    default_dataset_name='VisA',
    temperature=0.07,
)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='AACLIPJsonDataset',
        data_root='data/visa',
        metadata_path='runs/alignment/aaclip_shot0_reproduce/metadata/visa_32shot_seed111.jsonl',
        dataset_name='VisA',
        img_size=img_size,
        multi_class=True,
        text_mode=True,
        augment=True,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AACLIPJsonDataset',
        data_root='data/mvtec_ad',
        metadata_path='.refs/AA-CLIP/dataset/metadata/MVTec/full-shot.jsonl',
        dataset_name='MVTec',
        img_size=img_size,
        multi_class=True,
        text_mode=False,
        augment=False,
    ),
)

test_dataloader = val_dataloader
val_evaluator = [
    dict(
        type='AACLIPOfficialMetric',
        metrics=['image_auroc', 'pixel_auroc'],
    ),
]
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-5, betas=(0.5, 0.999), weight_decay=0.0),
)

param_scheduler = []
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_last=True),
)

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=5)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
