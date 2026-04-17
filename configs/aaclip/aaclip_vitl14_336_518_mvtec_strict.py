"""AA-CLIP strict official-alignment config.

Reference freeze:
- Official repository: `Mwxinnn/AA-CLIP`
- Commit: `53db195f230442aa118c246876c94ba1c76139cc`
- Runtime authority: `train.py`, `test.py`, `forward_utils.py`
- Official protocol: two-stage VisA training, then MVTec evaluation

This config is the strict MVTec evaluation identity for benchmark/probe use.
`benchmark_eval_only=True` keeps benchmark.py on the official evaluation path.
The train-time fields intentionally mirror the official stage-2 image-adapter
hyper-parameters so `tools/train.py` can still run a local smoke fine-tune when
needed. `use_fast_build=False` is also intentional here: the auxiliary
OpenCLIP fast-load path is not part of the official runtime and is not used as
strict-alignment evidence.
"""

_base_ = [
    '../_base_/default_runtime.py',
]

data_root = 'data/mvtec_ad'
img_size = 518

benchmark_multi_class = True
benchmark_eval_only = True
benchmark_keep_train_data_root = True
benchmark_preserve_checkpoint_hooks = True

randomness = dict(seed=111, deterministic=False)
custom_hooks = []

model = dict(
    type='AACLIPDetector',
    clip_model='ViT-L-14-336',
    model_name='ViT-L-14-336',
    pretrained='openai',
    image_size=img_size,
    training_stage='image',
    reference_root='.refs/AA-CLIP',
    text_adapter_ckpt='.refs/AA-CLIP/ckpt/joint_text_adapter.pth',
    image_adapter_ckpt='.refs/AA-CLIP/ckpt/joint_image_adapter_15.pth',
    text_norm_weight=0.1,
    text_adapt_weight=0.1,
    image_adapt_weight=0.1,
    text_adapt_until=3,
    image_adapt_until=6,
    surgery_until_layer=20,
    levels=[6, 12, 18, 24],
    relu=False,
    default_dataset_name='MVTec',
    temperature=0.07,
    use_fast_build=False,
    require_official_assets=True,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='AACLIPJsonDataset',
        data_root=data_root,
        metadata_path='.refs/AA-CLIP/dataset/metadata/MVTec/full-shot.jsonl',
        dataset_name='MVTec',
        img_size=img_size,
        multi_class=True,
        text_mode=False,
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
        data_root=data_root,
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
        metrics=['image_auroc', 'pixel_auroc', 'image_ap', 'pixel_ap'],
    ),
]
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4, betas=(0.5, 0.999), weight_decay=0.0),
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        milestones=[16000, 32000],
        gamma=0.5,
        by_epoch=False,
    ),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=20, save_last=True),
)

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
