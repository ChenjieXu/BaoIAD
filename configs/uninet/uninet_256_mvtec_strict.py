_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

# anomalib UniNet mainline:
# - wide_resnet50_2 teacher/student
# - AdamW(lr=5e-3, wd=1e-5, eps=1e-10, amsgrad=True)
# - target teacher lr=1e-6
# - MultiStepLR milestone at 80% of the 100-epoch budget
# - early stopping on image AUROC with patience 20
benchmark_multi_class = False
benchmark_keep_dataloader_workers = False
benchmark_preserve_checkpoint_hooks = True
benchmark_result_selector = dict(mode='best', metric='image_auroc')

custom_hooks = [
    dict(type='MemoryBankHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='ad/image_auroc',
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
        save_best='ad/image_auroc',
        rule='greater',
    ),
)

model = dict(
    type='UniNetDetector',
    teacher_backbone=dict(
        type='FeatureExtractor',
        backbone_name='wide_resnet50_2',
        pretrained=True,
        out_indices=(1, 2, 3),
        frozen=False,
    ),
    lambda_weight=0.7,
    temperature=0.1,
)

# anomalib MVTecAD defaults to train/eval batch size 32 with 8 workers.
# Resource-constrained smoke runs may still override these via CLI.
train_dataloader = dict(batch_size=32, num_workers=8)
test_dataloader = dict(batch_size=32, num_workers=8)
val_dataloader = test_dataloader

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
        eps=1e-10,
        amsgrad=True,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'teachers.target_teacher': dict(lr_mult=2e-4),
            # anomalib's optimizer only updates student/bottleneck/dfs/target_teacher.
            'fc': dict(lr_mult=0.0, decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(type='MultiStepLR', milestones=[80], gamma=0.2, by_epoch=True),
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
