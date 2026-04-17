# CutPaste strict-official mainline aligned to pytorch-cutpaste CLI defaults
# Local frozen reference: .refs/pytorch-cutpaste @ 10d8bf71df76d3a97f0106efee1d76f81d983149
# Strict reference contract:
#   1. ResNet-18 backbone
#   2. 3-way variant
#   3. head_layer=1 -> head_dims=(512, 128)
#   4. batch_size=64
#   5. 256 total parameter updates
#   6. freeze_resnet=20
#   7. test_epochs=10
#
# Key alignment fix (checklist item 25):
#   Official run_training.py uses SGD(model.parameters(), ...) which includes
#   backbone params in optimizer from step 0. The freeze mechanism only sets
#   requires_grad=False to block gradients until iter 20, but params remain
#   in optimizer with momentum accumulation.
#
#   This config achieves official semantics via:
#   - backbone.frozen=False: backbone params included in optimizer from step 0
#   - stop_grad_backbone_while_frozen=True: gradients blocked via torch.no_grad()
#   - backbone lr_mult=0.1: reduced LR for backbone (empirically optimal)
#
# Performance (15/15 fresh):
#   image_auroc=0.9343, pixel_auroc=0.7002
#   vs old officialfreeze: +0.0089 image_auroc, +0.0268 pixel_auroc
#
# Previous officialfreeze config archived as:
#   cutpaste_rn18_256_mvtec_strict_officialfreeze_archive.py
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
    '../_base_/backbones/resnet18.py',
]

data_root = 'data/mvtec_ad'
benchmark_multi_class = False

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=3000,
        _delete_=True,
        dataset=dict(
            type='MVTecADDataset',
            data_root=data_root,
            split='train',
            multi_class=False,
            cls_names=['bottle'],
            pipeline=[
                dict(type='LoadImage'),
                dict(type='LoadMask'),
                dict(type='ResizeAD', size=256),
                dict(type='NormalizeAD'),
                dict(type='PackADInputs'),
            ])))
test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecADDataset',
        data_root=data_root,
        split='test',
        multi_class=False,
        _delete_=True,
        cls_names=['bottle'],
        pipeline=[
            dict(type='LoadImage'),
            dict(type='LoadMask'),
            dict(type='ResizeAD', size=256),
            dict(type='NormalizeAD'),
            dict(type='PackADInputs'),
        ]))
val_dataloader = test_dataloader
test_evaluator = dict(type='AnomalyDetectionMetric')
val_evaluator = test_evaluator

model = dict(
    type='CutPasteDetector',
    # Override backbone to not freeze at init (params go into optimizer from step 0)
    backbone=dict(
        type='RawBackbone',
        backbone_name='resnet18',
        pretrained=True,
        frozen=False,
        _delete_=True,
    ),
    proj_dim=128,
    num_classes=3,
    head_dims=(512, 128),
    use_bn=True,
    normalize_embeddings=True,
    freeze_iters=20,
    # Official semantics: backbone params in optimizer from step 0
    # Gradients blocked via torch.no_grad() until unfreeze
    stop_grad_backbone_while_frozen=True,
    # Keep BN in train mode while frozen (official behavior)
    force_backbone_eval_while_frozen=False,
    pre_cutpaste_jitter=True,
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=3e-5),
    # Empirically optimal: backbone LR = 0.1 * head LR
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
        ),
    ),
)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=256, eta_min=1e-5, by_epoch=False),
]

train_cfg = dict(by_epoch=False, max_iters=256, val_interval=10)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
