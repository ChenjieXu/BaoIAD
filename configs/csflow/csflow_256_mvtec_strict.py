_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/mvtec_ad.py',
]

# Strict CSFlow alignment — aligned against original paper repo (marco-rudolph/cs-flow).
# Reference: "Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection" (WACV 2022)
# Reference repo: .refs/cs-flow/ (original paper code)
#
# Key hyperparameters match original config.py + train.py:
# - Adam(lr=2e-4, eps=1e-4, weight_decay=1e-5) with DEFAULT betas (0.9, 0.999)
# - Gradient clipping max_norm=1.0 (c.max_grad_norm)
# - NO LR scheduler (constant lr throughout training)
# - 4 meta_epochs × 60 sub_epochs = 240 total epochs
# - Evaluation every 60 epochs (= 1 meta_epoch)
# - batch_size=16 (original default)
#
# Intentional diffs from original paper repo:
# - Input resolution: 256×256 (online feature extraction) vs original 768×768 (pre-extracted features)
# - Feature extractor: torchvision EfficientNet-B5 vs efficientnet_pytorch package
# - Feature map spatial size: 8×8/4×4/2×2 vs original 24×24/12×12/6×6
# - These diffs are necessary for unified benchmark compatibility
#
# Primary metric: image_auroc (paper reports image-level detection only, no pixel AUROC)

benchmark_result_selector = dict(mode='best', metric='image_auroc')
benchmark_preserve_checkpoint_hooks = True
benchmark_keep_dataloader_workers = True
benchmark_resume_existing = True
benchmark_test_after_train = True
benchmark_checkpoint_source = 'best'
benchmark_timeout = 14400


custom_hooks = [
    dict(type='MemoryBankHook'),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=60,
        max_keep_ckpts=3,
        save_last=True,
        save_best='ad/image_auroc',
        rule='greater',
    ),
)

model = dict(
    type='CSFlowDetector',
    input_size=(256, 256),
    n_coupling_blocks=4,
    cross_conv_hidden_channels=1024,
    clamp=3,
)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4, eps=1e-4, weight_decay=1e-5),
    clip_grad=dict(max_norm=1.0),
)

# No LR scheduler — original uses constant lr throughout training

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=False,
)

val_dataloader = dict(
    num_workers=8,
    persistent_workers=False,
)

test_dataloader = dict(
    num_workers=8,
    persistent_workers=False,
)

train_cfg = dict(by_epoch=True, max_epochs=240, val_interval=60)
val_cfg = dict()
test_cfg = dict(type='ADTestLoop')
