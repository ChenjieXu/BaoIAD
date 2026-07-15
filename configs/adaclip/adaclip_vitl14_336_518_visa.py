# AdaCLIP: Zero-shot evaluation on VisA dataset
# Uses the checkpoint filename referenced by the official AdaCLIP scripts for
# VisA evaluation: `weights/pretrained_mvtec_colondb.pth`.
#
# Usage:
#   python tools/test.py configs/adaclip/adaclip_vitl14_336_518_visa.py \
#       --work-dir runs/adaclip_visa_zeroshot

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/datasets/visa.py",
]

# Benchmark mode: MC
benchmark_multi_class = True

# Override image size for VisA (official uses 518)
img_size = 518

# Official AdaCLIP evaluation fixes the random seed to 111.
randomness = dict(seed=111, deterministic=False)

# Rebuild pipelines with correct image size
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
    type="AdaCLIPDetector",
    clip_model="ViT-L-14-336",
    pretrained="openai",
    image_size=518,  # Official default
    features_list=[6, 12, 18, 24],  # Multi-hierarchy feature extraction
    prompting_depth=4,  # Official default
    prompting_length=5,  # Official default
    prompting_branch="VL",  # V=visual, L=language, VL=both (official default)
    prompting_type="SD",  # S=static, D=dynamic, SD=both (official default)
    use_hsf=True,  # Hybrid Semantic Fusion (official default)
    k_clusters=20,  # K-means clusters for HSF (official default)
    temperature=0.07,
    gaussian_sigma=4.0,  # Official uses gaussian_filter with sigma=4
    official_checkpoint="pretrained/pretrained_mvtec_colondb.pth",
    require_official_checkpoint=True,
)

# Prompt learning: AdamW with official LR (0.01 as per official implementation)
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=0.01, betas=(0.5, 0.999), weight_decay=0.0),
)

# Cosine annealing scheduler (5 epochs as per official implementation)
param_scheduler = [
    dict(type="CosineAnnealingLR", T_max=5, eta_min=1e-6, by_epoch=True),
]

# Training config for zero-shot mode (minimal training)
train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)

# Batch size must be 1 (official limitation)
train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
