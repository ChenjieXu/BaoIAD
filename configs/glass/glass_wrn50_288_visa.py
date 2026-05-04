# VisA evaluation config: inherits from glass_wrn50_288_mvtec_strict.py, overrides for VisA
# Usage: python tools/benchmark.py --config configs/glass/glass_wrn50_288_visa.py \
#           --data_root data/visa --categories all --output runs/visa/glass.json

_base_ = ['./glass_wrn50_288_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

data_root = 'data/visa'

# Use VisADataset for both train and test since VisA lacks GLASS-specific assets.
# The legacy (non-strict) training path generates anomalies internally.
train_dataloader = dict(
    dataset=dict(
        type='VisADataset',
        data_root=data_root,
        split='train',
        multi_class=True,
        _delete_=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='LoadMask'),
            dict(type='ResizeAD', size=288),
            dict(type='NormalizeAD'),
            dict(type='PackADInputs'),
        ],
    ))

test_dataloader = dict(
    dataset=dict(
        type='VisADataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        _delete_=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='LoadMask'),
            dict(type='ResizeAD', size=288),
            dict(type='NormalizeAD'),
            dict(type='PackADInputs'),
        ],
    ))
val_dataloader = test_dataloader

# Disable strict mode in model since no distribution assets
model = dict(
    strict=False,
    distribution_meta_path=None,
)

# Override train_cfg to use standard loop (not GLASSTrainLoop which needs strict assets)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ADValLoop')
test_cfg = dict(type='ADTestLoop')
