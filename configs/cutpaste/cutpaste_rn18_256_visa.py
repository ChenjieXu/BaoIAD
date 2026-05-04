# VisA evaluation config: inherits from cutpaste_rn18_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/cutpaste/cutpaste_rn18_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/cutpaste.json

_base_ = ['./cutpaste_rn18_256_mvtec_strict.py']

data_root = 'data/visa'

# CutPaste is single-class: train per-category
benchmark_multi_class = False

# Override train dataloader: RepeatDataset wrapping VisADataset
# Must use _delete_=True to avoid leaking RepeatDataset keys from base config
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
            type='VisADataset',
            data_root=data_root,
            split='train',
            multi_class=True,
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
        type='VisADataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        _delete_=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='LoadMask'),
            dict(type='ResizeAD', size=256),
            dict(type='NormalizeAD'),
            dict(type='PackADInputs'),
        ]))
val_dataloader = test_dataloader
