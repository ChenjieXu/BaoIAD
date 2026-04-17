# VisA evaluation config: inherits from nsa_rn18_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/nsa/nsa_rn18_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/nsa.json

_base_ = ['./nsa_rn18_256_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

data_root = 'data/visa'

# Keep NSATrainDataset for training (it now supports VisA directory structure)
# Only override data_root and cls_names
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
    ))

# Test/val: use VisADataset
test_dataloader = dict(
    dataset=dict(
        type='VisADataset',
        data_root=data_root,
        split='test',
        multi_class=True,
        _delete_=True,
        cls_names=None,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='LoadMask'),
            dict(type='ResizeAD', size=256),
            dict(type='NSATestTransform'),
            dict(type='ScaleNormalizeAD'),
            dict(type='PackADInputs'),
        ],
    ))
val_dataloader = test_dataloader
