# VisA evaluation config: inherits from ganomaly_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/ganomaly/ganomaly_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/ganomaly.json

_base_ = ['./ganomaly_256_mvtec_strict.py']

# GANomaly is single-class: train per-category
benchmark_multi_class = False

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
