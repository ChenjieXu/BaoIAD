# VisA evaluation config: inherits from cfa_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/cfa/cfa_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/cfa.json

_base_ = ['./cfa_256_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
