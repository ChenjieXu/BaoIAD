# VisA evaluation config: inherits from dfkde_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/dfkde/dfkde_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/dfkde.json

_base_ = ['./dfkde_256_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
