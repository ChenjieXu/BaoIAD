# VisA evaluation config: inherits from anovl_vitb16plus_240_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/anovl/anovl_vitb16plus_240_visa.py \
#           --data_root data/visa --categories all --output runs/visa/anovl.json

_base_ = ['./anovl_vitb16plus_240_mvtec_strict.py']

# Benchmark mode: MC
benchmark_multi_class = True

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
