# VisA evaluation config: inherits from anomalydino_vitb14_448_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/anomalydino/anomalydino_vitb14_448_visa.py \
#           --data_root data/visa --categories all --output runs/visa/anomalydino.json

_base_ = ['./anomalydino_vitb14_448_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
