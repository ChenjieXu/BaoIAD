# VisA evaluation config: inherits from uflow_mcait_448_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/uflow/uflow_mcait_448_visa.py \
#           --data_root data/visa --categories all --output runs/visa/uflow.json

_base_ = ['./uflow_mcait_448_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
