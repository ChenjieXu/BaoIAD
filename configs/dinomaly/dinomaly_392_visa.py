# VisA evaluation config: inherits from dinomaly_392_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/dinomaly/dinomaly_392_visa.py \
#           --data_root data/visa --categories all --output runs/visa/dinomaly.json

_base_ = ['./dinomaly_392_mvtec_strict.py']

# Benchmark mode: MC
benchmark_multi_class = True

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True, cls_names=None))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True, cls_names=None))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True, cls_names=None))
