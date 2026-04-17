# VisA evaluation config: inherits from memseg_rn18_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/memseg/memseg_rn18_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/memseg.json

_base_ = ['./memseg_rn18_256_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True, cls_names=None))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True, cls_names=None))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True, cls_names=None))
