# VisA evaluation config: inherits from rd_wrn50_256_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/rd/rd_wrn50_256_visa.py \
#           --data_root data/visa --categories all --output runs/visa/rd.json

_base_ = ['./rd_wrn50_256_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
