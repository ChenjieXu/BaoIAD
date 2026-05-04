# VisA evaluation config: inherits from musc_vitl14_336_518_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/musc/musc_vitl14_336_518_visa.py \
#           --data_root data/visa --categories all --output runs/visa/musc.json

_base_ = ['./musc_vitl14_336_518_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
