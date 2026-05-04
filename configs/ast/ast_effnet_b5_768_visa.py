# VisA evaluation config: inherits from ast_effnet_b5_768_mvtec_strict.py, overrides dataset to VisA
# Usage: python tools/benchmark.py --config configs/ast/ast_effnet_b5_768_visa.py \
#           --data_root data/visa --categories all --output runs/visa/ast.json

_base_ = ['./ast_effnet_b5_768_mvtec_strict.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

# Override dataset to VisA
train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
