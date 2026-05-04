# VisA evaluation config for WinCLIP
_base_ = ['./winclip_256_mvtec.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

train_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
test_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
val_dataloader = dict(
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True))
