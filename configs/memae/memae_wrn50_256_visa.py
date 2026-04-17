# VisA evaluation config for MemAE
_base_ = ['./memae_wrn50_256_mvtec.py']

# Benchmark mode: SC (per-category)
benchmark_multi_class = False

train_dataloader = dict(
    num_workers=4,
    persistent_workers=True,
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True, cls_names=None))
test_dataloader = dict(
    num_workers=4,
    persistent_workers=True,
    dataset=dict(type='VisADataset', data_root='data/visa', multi_class=True, cls_names=None))
val_dataloader = test_dataloader
