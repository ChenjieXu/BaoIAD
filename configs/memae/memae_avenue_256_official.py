_base_ = ['./memae_ucsdped2_256_official.py']

data_root = 'data/memae_video/Avenue_256'
dataset_name = 'Avenue'

train_dataloader = dict(dataset=dict(data_root=data_root, dataset_name=dataset_name))
test_dataloader = dict(dataset=dict(data_root=data_root, dataset_name=dataset_name))
val_dataloader = dict(dataset=dict(data_root=data_root, dataset_name=dataset_name))
