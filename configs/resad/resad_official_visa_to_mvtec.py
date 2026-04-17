"""Official ResAD protocol: train on VisA and evaluate on MVTec AD.

This config is consumed by ``tools/resad_official_eval.py`` rather than the
standard MMEngine runner. It is the canonical strict mainline for ResAD.
"""

ref_repo_root = '.refs/ResAD'

setting = 'visa_to_mvtec'
train_dataset_dir = 'data/VisA_20220922'
test_dataset_dir = 'data/mvtec_ad'
test_ref_feature_dir = 'pretrained/resad/ref_features/w50/mvtec_4shot'
test_ref_few_shot_dir = 'pretrained/resad/data/4shot/mvtec'
test_ref_feature_archive = 'pretrained/resad/ResAD-data.zip'

work_dir = 'runs/resad_official_visa_to_mvtec'
checkpoint_path = 'runs/resad_official_visa_to_mvtec/checkpoints'

device = 'cuda:0'
batch_size = 32
lr = 1e-5
epochs = 100
eval_freq = 1
cache_train_ref_tensors = True
cache_train_ref_preload = False

backbone = 'wide_resnet50_2'
flow_arch = 'conditional_flow_model'
feature_levels = 3
coupling_layers = 10
clamp_alpha = 1.9
pos_embed_dim = 256
pos_beta = 0.05
margin_tau = 0.1
bgspp_lambda = 1.0
fdm_alpha = 0.4
num_embeddings = 1536
train_ref_shot = 4
num_ref_shot = 4

This config is consumed by ``tools/resad_official_eval.py`` rather than the
standard MMEngine runner. It is the canonical strict mainline for ResAD.
"""

ref_repo_root = '.refs/ResAD'

setting = 'visa_to_mvtec'
train_dataset_dir = 'data/VisA_20220922'
test_dataset_dir = 'data/mvtec_ad'
test_ref_feature_dir = 'pretrained/resad/ref_features/w50/mvtec_4shot'
test_ref_few_shot_dir = 'pretrained/resad/data/4shot/mvtec'
test_ref_feature_archive = 'pretrained/resad/ResAD-data.zip'

work_dir = 'runs/resad_official_visa_to_mvtec'
checkpoint_path = 'runs/resad_official_visa_to_mvtec/checkpoints'

device = 'cuda:0'
batch_size = 32
lr = 1e-5
epochs = 100
eval_freq = 1
cache_train_ref_tensors = True
cache_train_ref_preload = False

backbone = 'wide_resnet50_2'
flow_arch = 'conditional_flow_model'
feature_levels = 3
coupling_layers = 10
clamp_alpha = 1.9
pos_embed_dim = 256
pos_beta = 0.05
margin_tau = 0.1
bgspp_lambda = 1.0
fdm_alpha = 0.4
num_embeddings = 1536
train_ref_shot = 4
num_ref_shot = 4
