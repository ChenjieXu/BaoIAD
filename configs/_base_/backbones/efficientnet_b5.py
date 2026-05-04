backbone = dict(
    type='TIMMBackbone',
    model_name='tf_efficientnet_b5',
    pretrained=True,
    features_only=False,
    frozen=True,
    data_preprocessor=dict(type='mmengine.model.ImgDataPreprocessor'),
)
