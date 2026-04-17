backbone = dict(
    type='TIMMBackbone',
    model_name='tf_efficientnet_b4',
    pretrained=True,
    features_only=True,
    out_indices=(4,),  # Use last layer for CutPaste
    frozen=True,
)
