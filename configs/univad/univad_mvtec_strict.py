_base_ = ['./univad_mvtec.py']

model = dict(
    strict_mode=True,
    require_mask_dir=True,
    require_heat_mask_dir=True,
    require_query_masks=True,
    mask_dir='data/mvtec_ad_univad_assets/masks',
    heat_mask_dir='data/mvtec_ad_univad_assets/heat_masks',
    # Gate overrides for strict alignment with official implementation
    # C3 mask assets have incorrect object_ratio, causing wrong gate classification
    # Reference: empty masks → object_ratio=1 → TEXTURE
    gate_overrides={
        # TEXTURE categories (texture-like surfaces OR empty/no-foreground masks)
        'carpet': 'texture',
        'grid': 'texture',
        'leather': 'texture',
        'tile': 'texture',
        'wood': 'texture',
        'zipper': 'texture',
        'screw': 'texture',      # Empty mask → TEXTURE
        'toothbrush': 'texture',  # Empty mask → TEXTURE
    },
)
