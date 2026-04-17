"""Tests for UniVADDetector."""

import importlib.util
import numpy as np
import pytest
import torch
from PIL import Image
from unittest import TestCase

from baoiad.structures import ADDataSample

import baoiad  # noqa: F401
from baoiad.models.detectors.univad import (
    CFA,
    ObjectType,
    UniVADDetector,
    _create_prompt_ensemble,
    _supports_out_layers_api,
    _vl_clip_layer_index,
)
from baoiad.registry import MODELS


def _make_data_samples(batch_size, H=56, W=56, cls_name='bottle'):
    samples = []
    for i in range(batch_size):
        s = ADDataSample()
        s.gt_label = i % 2
        s.gt_mask = torch.zeros(H, W)
        s.cls_name = cls_name
        s.img_path = f'/fake/{i:03d}.png'
        s.defect_type = 'good'
        samples.append(s)
    return samples


HAS_OPEN_CLIP = importlib.util.find_spec('open_clip') is not None


def _can_load_dinov2():
    """Check if DINOv2 weights are available."""
    try:
        import timm
        timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True,
                          dynamic_img_size=True)
        return True
    except Exception:
        return False


_SKIP = not HAS_OPEN_CLIP or not _can_load_dinov2()


def test_supports_out_layers_api():
    class LocalClip:
        def encode_image(self, image, out_layers, normalize=False):
            return image, out_layers, normalize

    class PipOpenClip:
        def encode_image(self, image, normalize=False):
            return image, normalize

    assert _supports_out_layers_api(LocalClip().encode_image)
    assert not _supports_out_layers_api(PipOpenClip().encode_image)


def test_vl_clip_layer_index_matches_reference_and_fallback_paths():
    assert _vl_clip_layer_index(8, 4) == 6
    assert _vl_clip_layer_index(4, 4) == 3
    assert _vl_clip_layer_index(1, 4) == 0


def test_extract_clip_component_layers_keeps_local_reference_tokens():
    class LocalClip:
        def encode_image(self, image, out_layers, normalize=False):
            return torch.zeros(1, 2), [
                torch.arange(6, dtype=torch.float32).reshape(1, 3, 2),
                torch.arange(6, 12, dtype=torch.float32).reshape(1, 3, 2),
            ]

    model = object.__new__(UniVADDetector)
    model.clip = LocalClip()
    model.clip_layers = (6, 12)

    layers = model._extract_clip_component_layers(torch.zeros(1, 3, 2, 2))

    assert len(layers) == 2
    assert layers[0].shape == (1, 3, 2)
    assert torch.equal(layers[0], torch.arange(6, dtype=torch.float32).reshape(1, 3, 2))


def test_extract_component_feature_bank_uses_official_reference_extractor():
    class FakeExtractor:
        def extract(self, image, masks):
            return {
                'area': torch.tensor([[0.1]], dtype=torch.float32),
                'color': torch.tensor([[0.2]], dtype=torch.float32),
                'position': torch.tensor([[0.3, 0.4]], dtype=torch.float32),
                'clip_image': torch.tensor([[[0.5, 0.6]]], dtype=torch.float32),
                'dino_image': torch.tensor([[0.7, 0.8]], dtype=torch.float32),
            }

    model = object.__new__(UniVADDetector)
    model.official_scoring = True
    model._official_component_extractor = FakeExtractor()

    features = model._extract_component_feature_bank(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [np.ones((4, 4), dtype=np.uint8) * 255],
        torch.device('cpu'),
    )

    assert features is not None
    assert torch.equal(features['area'], torch.tensor([[0.1]], dtype=torch.float32))
    assert torch.equal(features['clip_image'], torch.tensor([[[0.5, 0.6]]], dtype=torch.float32))


def test_cfa_matches_official_single_layer_graph_aggregation():
    cfa = CFA(n_layers=1)
    node_features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    out = cfa(node_features)
    adj = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    expected = adj @ node_features

    assert torch.allclose(out, expected)


def test_prompt_ensemble_matches_official_templates():
    normal_prompts, anomalous_prompts = _create_prompt_ensemble('metal_nut')

    assert len(normal_prompts) == 7 * 35
    assert len(anomalous_prompts) == 5 * 35
    assert 'a photo of a broken metal nut.' in anomalous_prompts


def test_determine_gate_matches_reference_first_mask_logic():
    model = object.__new__(UniVADDetector)
    model.object_ratio_threshold = 0.65
    model.max_segment_for_texture = 2

    texture_like = np.zeros((10, 10), dtype=np.int32)
    texture_like[:, :8] = 1
    multi_part = np.zeros((10, 10), dtype=np.int32)
    multi_part[:5, :5] = 1
    multi_part[5:, 5:] = 2

    assert model._determine_gate('bottle', [texture_like, multi_part]) == ObjectType.TEXTURE
    assert model._determine_gate('bottle', [multi_part]) == ObjectType.MULTI


@pytest.mark.parametrize('layout', ['npy', 'grounding_png'])
def test_load_single_mask_supports_reference_layouts(tmp_path, layout):
    mask_root = tmp_path / 'masks'
    model = object.__new__(UniVADDetector)
    model.mask_dir = str(mask_root)
    model.image_size = 2

    expected = np.array([[0, 1], [2, 2]], dtype=np.int32)
    image_path = 'data/mvtec_ad/bottle/train/good/000.png'

    if layout == 'npy':
        target = mask_root / 'bottle' / 'train' / 'good' / '000.npy'
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, expected)
    else:
        target = mask_root / 'mvtec_ad' / 'bottle' / 'train' / 'good' / '000' / 'grounding_mask.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(expected.astype(np.uint8)).save(target)

    loaded = model._load_single_mask('bottle', image_path)
    assert loaded is not None
    assert np.array_equal(loaded, expected)


def test_load_heat_mask_supports_path_layout(tmp_path):
    heat_root = tmp_path / 'heat_masks'
    model = object.__new__(UniVADDetector)
    model.heat_mask_dir = str(heat_root)
    model.image_size = 2

    expected = np.array([[0, 1], [2, 2]], dtype=np.int32)
    target = heat_root / 'bottle' / 'test' / 'broken_large' / '000.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(expected.astype(np.uint8)).save(target)

    loaded = model._load_heat_mask('bottle', 'data/mvtec_ad/bottle/test/broken_large/000.png')
    assert loaded is not None
    assert np.array_equal(loaded, expected)


def test_resolve_single_branch_weights_supports_class_overrides():
    model = object.__new__(UniVADDetector)

    resolved = model._resolve_single_branch_weights(
        'metal_nut',
        clip_weight=1.0,
        dinov2_weight=1.0,
        vl_weight=1.0,
        global_weight=1.0,
        part_weight=1.0,
        overrides=dict(
            metal_nut=dict(
                clip_weight=0.5,
                vl_weight=0.25,
                global_weight=0.75,
                part_weight=2.0,
            ),
        ),
    )

    assert resolved == dict(
        clip_weight=0.5,
        dinov2_weight=1.0,
        vl_weight=0.25,
        global_weight=0.75,
        part_weight=2.0,
    )


def test_resolve_multi_pixel_gecm_weight_supports_class_overrides():
    model = object.__new__(UniVADDetector)
    model.multi_pixel_gecm_weight = 1.0
    model.multi_pixel_weight_overrides = dict(
        transistor=dict(gecm_weight=2.5),
    )

    assert model._resolve_multi_pixel_gecm_weight('transistor') == 2.5
    assert model._resolve_multi_pixel_gecm_weight('cable') == 1.0


def test_resolve_multi_image_gecm_weight_supports_class_overrides():
    model = object.__new__(UniVADDetector)
    model.multi_image_gecm_weight = 1.0
    model.multi_image_weight_overrides = dict(
        transistor=dict(gecm_weight=0.0),
    )

    assert model._resolve_multi_image_gecm_weight('transistor') == 0.0
    assert model._resolve_multi_image_gecm_weight('cable') == 1.0


def test_combine_multi_gecm_distance_matches_official_sum_path():
    model = object.__new__(UniVADDetector)
    model.official_scoring = True

    dist, weights = model._combine_multi_gecm_distance(
        'transistor',
        dist_clip=0.2,
        dist_dino=0.3,
        dist_geo=0.4,
    )

    assert dist == pytest.approx(0.9)
    assert weights == dict(clip_weight=1.0, dino_weight=1.0, geo_weight=1.0)


def test_combine_multi_gecm_distance_keeps_weighted_experimental_path():
    model = object.__new__(UniVADDetector)
    model.official_scoring = False
    model.multi_gecm_clip_weight = 1.0
    model.multi_gecm_dino_weight = 2.0
    model.multi_gecm_geo_weight = 3.0
    model.multi_gecm_feature_weight_overrides = {}

    dist, weights = model._combine_multi_gecm_distance(
        'transistor',
        dist_clip=0.2,
        dist_dino=0.3,
        dist_geo=0.4,
    )

    assert dist == pytest.approx((1.0 * 0.2 + 2.0 * 0.3 + 3.0 * 0.4) / 6.0)
    assert weights == dict(clip_weight=1.0, dino_weight=2.0, geo_weight=3.0)


def test_score_map_align_corners_follows_official_scoring_flag():
    model = object.__new__(UniVADDetector)
    model.official_scoring = True
    assert model._score_map_align_corners() is True

    model.official_scoring = False
    assert model._score_map_align_corners() is False


def test_resolve_multi_component_min_ratio_supports_class_overrides():
    model = object.__new__(UniVADDetector)
    model.multi_component_min_ratio = 1e-4
    model.multi_component_min_ratio_overrides = dict(transistor=0.0)

    assert model._resolve_multi_component_min_ratio('transistor') == 0.0
    assert model._resolve_multi_component_min_ratio('cable') == 1e-4


def test_resolve_multi_component_masks_falls_back_to_raw_when_heat_collapses():
    model = object.__new__(UniVADDetector)
    model.official_scoring = False
    model.multi_component_min_ratio = 1e-4
    model.multi_component_min_ratio_overrides = {}

    raw_mask = np.array([
        [1, 1, 0, 2],
        [1, 1, 0, 2],
        [0, 0, 0, 2],
        [0, 0, 0, 2],
    ], dtype=np.int32)
    heat_mask = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    masks, indices = model._resolve_multi_component_masks('transistor', raw_mask, heat_mask)

    assert indices == [1, 2]
    assert len(masks) == 2


def test_resolve_multi_component_masks_uses_dilated_heat_masks_in_official_mode():
    model = object.__new__(UniVADDetector)
    model.official_scoring = True
    model.component_mask_dilation_kernel = 3
    model.multi_component_min_ratio = 1e-4
    model.multi_component_min_ratio_overrides = {}

    raw_mask = np.array([
        [1, 1, 0, 2],
        [1, 1, 0, 2],
        [0, 0, 0, 2],
        [0, 0, 0, 2],
    ], dtype=np.int32)
    heat_mask = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    masks, indices = model._resolve_multi_component_masks('transistor', raw_mask, heat_mask)

    assert indices == [1]
    assert len(masks) == 1
    assert int(np.count_nonzero(masks[0])) > 1


def test_strict_query_mask_missing_raises(tmp_path):
    model = object.__new__(UniVADDetector)
    model.strict_mode = True
    model.mask_dir = str(tmp_path / 'masks')
    model.heat_mask_dir = str(tmp_path / 'heat_masks')
    model.image_size = 32
    model.query_mask_dilation_kernel = 1

    with pytest.raises(FileNotFoundError):
        model._load_query_masks('bottle', 'data/mvtec_ad/bottle/test/good/000.png', ObjectType.SINGLE)


def test_official_image_score_matches_reference_reduction():
    model = object.__new__(UniVADDetector)
    score_map = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32)

    assert model._official_image_score_from_map(score_map, '/fake/mvtec/sample.png') == pytest.approx(0.4)
    assert model._official_image_score_from_map(score_map, '/fake/HIS/sample.png') == pytest.approx(0.25)


def test_finalize_prediction_scores_uses_shared_final_map_in_official_mode():
    model = object.__new__(UniVADDetector)
    model.official_scoring = True

    capm_map = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32)
    gecm_map = torch.tensor([[[[0.0, 0.1], [0.2, 0.3]]]], dtype=torch.float32)
    final_map, final_img, image_weight, pixel_weight, global_score = model._finalize_prediction_scores(
        capm_map,
        0.9,
        gecm_map,
        '/fake/mvtec/sample.png',
        image_gecm_weight=0.0,
        pixel_gecm_weight=2.0,
    )

    assert torch.allclose(final_map, capm_map + gecm_map)
    assert final_img == pytest.approx(1.2)
    assert image_weight == pytest.approx(1.0)
    assert pixel_weight == pytest.approx(1.0)
    assert global_score == pytest.approx(0.5)


def test_finalize_prediction_scores_keeps_split_weights_in_experimental_mode():
    model = object.__new__(UniVADDetector)
    model.official_scoring = False

    capm_map = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32)
    gecm_map = torch.tensor([[[[0.0, 0.1], [0.2, 0.3]]]], dtype=torch.float32)
    final_map, final_img, image_weight, pixel_weight, global_score = model._finalize_prediction_scores(
        capm_map,
        0.9,
        gecm_map,
        '/fake/mvtec/sample.png',
        image_gecm_weight=0.0,
        pixel_gecm_weight=2.0,
    )

    assert torch.allclose(final_map, capm_map + 2.0 * gecm_map)
    assert final_img == pytest.approx(0.9)
    assert image_weight == pytest.approx(0.0)
    assert pixel_weight == pytest.approx(2.0)
    assert global_score == pytest.approx(0.5)


@pytest.mark.skipif(_SKIP, reason='open_clip or DINOv2 weights unavailable')
class TestUniVADDetector(TestCase):
    """Unit tests for UniVADDetector."""

    def setUp(self):
        # Use smaller models for testing
        self.cfg = dict(
            type='UniVADDetector',
            clip_model='ViT-B-16',
            clip_pretrained='openai',
            dinov2_model='dinov2_vitb14',
            clip_layers=(3, 6, 9, 12),
            k_shot=2,
            image_size=56,
            mask_dir='',
            gaussian_sigma=2.0,
            gecm_enable=False,
        )
        self.img_size = 56

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, self.img_size, self.img_size), mode='tensor')
        assert out is not None

    def test_forward_loss(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, self.img_size, self.img_size)
        out = model(torch.randn(2, 3, self.img_size, self.img_size),
                    data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        # Collect phase: loss mode to gather references
        for _ in range(2):
            data_samples = _make_data_samples(2, self.img_size, self.img_size)
            model(torch.randn(2, 3, self.img_size, self.img_size),
                  data_samples, mode='loss')
        # Build memory bank
        model.fit()
        # Predict
        data_samples = _make_data_samples(2, self.img_size, self.img_size)
        out = model(torch.randn(2, 3, self.img_size, self.img_size),
                    data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        for sample in out:
            assert hasattr(sample, 'pred_score')
            assert hasattr(sample, 'pred_anomaly_map')

    def test_gate_fallback_texture(self):
        """Without masks, gate should default to TEXTURE."""
        model = MODELS.build(self.cfg)
        model.eval()
        # Collect
        data_samples = _make_data_samples(2, self.img_size, self.img_size)
        model(torch.randn(2, 3, self.img_size, self.img_size),
              data_samples, mode='loss')
        model.fit()
        assert model._gate.get('bottle') == ObjectType.TEXTURE

    def test_k_shot_limit(self):
        """Should not collect more than k_shot references per class."""
        model = MODELS.build(self.cfg)
        model.train()
        # Feed more images than k_shot
        for _ in range(5):
            data_samples = _make_data_samples(2, self.img_size, self.img_size)
            model(torch.randn(2, 3, self.img_size, self.img_size),
                  data_samples, mode='loss')
        assert len(model._ref_images['bottle']) == model.k_shot

    def test_multi_class(self):
        """Test with multiple class names."""
        model = MODELS.build(self.cfg)
        model.train()
        samples = _make_data_samples(1, self.img_size, self.img_size, cls_name='bottle')
        samples += _make_data_samples(1, self.img_size, self.img_size, cls_name='cable')
        model(torch.randn(2, 3, self.img_size, self.img_size), samples, mode='loss')
        assert 'bottle' in model._ref_images
        assert 'cable' in model._ref_images
