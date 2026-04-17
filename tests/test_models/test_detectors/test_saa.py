"""Tests for SAADetector (SAA/SAA+)."""

import math

from mmengine.config import Config
import numpy as np
import pytest
import torch
import torch.nn as nn

import baoiad  # noqa: F401  - trigger registry import

from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_data_samples(batch_size, H=256, W=256):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.gt_label = i % 2
        sample.gt_mask = torch.zeros(H, W)
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{i}.png',
            'defect_type': 'good',
        })
        samples.append(sample)
    return samples


def _build_saa_detector(monkeypatch, **overrides):
    from baoiad.models.detectors import saa as saa_module

    monkeypatch.setattr(saa_module, 'HAS_GROUNDING_DINO', True)
    monkeypatch.setattr(saa_module, 'HAS_SAM', True)

    kwargs = dict(
        mode='saa',
        class_name='bottle',
        image_size=256,
        grounding_dino_cfg=dict(
            config_path='groundingdino/config/GroundingDINO_SwinT_OGC.py',
            checkpoint='dummy_gdino.pth',
        ),
        sam_cfg=dict(
            model_type='vit_h',
            checkpoint='dummy_sam.pth',
        ),
    )
    kwargs.update(overrides)
    return saa_module.SAADetector(**kwargs)


class _DummySaliencyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.requested_sizes = []

    def set_img_size(self, image_size):
        self.requested_sizes.append(image_size)

    def forward(self, image):
        features = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]]]],
            dtype=torch.float32,
        )
        return features, 1.0, 1.0


class TestSAAPrompts:
    """Prompt generation should stay aligned with the official repo layout."""

    def test_build_saa_prompts_vanilla_pairs(self):
        from baoiad.models.detectors.saa_prompts import build_saa_prompts

        prompts, property_prompt = build_saa_prompts('bottle', mode='saa')

        assert property_prompt is None
        assert prompts == [
            ('defect on bottle', 'bottle'),
            ('damage on bottle', 'bottle'),
            ('flaw on bottle', 'bottle'),
        ]

    def test_build_saa_prompts_plus_mvtec(self):
        from baoiad.models.detectors.saa_prompts import build_saa_prompts

        prompts_saa, _ = build_saa_prompts('bottle', mode='saa')
        prompts_plus, property_prompt = build_saa_prompts('bottle', mode='saa+')

        assert len(prompts_plus) > len(prompts_saa)
        assert ('broken part. contamination. white broken.', 'bottle') in prompts_plus
        assert property_prompt is not None
        assert 'maximum of 5 anomaly' in property_prompt
        assert '0.3 object area' in property_prompt

    def test_build_saa_prompts_custom(self):
        from baoiad.models.detectors.saa_prompts import build_saa_prompts

        custom = [
            ('custom defect', 'bottle'),
            ('custom flaw', 'bottle'),
        ]
        prompts, property_prompt = build_saa_prompts(
            'bottle', mode='saa', custom_prompts=custom
        )

        assert prompts == custom
        assert property_prompt is None

    def test_build_saa_prompts_unknown_category(self):
        from baoiad.models.detectors.saa_prompts import build_saa_prompts

        prompts, property_prompt = build_saa_prompts('unknown thing', mode='saa+')

        assert property_prompt is None
        assert prompts == [
            ('defect on unknown thing', 'unknown thing'),
            ('damage on unknown thing', 'unknown thing'),
            ('flaw on unknown thing', 'unknown thing'),
        ]

    def test_parse_property_prompt(self):
        from baoiad.models.detectors.saa_prompts import parse_property_prompt

        props = parse_property_prompt(
            'the image of bottle have 1 dissimilar bottle, '
            'with a maximum of 5 anomaly. '
            'The anomaly would not exceed 0.3 object area. '
        )

        assert props == {
            'object_prompt': 'bottle',
            'object_number': 1,
            'k_mask': 5,
            'defect_area_threshold': 0.3,
            'object_max_area': 1.0,
        }


class TestSAARegistry:
    """Import and registry wiring should not depend on optional weights."""

    def test_import(self):
        from baoiad.models.detectors.saa import SAADetector

        assert SAADetector is not None

    def test_registry(self):
        assert MODELS.get('SAADetector') is not None


class TestSAAConfig:
    """Config-level guards for the strict SAA evaluation path."""

    def test_saaplus_strict_config_freezes_official_mapmax_raw_bgr_sam_path(self):
        cfg = Config.fromfile('configs/saaplus/saaplus_400_mvtec_strict.py')

        assert cfg.benchmark_eval_only is True
        assert cfg.benchmark_multi_class is False
        assert cfg.model.mode == 'saa+'
        assert cfg.model.image_size == 400
        assert cfg.model.box_area_tolerance == pytest.approx(0.0)
        assert cfg.model.box_area_tolerance_overrides == dict(pill=0.003)
        assert cfg.model.image_score_aggregation == 'map_max'
        assert cfg.model.sam_preconvert_rgb is False
        assert cfg.model.saliency_backbone.type == 'SAASaliencyBackbone'
        assert 'image_score_phrase_allowlist' not in cfg.model
        assert 'image_score_phrase_blocklist' not in cfg.model
        assert 'image_score_area_range_overrides' not in cfg.model
        assert 'image_score_rank_mode_overrides' not in cfg.model

        pipeline = cfg.test_dataloader.dataset.pipeline
        assert pipeline[0].type == 'LoadImage'
        assert pipeline[0].to_rgb is False
        assert pipeline[0].keep_bgr_copy is True
        assert pipeline[-1].type == 'PackADInputs'


class TestSAADetectorBehavior:
    """Lightweight behavioral guards for zero-shot SAA logic."""

    def test_init_requires_groundingdino(self, monkeypatch):
        from baoiad.models.detectors import saa as saa_module

        monkeypatch.setattr(saa_module, 'HAS_GROUNDING_DINO', False)
        monkeypatch.setattr(saa_module, 'HAS_SAM', True)

        with pytest.raises(ImportError, match='groundingdino is required'):
            saa_module.SAADetector()

    def test_init_requires_segment_anything(self, monkeypatch):
        from baoiad.models.detectors import saa as saa_module

        monkeypatch.setattr(saa_module, 'HAS_GROUNDING_DINO', True)
        monkeypatch.setattr(saa_module, 'HAS_SAM', False)

        with pytest.raises(ImportError, match='segment_anything is required'):
            saa_module.SAADetector()

    def test_ensure_models_loaded_delegates_to_split_loaders(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch)
        calls = []

        monkeypatch.setattr(detector, '_ensure_gdino_loaded', lambda: calls.append('gdino'))
        monkeypatch.setattr(detector, '_ensure_sam_loaded', lambda: calls.append('sam'))

        detector._ensure_models_loaded()

        assert calls == ['gdino', 'sam']

    def test_resolve_runtime_vanilla_class_overrides(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            k_mask=5,
            k_mask_overrides=dict(pill=1),
            defect_area_threshold=0.5,
            defect_area_threshold_overrides=dict(pill=1.0),
        )

        assert detector._resolve_runtime_k_mask('pill') == 1
        assert detector._resolve_runtime_k_mask('bottle') == 5
        assert detector._resolve_runtime_defect_area_threshold('pill') == pytest.approx(1.0)
        assert detector._resolve_runtime_defect_area_threshold('bottle') == pytest.approx(0.5)

    def test_resolve_runtime_box_area_tolerance_uses_class_override(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            mode='saa+',
            box_area_tolerance=0.0,
            box_area_tolerance_overrides=dict(pill=0.003),
        )

        assert detector._resolve_runtime_box_area_tolerance('pill') == pytest.approx(0.003)
        assert detector._resolve_runtime_box_area_tolerance('bottle') == pytest.approx(0.0)

    def test_denormalize_shape_range_and_bgr_order(self):
        from baoiad.models.detectors.saa import SAADetector

        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        rgb_pixel = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        normalized = ((rgb_pixel - mean) / std).view(3, 1, 1)

        result = SAADetector._denormalize_to_bgr_numpy(normalized)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1, 3)
        assert result.dtype == np.uint8
        assert result[0, 0].tolist() == [30, 20, 10]

    def test_aggregate_anomaly_map_matches_average_overlay(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=2, k_mask=5)
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ])
        scores = torch.tensor([0.6, 0.8])

        img_score, anomaly_map = detector._aggregate_anomaly_map(
            masks, scores, H=2, W=2, defect_max_area=1.0
        )

        assert img_score == pytest.approx(1.4 / 3.0)
        assert anomaly_map.shape == (1, 2, 2)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(1.4 / 3.0)
        assert float(anomaly_map[0, 1, 1]) == pytest.approx(0.8 / 2.0)
        assert float(anomaly_map[0, 0, 1]) == pytest.approx(0.0)

    def test_aggregate_anomaly_map_can_rank_topk_by_separate_scores(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=2, k_mask=1)
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ])
        scores = torch.tensor([0.9, 0.4])
        rank_scores = torch.tensor([0.1, 0.8])

        img_score, anomaly_map = detector._aggregate_anomaly_map(
            masks,
            scores,
            H=2,
            W=2,
            k_mask=1,
            rank_scores=rank_scores,
        )

        assert img_score == pytest.approx(0.4 / 2.0)
        assert float(anomaly_map[0, 1, 1]) == pytest.approx(0.4 / 2.0)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(0.0)

    def test_aggregate_anomaly_map_supports_topk_combined_score_max(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            image_size=2,
            image_score_aggregation='topk_combined_score_max',
            k_mask=2,
        )
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ])
        scores = torch.tensor([0.9, 0.4])

        img_score, anomaly_map = detector._aggregate_anomaly_map(
            masks, scores, H=2, W=2, k_mask=2
        )

        assert img_score == pytest.approx(0.9)
        assert anomaly_map.shape == (1, 2, 2)

    def test_aggregate_anomaly_map_supports_topk_combined_score_mean(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            image_size=2,
            image_score_aggregation='topk_combined_score_mean',
            image_score_topk=2,
            k_mask=2,
        )
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ])
        scores = torch.tensor([0.9, 0.4])

        img_score, anomaly_map = detector._aggregate_anomaly_map(
            masks, scores, H=2, W=2, k_mask=2
        )

        assert img_score == pytest.approx((0.9 + 0.4) / 2.0)
        assert anomaly_map.shape == (1, 2, 2)

    def test_aggregate_anomaly_map_can_use_separate_image_score_scores(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            image_size=2,
            image_score_aggregation='topk_combined_score_mean',
            image_score_topk=2,
            k_mask=2,
        )
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ])
        scores = torch.tensor([0.9, 0.4])
        image_score_scores = torch.tensor([0.2, 0.1])

        img_score, anomaly_map = detector._aggregate_anomaly_map(
            masks,
            scores,
            H=2,
            W=2,
            k_mask=2,
            image_score_scores=image_score_scores,
        )

        assert img_score == pytest.approx(0.15)
        assert anomaly_map.shape == (1, 2, 2)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 1, 1]) == pytest.approx(0.4 / 2.0)

    def test_predict_single_phrase_blocklist_only_affects_image_score(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            mode='saa+',
            image_size=2,
            image_score_aggregation='topk_combined_score_mean',
            image_score_topk=2,
            image_score_phrase_blocklist=dict(zipper=['damage', 'flaw']),
            k_mask=2,
        )
        image_tensor = torch.zeros(3, 2, 2)
        object_masks = torch.ones(1, 2, 2)
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ])
        det_scores = torch.tensor([0.9, 0.7, 0.4])
        phrases = ['damage(0.37)', 'flaw(0.58)', 'crack leather(0.24)']

        monkeypatch.setattr(
            detector,
            '_get_prompts',
            lambda cls_name: ([('defect on zipper', 'zipper')], None),
        )
        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (1.0, object_masks),
        )
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                torch.zeros(3, 4),
                det_scores,
                phrases,
            ),
        )
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda image_bgr, boxes, H, W: (masks, torch.ones(3)),
        )
        monkeypatch.setattr(
            detector,
            '_compute_saliency',
            lambda image_bgr, object_masks, masks, object_number=1: torch.ones(3),
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'zipper')

        assert score == pytest.approx(0.4)
        assert anomaly_map.shape == (1, 2, 2)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 0, 1]) == pytest.approx(0.7 / 2.0)
        assert float(anomaly_map[0, 1, 0]) == pytest.approx(0.0)

    def test_predict_single_vanilla_ignores_hybrid_image_score_overrides(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            image_size=2,
            image_score_aggregation='topk_combined_score_mean',
            image_score_topk=2,
            image_score_phrase_blocklist=dict(zipper=['damage', 'flaw']),
            image_score_area_range_overrides=dict(zipper=(None, 0.1)),
            image_score_rank_mode_overrides=dict(zipper='det'),
            k_mask=2,
        )
        image_tensor = torch.zeros(3, 2, 2)
        object_masks = torch.ones(1, 2, 2)
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ])
        det_scores = torch.tensor([0.9, 0.7, 0.4])
        phrases = ['damage(0.37)', 'flaw(0.58)', 'crack leather(0.24)']

        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (1.0, object_masks),
        )
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                torch.zeros(3, 4),
                det_scores,
                phrases,
            ),
        )
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda image_bgr, boxes, H, W: (masks, torch.ones(3)),
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'zipper')

        assert score == pytest.approx(0.9 / 2.0)
        assert anomaly_map.shape == (1, 2, 2)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 0, 1]) == pytest.approx(0.7 / 2.0)
        assert float(anomaly_map[0, 1, 0]) == pytest.approx(0.0)

    def test_predict_single_vanilla_uses_class_k_mask_and_area_overrides(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            image_size=2,
            k_mask=5,
            k_mask_overrides=dict(pill=1),
            defect_area_threshold=0.5,
            defect_area_threshold_overrides=dict(pill=1.0),
        )
        image_tensor = torch.zeros(3, 2, 2)
        object_masks = torch.ones(1, 2, 2)
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ])
        det_scores = torch.tensor([0.9, 0.4])
        phrases = ['defect(0.45)', 'defect(0.44)']
        seen = {}

        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (0.25, object_masks),
        )

        def _fake_detect_with_grounding_dino(
            image_bgr,
            prompts,
            object_max_area=1.0,
            object_min_area=0.0,
            box_area_tolerance=None,
        ):
            seen['object_max_area'] = object_max_area
            return torch.zeros(2, 4), det_scores, phrases

        monkeypatch.setattr(detector, '_detect_with_grounding_dino', _fake_detect_with_grounding_dino)
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda image_bgr, boxes, H, W: (masks, torch.ones(2)),
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'pill')

        assert seen['object_max_area'] == pytest.approx(0.25)
        assert score == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 0, 1]) == pytest.approx(0.0)

    def test_predict_single_saaplus_mapmax_ignores_image_score_overrides(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            mode='saa+',
            image_size=2,
            image_score_aggregation='map_max',
            image_score_topk=2,
            image_score_phrase_blocklist=dict(zipper=['damage', 'flaw']),
            image_score_area_range_overrides=dict(zipper=(None, 0.1)),
            image_score_rank_mode_overrides=dict(zipper='det'),
            k_mask=2,
        )
        detector.saliency_backbone = object()
        image_tensor = torch.zeros(3, 2, 2)
        object_masks = torch.ones(1, 2, 2)
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ])
        det_scores = torch.tensor([0.9, 0.7, 0.4])
        phrases = ['damage(0.37)', 'flaw(0.58)', 'crack leather(0.24)']

        monkeypatch.setattr(
            detector,
            '_get_prompts',
            lambda cls_name: ([('defect on zipper', 'zipper')], None),
        )
        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (1.0, object_masks),
        )
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                torch.zeros(3, 4),
                det_scores,
                phrases,
            ),
        )
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda image_bgr, boxes, H, W: (masks, torch.ones(3)),
        )
        monkeypatch.setattr(
            detector,
            '_compute_saliency',
            lambda image_bgr, object_masks, masks, object_number=1: torch.ones(3),
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'zipper')

        assert score == pytest.approx(0.9 / 2.0)
        assert anomaly_map.shape == (1, 2, 2)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 0, 1]) == pytest.approx(0.7 / 2.0)
        assert float(anomaly_map[0, 1, 0]) == pytest.approx(0.0)

    def test_phrase_filter_matching_is_case_insensitive_substring(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch)

        matched = detector._match_phrase_tokens(
            ['Damage(0.37)', 'crack leather(0.24)', 'Flaw(0.58)'],
            ['damage', 'FLAW'],
        )

        assert matched == [0, 2]

    def test_predict_single_image_score_rank_override_only_affects_image_score(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            mode='saa+',
            image_size=2,
            image_score_aggregation='topk_combined_score_mean',
            image_score_topk=2,
            image_score_rank_mode='combined',
            image_score_rank_mode_overrides=dict(zipper='det'),
            k_mask=2,
        )
        detector.saliency_backbone = object()
        image_tensor = torch.zeros(3, 2, 2)
        object_masks = torch.ones(1, 2, 2)
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ])
        det_scores = torch.tensor([0.9, 0.8, 0.1])
        saliency_scores = torch.tensor([1.0, 0.1, 8.0])
        phrases = ['defect(0.45)', 'defect(0.44)', 'defect(0.43)']

        monkeypatch.setattr(
            detector,
            '_get_prompts',
            lambda cls_name: ([('defect on zipper', 'zipper')], None),
        )
        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (1.0, object_masks),
        )
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                torch.zeros(3, 4),
                det_scores,
                phrases,
            ),
        )
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda image_bgr, boxes, H, W: (masks, torch.ones(3)),
        )
        monkeypatch.setattr(
            detector,
            '_compute_saliency',
            lambda image_bgr, object_masks, masks, object_number=1: saliency_scores,
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'zipper')

        assert score == pytest.approx((0.9 + 0.08) / 2.0)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 1, 0]) == pytest.approx(0.8 / 2.0)

    def test_predict_single_area_range_only_affects_image_score(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            mode='saa+',
            image_size=4,
            image_score_aggregation='topk_combined_score_mean',
            image_score_topk=2,
            image_score_area_range_overrides=dict(zipper=(None, 0.1)),
            k_mask=2,
        )
        image_tensor = torch.zeros(3, 4, 4)
        object_masks = torch.ones(1, 4, 4)
        masks = torch.tensor([
            [[1.0, 1.0, 0.0, 0.0],
             [1.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0],
             [0.0, 0.0, 0.0, 0.0]],
        ])
        det_scores = torch.tensor([0.9, 0.8, 0.4])
        phrases = ['defect(0.45)', 'defect(0.44)', 'defect(0.43)']

        monkeypatch.setattr(
            detector,
            '_get_prompts',
            lambda cls_name: ([('defect on zipper', 'zipper')], None),
        )
        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (1.0, object_masks),
        )
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                torch.zeros(3, 4),
                det_scores,
                phrases,
            ),
        )
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda image_bgr, boxes, H, W: (masks, torch.ones(3)),
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'zipper')

        assert score == pytest.approx((0.8 + 0.4) / 2.0)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 0, 2]) == pytest.approx(0.8 / 2.0)

    def test_predict_single_saliency_identity_mode_uses_det_scores(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            mode='saa+',
            image_size=2,
            saliency_score_mode_overrides=dict(transistor='identity'),
            k_mask=2,
        )
        detector.saliency_backbone = object()
        image_tensor = torch.zeros(3, 2, 2)
        object_masks = torch.ones(1, 2, 2)
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ])
        det_scores = torch.tensor([0.9, 0.4])
        saliency_scores = torch.tensor([4.0, 5.0])
        phrases = ['defect(0.45)', 'defect(0.44)']

        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (1.0, object_masks),
        )
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                torch.zeros(2, 4),
                det_scores,
                phrases,
            ),
        )
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda image_bgr, boxes, H, W: (masks, torch.ones(2)),
        )
        monkeypatch.setattr(
            detector,
            '_compute_saliency',
            lambda image_bgr, object_masks, masks, object_number=1: saliency_scores,
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'transistor')

        assert score == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx(0.9 / 2.0)
        assert float(anomaly_map[0, 0, 1]) == pytest.approx(0.4 / 2.0)

    def test_predict_single_saliency_clipped_mode_caps_scores(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            mode='saa+',
            image_size=2,
            saliency_score_mode_overrides=dict(transistor='clipped_multiply'),
            saliency_score_clip_max_overrides=dict(transistor=1.25),
            k_mask=2,
        )
        detector.saliency_backbone = object()
        image_tensor = torch.zeros(3, 2, 2)
        object_masks = torch.ones(1, 2, 2)
        masks = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ])
        det_scores = torch.tensor([0.9, 0.4])
        saliency_scores = torch.tensor([4.0, 1.1])
        phrases = ['defect(0.45)', 'defect(0.44)']

        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (1.0, object_masks),
        )
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                torch.zeros(2, 4),
                det_scores,
                phrases,
            ),
        )
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda image_bgr, boxes, H, W: (masks, torch.ones(2)),
        )
        monkeypatch.setattr(
            detector,
            '_compute_saliency',
            lambda image_bgr, object_masks, masks, object_number=1: saliency_scores,
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'transistor')

        assert score == pytest.approx((0.9 * 1.25) / 2.0)
        assert float(anomaly_map[0, 0, 0]) == pytest.approx((0.9 * 1.25) / 2.0)
        assert float(anomaly_map[0, 0, 1]) == pytest.approx((0.4 * 1.1) / 2.0)

    def test_predict_single_returns_zero_map_when_no_boxes(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=4)
        image_tensor = torch.zeros(3, 4, 4)

        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (1.0, torch.zeros(1, 4, 4)),
        )
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                torch.zeros(0, 4), torch.zeros(0), []
            ),
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'bottle')

        assert score == 0.0
        assert anomaly_map.shape == (1, 4, 4)
        assert torch.count_nonzero(anomaly_map) == 0
        assert torch.isfinite(anomaly_map).all()

    def test_predict_single_filters_boxes_by_object_area(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            image_size=4,
            defect_area_threshold=0.5,
        )
        image_tensor = torch.zeros(3, 4, 4)

        monkeypatch.setattr(
            detector,
            '_detect_object',
            lambda image_bgr, object_prompt, object_max_area=1.0: (0.2, torch.zeros(1, 4, 4)),
        )
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                (torch.zeros(0, 4), torch.zeros(0), [])
                if object_max_area <= 0.1 else
                (
                    torch.tensor([[0.5, 0.5, 0.8, 0.2]], dtype=torch.float32),
                    torch.tensor([0.9], dtype=torch.float32),
                    ['defect on bottle'],
                )
            ),
        )

        score, anomaly_map = detector._predict_single(image_tensor, 'bottle')

        assert score == 0.0
        assert anomaly_map.shape == (1, 4, 4)
        assert torch.count_nonzero(anomaly_map) == 0

    def test_bbox_suppression_box_area_tolerance_keeps_borderline_box(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=4, box_area_tolerance=0.001)

        class _FakeGDINO:
            tokenizer = staticmethod(lambda text: text)

        detector.gdino_model = _FakeGDINO()
        monkeypatch.setattr(
            'baoiad.models.detectors.saa.get_phrases_from_posmap',
            lambda mask, tokenized, tokenizer: 'defect',
        )

        boxes = torch.tensor([[0.5, 0.5, 0.726, 0.727]], dtype=torch.float32)
        logits = torch.tensor([[0.2, 0.3]], dtype=torch.float32)

        boxes_filtered, logits_filtered, phrases = detector._bbox_suppression(
            boxes,
            logits,
            object_phrase='defect on pill.',
            filtered_phrase='pill',
            bbox_score_thr=0.1,
            text_score_thr=0.1,
            object_max_area=0.527,
            object_min_area=0.0,
        )

        assert boxes_filtered is not None
        assert logits_filtered is not None
        assert len(phrases) == 1
        assert phrases[0].startswith('defect(0.3')

    def test_bbox_suppression_without_tolerance_filters_borderline_box(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=4, box_area_tolerance=0.0)

        class _FakeGDINO:
            tokenizer = staticmethod(lambda text: text)

        detector.gdino_model = _FakeGDINO()
        monkeypatch.setattr(
            'baoiad.models.detectors.saa.get_phrases_from_posmap',
            lambda mask, tokenized, tokenizer: 'defect',
        )

        boxes = torch.tensor([[0.5, 0.5, 0.726, 0.727]], dtype=torch.float32)
        logits = torch.tensor([[0.2, 0.3]], dtype=torch.float32)

        boxes_filtered, logits_filtered, phrases = detector._bbox_suppression(
            boxes,
            logits,
            object_phrase='defect on pill.',
            filtered_phrase='pill',
            bbox_score_thr=0.1,
            text_score_thr=0.1,
            object_max_area=0.527,
            object_min_area=0.0,
        )

        assert boxes_filtered is None
        assert logits_filtered is None
        assert phrases is None

    def test_detect_with_grounding_dino_uses_bbox_suppression_path(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=4)
        image_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        prompts = [
            ('defect on bottle', 'bottle'),
            ('damage on bottle', 'bottle'),
        ]
        calls = {
            'captions': [],
            'bbox_args': [],
        }
        dummy_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
        dummy_logits = torch.tensor([[0.9, 0.1]], dtype=torch.float32)

        def _fake_prepare(image):
            calls['prepared_shape'] = image.shape
            return torch.ones(3, 2, 2)

        def _fake_grounding_output(dino_image, caption):
            assert dino_image.shape == (3, 2, 2)
            calls['captions'].append(caption)
            return dummy_boxes, dummy_logits, caption

        def _fake_bbox_suppression(
            boxes,
            logits,
            object_phrase,
            filtered_phrase,
            bbox_score_thr,
            text_score_thr,
            object_max_area,
            object_min_area,
            box_area_tolerance=None,
        ):
            calls['bbox_args'].append({
                'object_phrase': object_phrase,
                'filtered_phrase': filtered_phrase,
                'bbox_score_thr': bbox_score_thr,
                'text_score_thr': text_score_thr,
                'object_max_area': object_max_area,
                'object_min_area': object_min_area,
            })
            score = 0.8 if 'defect' in object_phrase else 0.6
            phrase = object_phrase.replace('.', '')
            return dummy_boxes, torch.tensor([score], dtype=torch.float32), [phrase]

        monkeypatch.setattr(detector, '_prepare_gdino_image', _fake_prepare)
        monkeypatch.setattr(detector, '_get_grounding_output', _fake_grounding_output)
        monkeypatch.setattr(detector, '_bbox_suppression', _fake_bbox_suppression)

        boxes, scores, phrases = detector._detect_with_grounding_dino(
            image_bgr,
            prompts,
            object_max_area=0.3,
            object_min_area=0.05,
        )

        assert calls['prepared_shape'] == (4, 4, 3)
        assert calls['captions'] == ['defect on bottle', 'damage on bottle']
        assert len(calls['bbox_args']) == 2
        assert all(arg['bbox_score_thr'] == pytest.approx(detector.box_threshold)
                   for arg in calls['bbox_args'])
        assert all(arg['text_score_thr'] == pytest.approx(detector.text_threshold)
                   for arg in calls['bbox_args'])
        assert all(arg['object_max_area'] == pytest.approx(0.3)
                   for arg in calls['bbox_args'])
        assert all(arg['object_min_area'] == pytest.approx(0.05)
                   for arg in calls['bbox_args'])
        assert boxes.shape == (2, 4)
        assert scores.tolist() == pytest.approx([0.8, 0.6])
        assert phrases == ['defect on bottle', 'damage on bottle']

    def test_detect_object_can_skip_sam_in_proposal_only_mode(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=4)
        image_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        dummy_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]], dtype=torch.float32)
        dummy_logits = torch.tensor([[0.9, 0.1]], dtype=torch.float32)

        monkeypatch.setattr(detector, '_prepare_gdino_image', lambda image: torch.ones(3, 2, 2))
        monkeypatch.setattr(
            detector,
            '_get_grounding_output',
            lambda dino_image, caption: (dummy_boxes, dummy_logits, caption),
        )
        monkeypatch.setattr(
            detector,
            '_bbox_suppression',
            lambda *args, **kwargs: (dummy_boxes, torch.tensor([0.8]), ['pill']),
        )
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('SAM should be skipped')),
        )

        object_area, object_masks = detector._detect_object(
            image_bgr,
            'pill',
            use_sam=False,
        )

        assert object_area == pytest.approx(0.06)
        assert object_masks.shape == (1, 4, 4)
        assert torch.count_nonzero(object_masks) == 0

    def test_segment_with_sam_can_preserve_raw_bgr_for_official_mode(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=4, sam_preconvert_rgb=False)
        image_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        image_bgr[..., 0] = 10
        image_bgr[..., 1] = 20
        image_bgr[..., 2] = 30
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32)
        captured = {}

        class _FakeTransform:
            @staticmethod
            def apply_boxes_torch(boxes_pixel, image_shape):
                captured['boxes_pixel'] = boxes_pixel.clone()
                captured['image_shape'] = image_shape
                return boxes_pixel

        class _FakeModel:
            device = torch.device('cpu')

        class _FakePredictor:
            def __init__(self):
                self.model = _FakeModel()
                self.transform = _FakeTransform()

            def set_image(self, image):
                captured['image'] = image.copy()

            def predict_torch(self, **kwargs):
                masks = torch.ones(1, 1, 4, 4)
                iou = torch.ones(1, 1)
                logits = torch.zeros(1, 1, 4, 4)
                return masks, iou, logits

        detector.sam_predictor = _FakePredictor()
        masks, iou = detector._segment_with_sam(image_bgr, boxes, 4, 4)

        assert np.array_equal(captured['image'], image_bgr)
        assert captured['image_shape'] == (4, 4)
        assert masks.shape == (1, 4, 4)
        assert iou.shape == (1,)

    def test_forward_predict_uses_data_sample_class_name(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=4)
        samples = _make_data_samples(2, H=4, W=4)
        samples[1].set_metainfo({
            'cls_name': 'cable',
            'img_path': '/fake/1.png',
            'defect_type': 'good',
        })
        seen_cls_names = []
        seen_raw_flags = []

        monkeypatch.setattr(detector, '_ensure_models_loaded', lambda: None)

        def _fake_predict_single(image_tensor, cls_name, image_bgr=None):
            seen_cls_names.append(cls_name)
            seen_raw_flags.append(image_bgr is not None)
            value = float(image_tensor.mean().item())
            return value, torch.full((1, 4, 4), value)

        monkeypatch.setattr(detector, '_predict_single', _fake_predict_single)

        inputs = torch.stack([
            torch.zeros(3, 4, 4),
            torch.ones(3, 4, 4),
        ])

        results = detector(inputs, samples, mode='predict')

        assert seen_cls_names == ['bottle', 'cable']
        assert seen_raw_flags == [False, False]
        assert len(results) == 2
        assert [result.pred_score for result in results] == pytest.approx([0.0, 1.0])
        assert results[0].pred_anomaly_map.shape == (1, 4, 4)
        assert results[1].pred_anomaly_map.shape == (1, 4, 4)
        assert all(math.isfinite(result.pred_score) for result in results)

    def test_forward_predict_prefers_ori_img_bgr_when_present(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=4)
        sample = _make_data_samples(1, H=4, W=4)[0]
        raw_bgr = np.full((7, 5, 3), 123, dtype=np.uint8)
        sample.set_metainfo({
            'ori_img_bgr': raw_bgr,
            'cls_name': 'bottle',
            'img_path': '/fake/0.png',
            'defect_type': 'good',
        })
        captured = {}

        monkeypatch.setattr(detector, '_ensure_models_loaded', lambda: None)

        def _fake_predict_single(image_tensor, cls_name, image_bgr=None):
            captured['image_bgr'] = image_bgr
            return 0.5, torch.ones(1, 4, 4)

        monkeypatch.setattr(detector, '_predict_single', _fake_predict_single)

        detector(torch.zeros(1, 3, 4, 4), [sample], mode='predict')

        assert np.array_equal(captured['image_bgr'], raw_bgr)

    def test_score_all_applies_dataset_level_minmax_normalization(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, image_size=2)
        samples = _make_data_samples(2, H=2, W=2)

        monkeypatch.setattr(detector, '_ensure_models_loaded', lambda: None)

        outputs = [
            (0.8, torch.tensor([[[2.0, 4.0], [0.0, 1.0]]])),
            (0.6, torch.tensor([[[3.0, 5.0], [1.0, 1.0]]])),
        ]

        def _fake_predict_single(image_tensor, cls_name, image_bgr=None):
            return outputs.pop(0)

        monkeypatch.setattr(detector, '_predict_single', _fake_predict_single)

        inputs = torch.stack([
            torch.zeros(3, 2, 2),
            torch.ones(3, 2, 2),
        ])
        results = detector(inputs, samples, mode='predict')
        finalized = detector.score_all()

        assert finalized == results
        assert detector._pending_samples == []
        assert detector._pending_score_maps == []

        expected_first = torch.tensor([[[0.4, 0.8], [0.0, 0.2]]])
        expected_second = torch.tensor([[[0.6, 1.0], [0.2, 0.2]]])

        assert torch.allclose(results[0].pred_anomaly_map, expected_first)
        assert torch.allclose(results[1].pred_anomaly_map, expected_second)
        assert results[0].pred_score == pytest.approx(0.8)
        assert results[1].pred_score == pytest.approx(1.0)

    def test_score_all_topk_combined_score_max_uses_raw_image_scores(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            mode='saa+',
            image_size=2,
            image_score_aggregation='topk_combined_score_max',
        )
        samples = _make_data_samples(2, H=2, W=2)

        monkeypatch.setattr(detector, '_ensure_models_loaded', lambda: None)

        outputs = [
            (0.9, torch.tensor([[[2.0, 4.0], [0.0, 1.0]]])),
            (0.6, torch.tensor([[[3.0, 5.0], [1.0, 1.0]]])),
        ]

        def _fake_predict_single(image_tensor, cls_name, image_bgr=None):
            return outputs.pop(0)

        monkeypatch.setattr(detector, '_predict_single', _fake_predict_single)

        inputs = torch.stack([
            torch.zeros(3, 2, 2),
            torch.ones(3, 2, 2),
        ])
        results = detector(inputs, samples, mode='predict')
        finalized = detector.score_all()

        assert finalized == results
        assert detector._pending_raw_image_scores == []
        assert results[0].pred_score == pytest.approx(1.0)
        assert results[1].pred_score == pytest.approx(0.0)

    def test_score_all_topk_combined_score_mean_uses_raw_image_scores(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            mode='saa+',
            image_size=2,
            image_score_aggregation='topk_combined_score_mean',
            image_score_topk=2,
        )
        samples = _make_data_samples(2, H=2, W=2)

        monkeypatch.setattr(detector, '_ensure_models_loaded', lambda: None)

        outputs = [
            (0.9, torch.tensor([[[2.0, 4.0], [0.0, 1.0]]])),
            (0.6, torch.tensor([[[3.0, 5.0], [1.0, 1.0]]])),
        ]

        def _fake_predict_single(image_tensor, cls_name, image_bgr=None):
            return outputs.pop(0)

        monkeypatch.setattr(detector, '_predict_single', _fake_predict_single)

        inputs = torch.stack([
            torch.zeros(3, 2, 2),
            torch.ones(3, 2, 2),
        ])
        results = detector(inputs, samples, mode='predict')
        finalized = detector.score_all()

        assert finalized == results
        assert detector._pending_raw_image_scores == []
        assert results[0].pred_score == pytest.approx(1.0)
        assert results[1].pred_score == pytest.approx(0.0)

    def test_score_all_vanilla_ignores_topk_raw_image_score_mode(self, monkeypatch):
        detector = _build_saa_detector(
            monkeypatch,
            image_size=2,
            image_score_aggregation='topk_combined_score_mean',
            image_score_topk=2,
        )
        samples = _make_data_samples(2, H=2, W=2)

        monkeypatch.setattr(detector, '_ensure_models_loaded', lambda: None)

        outputs = [
            (0.9, torch.tensor([[[2.0, 4.0], [0.0, 1.0]]])),
            (0.6, torch.tensor([[[3.0, 5.0], [1.0, 1.0]]])),
        ]

        def _fake_predict_single(image_tensor, cls_name, image_bgr=None):
            return outputs.pop(0)

        monkeypatch.setattr(detector, '_predict_single', _fake_predict_single)

        inputs = torch.stack([
            torch.zeros(3, 2, 2),
            torch.ones(3, 2, 2),
        ])
        results = detector(inputs, samples, mode='predict')
        finalized = detector.score_all()

        assert finalized == results
        assert detector._pending_raw_image_scores == []
        assert results[0].pred_score == pytest.approx(0.8)
        assert results[1].pred_score == pytest.approx(1.0)

    def test_compute_saliency_rescore_formula(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, mode='saa+')
        detector.saliency_backbone = _DummySaliencyBackbone()

        monkeypatch.setattr(
            detector,
            '_compute_saliency_map',
            lambda image_bgr, object_masks, object_number: np.array(
                [[0.2, 0.4], [0.0, 0.0]],
                dtype=np.float32,
            ),
        )

        defect_masks = torch.tensor([
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ])
        scores = detector._compute_saliency(
            np.zeros((2, 2, 3), dtype=np.uint8),
            torch.zeros(1, 2, 2),
            defect_masks,
            object_number=1,
        )

        assert float(scores[0]) == pytest.approx(math.exp(3 * 0.3))
        assert float(scores[1]) == pytest.approx(1.0)

    def test_compute_saliency_map_dispatches_multi_instance(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, mode='saa+')
        detector.saliency_backbone = _DummySaliencyBackbone()
        calls = []

        monkeypatch.setattr(
            detector,
            '_compute_single_object_saliency_map',
            lambda image_bgr: calls.append('single') or np.zeros((4, 4), dtype=np.float32),
        )
        monkeypatch.setattr(
            detector,
            '_compute_multi_object_saliency_map',
            lambda image_bgr, object_masks: calls.append('multi') or np.ones((4, 4), dtype=np.float32),
        )

        result = detector._compute_saliency_map(
            np.zeros((4, 4, 3), dtype=np.uint8),
            torch.ones(2, 4, 4),
            object_number=2,
        )

        assert calls == ['multi']
        assert np.all(result == 1.0)

    def test_compute_saliency_map_falls_back_when_multi_masks_empty(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, mode='saa+')
        detector.saliency_backbone = _DummySaliencyBackbone()
        calls = []

        monkeypatch.setattr(
            detector,
            '_compute_single_object_saliency_map',
            lambda image_bgr: calls.append('single') or np.zeros((4, 4), dtype=np.float32),
        )
        monkeypatch.setattr(
            detector,
            '_compute_multi_object_saliency_map',
            lambda image_bgr, object_masks: calls.append('multi') or np.ones((4, 4), dtype=np.float32),
        )

        result = detector._compute_saliency_map(
            np.zeros((4, 4, 3), dtype=np.uint8),
            torch.zeros(2, 4, 4),
            object_number=2,
        )

        assert calls == ['single']
        assert np.all(result == 0.0)

    def test_predict_single_uses_property_prompt_object_controls(self, monkeypatch):
        detector = _build_saa_detector(monkeypatch, mode='saa+', image_size=4)
        detector.saliency_backbone = _DummySaliencyBackbone()
        image_tensor = torch.zeros(3, 4, 4)
        captured = {}

        monkeypatch.setattr(
            detector,
            '_get_prompts',
            lambda cls_name: (
                [('defect on bottle', 'bottle')],
                'the image of bottle have 2 dissimilar bottle, '
                'with a maximum of 5 anomaly. '
                'The anomaly would not exceed 0.3 object area. ',
            ),
        )

        def _fake_detect_object(image_bgr, object_prompt, object_max_area=1.0):
            captured['object_prompt'] = object_prompt
            captured['object_max_area'] = object_max_area
            return 0.4, torch.ones(2, 4, 4)

        monkeypatch.setattr(detector, '_detect_object', _fake_detect_object)
        monkeypatch.setattr(
            detector,
            '_detect_with_grounding_dino',
            lambda image_bgr, prompts, object_max_area=1.0, object_min_area=0.0, box_area_tolerance=None: (
                torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
                torch.tensor([0.9], dtype=torch.float32),
                ['defect on bottle'],
            ),
        )
        monkeypatch.setattr(
            detector,
            '_segment_with_sam',
            lambda image_bgr, boxes, H, W: (
                torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0]]]),
                torch.ones(1),
            ),
        )

        def _fake_compute_saliency(image_bgr, object_masks, defect_masks, object_number=1):
            captured['object_number'] = object_number
            return torch.ones(defect_masks.shape[0])

        monkeypatch.setattr(detector, '_compute_saliency', _fake_compute_saliency)

        score, anomaly_map = detector._predict_single(image_tensor, 'bottle')

        assert captured['object_prompt'] == 'bottle'
        assert captured['object_max_area'] == pytest.approx(0.5)
        assert captured['object_number'] == 2
        assert score > 0.0
        assert anomaly_map.shape == (1, 4, 4)
