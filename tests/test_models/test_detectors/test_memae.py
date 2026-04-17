"""Tests for MemAEDetector."""

from pathlib import Path
from unittest import TestCase

import pytest
import torch
from mmengine import Config

import baoiad  # noqa: F401
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample

ROOT = Path(__file__).resolve().parents[3]


def _make_data_samples(batch_size, H=256, W=256):
    samples = []
    for i in range(batch_size):
        s = ADDataSample()
        s.gt_label = i % 2
        s.gt_mask = torch.zeros(H, W)
        s.cls_name = 'bottle'
        s.img_path = f'/fake/{i}.png'
        s.defect_type = 'good'
        samples.append(s)
    return samples


class TestMemAEDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='MemAEDetector',
            in_channels=3,
            frame_num=4,
            clip_mode='repeat_image',
            mem_dim=20,
            shrink_thres=0.0025,
            temporal_reduce_mode='mean',
            image_score_mode='spatiotemporal_mean',
        )

    def test_forward_tensor_returns_official_style_clip(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert tuple(out.shape) == (2, 3, 4, 64, 64)
        assert torch.isfinite(out).all()

    def test_forward_tensor_accepts_prebuilt_clip(self):
        model = MODELS.build(self.cfg)
        model.eval()
        clip = torch.randn(2, 3, 4, 64, 64)
        out = model(clip, mode='tensor')
        assert tuple(out.shape) == tuple(clip.shape)
        assert torch.isfinite(out).all()

    def test_forward_loss_returns_recon_and_entropy_terms(self):
        model = MODELS.build(self.cfg)
        model.train()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='loss')
        assert sorted(out.keys()) == ['entropy_loss', 'loss', 'recon_loss']
        assert all(torch.isfinite(value).all() for value in out.values())

    def test_forward_predict_outputs_finite_scores_and_maps(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_data_samples(2, 64, 64)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        maps = torch.stack([sample.pred_anomaly_map for sample in out])
        scores = torch.tensor([float(sample.pred_score) for sample in out])
        assert tuple(maps.shape) == (2, 1, 64, 64)
        assert torch.isfinite(maps).all()
        assert torch.isfinite(scores).all()


def test_memae_predict_uses_official_mean_spatiotemporal_score():
    cfg = dict(
        type='MemAEDetector',
        in_channels=3,
        frame_num=2,
        mem_dim=8,
        shrink_thres=0.0,
        temporal_reduce_mode='mean',
        image_score_mode='spatiotemporal_mean',
    )
    model = MODELS.build(cfg)
    model.eval()

    def _fake_details(clip_inputs):
        reconstruction = clip_inputs + 1.0
        attention = torch.ones(clip_inputs.shape[0], cfg['mem_dim'], 1, 1, 1, device=clip_inputs.device)
        residual = reconstruction - clip_inputs
        spatiotemporal_map = torch.sum(residual.pow(2), dim=1).sqrt()
        anomaly_map = spatiotemporal_map.mean(dim=1, keepdim=True)
        img_scores = spatiotemporal_map.flatten(1).mean(dim=1)
        return {
            'clip_inputs': clip_inputs,
            'encoded': torch.zeros(clip_inputs.shape[0], 1, clip_inputs.shape[2], 1, 1, device=clip_inputs.device),
            'memory_features': torch.zeros(clip_inputs.shape[0], 1, clip_inputs.shape[2], 1, 1, device=clip_inputs.device),
            'attention': attention,
            'reconstruction': reconstruction,
            'residual': residual,
            'spatiotemporal_map': spatiotemporal_map,
            'anomaly_map': anomaly_map,
            'img_scores': img_scores,
        }

    model._forward_clip_details = _fake_details  # type: ignore[method-assign]
    outputs = model(
        torch.zeros(1, 3, 4, 4),
        data_samples=_make_data_samples(1, 4, 4),
        mode='predict',
    )

    expected = 3.0 ** 0.5
    assert float(outputs[0].pred_score) == pytest.approx(expected)
    assert tuple(outputs[0].pred_anomaly_map.shape) == (1, 4, 4)
    assert torch.allclose(outputs[0].pred_anomaly_map, torch.full((1, 4, 4), expected))


def test_memae_map_max_score_mode_uses_reduced_map_max():
    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=2,
        mem_dim=8,
        shrink_thres=0.0,
        temporal_reduce_mode='max',
        image_score_mode='map_max',
    )
    model = MODELS.build(cfg)
    model.eval()

    reconstruction = torch.tensor(
        [[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 3.0], [0.0, 0.0]]]]],
        dtype=torch.float32,
    )
    attention = torch.ones(1, cfg['mem_dim'], 1, 1, 1)

    def _fake_details(clip_inputs):
        residual = reconstruction - clip_inputs
        spatiotemporal_map = torch.sum(residual.pow(2), dim=1).sqrt()
        anomaly_map = spatiotemporal_map.max(dim=1, keepdim=True).values
        img_scores = anomaly_map.flatten(1).max(dim=1).values
        return {
            'clip_inputs': clip_inputs,
            'encoded': torch.zeros(clip_inputs.shape[0], 1, clip_inputs.shape[2], 1, 1, device=clip_inputs.device),
            'memory_features': torch.zeros(clip_inputs.shape[0], 1, clip_inputs.shape[2], 1, 1, device=clip_inputs.device),
            'attention': attention.to(clip_inputs.device),
            'reconstruction': reconstruction.to(clip_inputs.device),
            'residual': residual,
            'spatiotemporal_map': spatiotemporal_map,
            'anomaly_map': anomaly_map,
            'img_scores': img_scores,
        }

    model._forward_clip_details = _fake_details  # type: ignore[method-assign]

    outputs = model(
        torch.zeros(1, 1, 2, 2),
        data_samples=_make_data_samples(1, 2, 2),
        mode='predict',
    )

    assert float(outputs[0].pred_score) == pytest.approx(3.0)
    expected_map = torch.tensor([[[0.0, 3.0], [0.0, 0.0]]], dtype=torch.float32)
    assert torch.allclose(outputs[0].pred_anomaly_map, expected_map)


def test_memae_map_topk_mean_score_mode_uses_topk_ratio():
    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=1,
        mem_dim=8,
        shrink_thres=0.0,
        temporal_reduce_mode='mean',
        image_score_mode='map_topk_mean',
        topk_ratio=0.25,
    )
    model = MODELS.build(cfg)
    model.eval()

    reconstruction = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]], dtype=torch.float32)
    attention = torch.ones(1, cfg['mem_dim'], 1, 1, 1)

    def _fake_details(clip_inputs):
        residual = reconstruction - clip_inputs
        spatiotemporal_map = torch.sum(residual.pow(2), dim=1).sqrt()
        anomaly_map = spatiotemporal_map.mean(dim=1, keepdim=True)
        flat = anomaly_map.flatten(1)
        img_scores = torch.topk(flat, k=1, dim=1).values.mean(dim=1)
        return {
            'clip_inputs': clip_inputs,
            'encoded': torch.zeros(clip_inputs.shape[0], 1, clip_inputs.shape[2], 1, 1, device=clip_inputs.device),
            'memory_features': torch.zeros(clip_inputs.shape[0], 1, clip_inputs.shape[2], 1, 1, device=clip_inputs.device),
            'attention': attention.to(clip_inputs.device),
            'reconstruction': reconstruction.to(clip_inputs.device),
            'residual': residual,
            'spatiotemporal_map': spatiotemporal_map,
            'anomaly_map': anomaly_map,
            'img_scores': img_scores,
        }

    model._forward_clip_details = _fake_details  # type: ignore[method-assign]

    outputs = model(
        torch.zeros(1, 1, 2, 2),
        data_samples=_make_data_samples(1, 2, 2),
        mode='predict',
    )

    assert float(outputs[0].pred_score) == pytest.approx(4.0)
    assert torch.allclose(
        outputs[0].pred_anomaly_map,
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),
    )


def test_memae_invalid_clip_mode_raises():
    with pytest.raises(ValueError, match='Unsupported clip_mode'):
        MODELS.build(
            dict(
                type='MemAEDetector',
                in_channels=3,
                frame_num=4,
                clip_mode='unsupported',
                mem_dim=8,
            )
        )


def test_memae_repeat_image_with_small_jitter_has_nonzero_temporal_variation():
    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=4,
        clip_mode='repeat_image_with_small_jitter',
        mem_dim=8,
        clip_jitter_strength=0.1,
    )
    model = MODELS.build(cfg)

    clip = model._to_clip(torch.zeros(1, 1, 2, 2))

    assert tuple(clip.shape) == (1, 1, 4, 2, 2)
    assert clip.std(dim=2).mean().item() > 0


def test_memae_two_frame_pingpong_alternates_shifted_frames():
    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=4,
        clip_mode='two_frame_pingpong',
        mem_dim=8,
        clip_pingpong_shift=1,
    )
    model = MODELS.build(cfg)

    image = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    clip = model._to_clip(image)

    assert torch.allclose(clip[:, :, 0], image)
    assert torch.allclose(clip[:, :, 1], torch.roll(image, shifts=1, dims=-1))
    assert torch.allclose(clip[:, :, 2], image)
    assert torch.allclose(clip[:, :, 3], torch.roll(image, shifts=1, dims=-1))


def test_memae_multi_shift_pingpong_cycles_spatial_offsets():
    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=8,
        clip_mode='multi_shift_pingpong',
        mem_dim=8,
        clip_pingpong_shift=1,
    )
    model = MODELS.build(cfg)

    image = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    clip = model._to_clip(image)

    assert torch.allclose(clip[:, :, 0], image)
    assert torch.allclose(clip[:, :, 1], torch.roll(image, shifts=1, dims=-1))
    assert torch.allclose(clip[:, :, 3], torch.roll(image, shifts=-1, dims=-1))
    assert torch.allclose(clip[:, :, 5], torch.roll(image, shifts=1, dims=-2))
    assert torch.allclose(clip[:, :, 7], torch.roll(image, shifts=-1, dims=-2))


def test_memae_multi_view_intensity_schedule_changes_frame_statistics():
    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=6,
        clip_mode='multi_view_intensity_schedule',
        mem_dim=8,
        clip_schedule_strength=0.1,
        clip_schedule_blur_kernel_size=3,
    )
    model = MODELS.build(cfg)

    image = torch.tensor([[[[0.0, 0.5], [-0.5, 0.25]]]])
    clip = model._to_clip(image)

    assert tuple(clip.shape) == (1, 1, 6, 2, 2)
    assert clip.std(dim=2).mean().item() > 0
    assert torch.allclose(clip[:, :, 0], image)
    assert torch.allclose(clip[:, :, 1], torch.clamp(image + 0.1, min=-1.0, max=1.0))
    assert torch.allclose(clip[:, :, 2], torch.clamp(image - 0.1, min=-1.0, max=1.0))


def test_memae_local_motion_window_changes_frames_monotonically():
    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=4,
        clip_mode='local_motion_window',
        mem_dim=8,
        clip_window_scale_min=0.8,
        clip_window_translation_max=0.2,
    )
    model = MODELS.build(cfg)

    image = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    clip = model._to_clip(image)

    assert tuple(clip.shape) == (1, 1, 4, 4, 4)
    assert clip.std(dim=2).mean().item() > 0
    assert not torch.allclose(clip[:, :, 0], clip[:, :, -1])


def test_memae_centered_local_motion_window_keeps_center_frame_exact():
    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=5,
        clip_mode='centered_local_motion_window',
        mem_dim=8,
        clip_window_scale_min=0.8,
        clip_window_translation_max=0.2,
    )
    model = MODELS.build(cfg)

    image = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    clip = model._to_clip(image)

    assert tuple(clip.shape) == (1, 1, 5, 4, 4)
    assert torch.allclose(clip[:, :, 2], image)
    assert not torch.allclose(clip[:, :, 0], image)
    assert not torch.allclose(clip[:, :, 4], image)


def test_memae_progressive_crop_window_zooms_without_translation():
    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=4,
        clip_mode='progressive_crop_window',
        mem_dim=8,
        clip_window_scale_min=0.7,
    )
    model = MODELS.build(cfg)

    image = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    clip = model._to_clip(image)

    assert tuple(clip.shape) == (1, 1, 4, 4, 4)
    assert clip.std(dim=2).mean().item() > 0
    assert torch.allclose(clip[:, :, 0], image)
    assert not torch.allclose(clip[:, :, 1], image)


def test_memae_adjacent_filename_window_uses_neighbor_images(tmp_path):
    import cv2

    image0 = torch.full((1, 2, 2), -1.0)
    image1 = torch.zeros((1, 2, 2))
    image2 = torch.full((1, 2, 2), 1.0)

    for index, value in enumerate([0, 127, 255]):
        array = torch.full((2, 2, 3), value, dtype=torch.uint8).numpy()
        cv2.imwrite(str(tmp_path / f'{index:03d}.png'), array)

    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=3,
        clip_mode='adjacent_filename_window',
        mem_dim=8,
        clip_neighbor_window_radius=1,
    )
    model = MODELS.build(cfg)
    sample = ADDataSample()
    sample.img_path = str(tmp_path / '001.png')

    clip = model._to_clip(image1.unsqueeze(0), [sample])

    assert torch.allclose(clip[0, :, 0], image0)
    assert torch.allclose(clip[0, :, 1], image1)
    assert torch.allclose(clip[0, :, 2], image2)
    assert len(model._adjacent_frame_cache) == 2

    clip_again = model._to_clip(image1.unsqueeze(0), [sample])
    assert torch.allclose(clip_again, clip)
    assert len(model._adjacent_frame_cache) == 2


def test_memae_train_good_reference_window_uses_train_context(tmp_path):
    import cv2

    train_dir = tmp_path / 'bottle' / 'train' / 'good'
    test_dir = tmp_path / 'bottle' / 'test' / 'broken_large'
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    for index, value in enumerate([0, 64, 127, 191, 255]):
        array = torch.full((2, 2, 3), value, dtype=torch.uint8).numpy()
        cv2.imwrite(str(train_dir / f'{index:03d}.png'), array)
    for index, value in enumerate([32, 96, 160]):
        array = torch.full((2, 2, 3), value, dtype=torch.uint8).numpy()
        cv2.imwrite(str(test_dir / f'{index:03d}.png'), array)

    current = torch.full((1, 2, 2), (160.0 - 127.5) / 127.5)

    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=3,
        clip_mode='train_good_reference_window',
        mem_dim=8,
        clip_neighbor_window_radius=1,
    )
    model = MODELS.build(cfg)
    sample = ADDataSample()
    sample.img_path = str(test_dir / '001.png')

    clip = model._to_clip(current.unsqueeze(0), [sample])

    left = torch.full((1, 2, 2), (64.0 - 127.5) / 127.5)
    # Cross-split train-good lookup scales the bad-sequence index into train/good.
    right = torch.full((1, 2, 2), (191.0 - 127.5) / 127.5)
    assert torch.allclose(clip[0, :, 0], left)
    assert torch.allclose(clip[0, :, 1], current)
    assert torch.allclose(clip[0, :, 2], right)


def test_memae_test_good_reference_window_uses_test_good_context(tmp_path):
    import cv2

    test_good_dir = tmp_path / 'bottle' / 'test' / 'good'
    test_bad_dir = tmp_path / 'bottle' / 'test' / 'broken_large'
    test_good_dir.mkdir(parents=True)
    test_bad_dir.mkdir(parents=True)

    for index, value in enumerate([0, 64, 127, 191, 255]):
        array = torch.full((2, 2, 3), value, dtype=torch.uint8).numpy()
        cv2.imwrite(str(test_good_dir / f'{index:03d}.png'), array)
    for index, value in enumerate([32, 96, 160]):
        array = torch.full((2, 2, 3), value, dtype=torch.uint8).numpy()
        cv2.imwrite(str(test_bad_dir / f'{index:03d}.png'), array)

    current = torch.full((1, 2, 2), (160.0 - 127.5) / 127.5)

    cfg = dict(
        type='MemAEDetector',
        in_channels=1,
        frame_num=3,
        clip_mode='test_good_reference_window',
        mem_dim=8,
        clip_neighbor_window_radius=1,
    )
    model = MODELS.build(cfg)
    sample = ADDataSample()
    sample.img_path = str(test_bad_dir / '001.png')

    clip = model._to_clip(current.unsqueeze(0), [sample])

    left = torch.full((1, 2, 2), (0.0 - 127.5) / 127.5)
    # Test-good lookup keeps the same local index within test/good siblings.
    right = torch.full((1, 2, 2), (127.0 - 127.5) / 127.5)
    assert torch.allclose(clip[0, :, 0], left)
    assert torch.allclose(clip[0, :, 1], current)
    assert torch.allclose(clip[0, :, 2], right)


def test_memae_invalid_score_modes_raise():
    with pytest.raises(ValueError, match='Unsupported temporal_reduce_mode'):
        MODELS.build(
            dict(
                type='MemAEDetector',
                in_channels=1,
                frame_num=1,
                mem_dim=8,
                temporal_reduce_mode='median',
            )
        )
    with pytest.raises(ValueError, match='Unsupported image_score_mode'):
        MODELS.build(
            dict(
                type='MemAEDetector',
                in_channels=1,
                frame_num=1,
                mem_dim=8,
                image_score_mode='bad_mode',
            )
        )
    with pytest.raises(ValueError, match='clip_jitter_strength'):
        MODELS.build(
            dict(
                type='MemAEDetector',
                in_channels=1,
                frame_num=1,
                mem_dim=8,
                clip_jitter_strength=-0.1,
            )
        )
    with pytest.raises(ValueError, match='clip_schedule_strength'):
        MODELS.build(
            dict(
                type='MemAEDetector',
                in_channels=1,
                frame_num=1,
                mem_dim=8,
                clip_schedule_strength=-0.1,
            )
        )
    with pytest.raises(ValueError, match='clip_schedule_blur_kernel_size'):
        MODELS.build(
            dict(
                type='MemAEDetector',
                in_channels=1,
                frame_num=1,
                mem_dim=8,
                clip_schedule_blur_kernel_size=4,
            )
        )
    with pytest.raises(ValueError, match='clip_window_scale_min'):
        MODELS.build(
            dict(
                type='MemAEDetector',
                in_channels=1,
                frame_num=1,
                mem_dim=8,
                clip_window_scale_min=1.5,
            )
        )
    with pytest.raises(ValueError, match='clip_window_translation_max'):
        MODELS.build(
            dict(
                type='MemAEDetector',
                in_channels=1,
                frame_num=1,
                mem_dim=8,
                clip_window_translation_max=-0.1,
            )
        )
    with pytest.raises(ValueError, match='clip_neighbor_window_radius'):
        MODELS.build(
            dict(
                type='MemAEDetector',
                in_channels=1,
                frame_num=1,
                mem_dim=8,
                clip_neighbor_window_radius=-1,
            )
        )


def test_memae_alignment_config_matches_official_migration_settings():
    cfg = Config.fromfile(ROOT / 'configs' / 'memae' / 'memae_wrn50_256_mvtec.py')
    assert cfg.model.type == 'MemAEDetector'
    assert cfg.model.in_channels == 3
    assert cfg.model.frame_num == 16
    assert cfg.model.clip_mode == 'repeat_image'
    assert cfg.model.mem_dim == 2000
    assert cfg.model.shrink_thres == 0.0025
    assert cfg.model.entropy_loss_weight == 0.0002
    assert cfg.model.temporal_reduce_mode == 'mean'
    assert cfg.model.image_score_mode == 'spatiotemporal_mean'
    assert cfg.optim_wrapper.optimizer.type == 'Adam'
    assert cfg.optim_wrapper.optimizer.lr == 1e-4
    assert cfg.optim_wrapper.optimizer.weight_decay == 0
    assert cfg.param_scheduler == []
    assert cfg.train_cfg.max_epochs == 100
    assert cfg.train_dataloader.dataset.pipeline[3].type == 'NormalizeAD'
    assert tuple(cfg.train_dataloader.dataset.pipeline[3].mean) == (127.5, 127.5, 127.5)
    assert tuple(cfg.train_dataloader.dataset.pipeline[3].std) == (127.5, 127.5, 127.5)


def test_memae_adapted_config_is_explicitly_non_strict_candidate():
    cfg = Config.fromfile(ROOT / 'configs' / 'memae' / 'memae_wrn50_256_mvtec_adapted.py')
    assert cfg.model.type == 'MemAEDetector'
    assert cfg.model.in_channels == 3
    assert cfg.model.frame_num == 16
    assert cfg.model.clip_mode == 'centered_local_motion_window'
    assert cfg.model.clip_window_scale_min == 0.98
    assert cfg.model.clip_window_translation_max == 0.02
    assert cfg.model.mem_dim == 2000
    assert cfg.model.shrink_thres == 0.0025
    assert cfg.model.entropy_loss_weight == 0.0002
    assert cfg.model.temporal_reduce_mode == 'mean'
    assert cfg.model.image_score_mode == 'map_topk_mean'
    assert cfg.model.topk_ratio == 0.01
    assert cfg.optim_wrapper.optimizer.type == 'Adam'
    assert cfg.optim_wrapper.optimizer.lr == 1e-4
    assert cfg.param_scheduler == []
    assert cfg.train_cfg.max_epochs == 100


def test_memae_ucsdped2_official_config_matches_video_strict_settings():
    cfg = Config.fromfile(ROOT / 'configs' / 'memae' / 'memae_ucsdped2_256_official.py')
    assert cfg.model.type == 'MemAEDetector'
    assert cfg.model.in_channels == 1
    assert cfg.model.frame_num == 16
    assert cfg.model.mem_dim == 2000
    assert cfg.model.shrink_thres == 0.0025
    assert cfg.model.entropy_loss_weight == 0.0002
    assert cfg.model.image_score_mode == 'spatiotemporal_mean'
    assert cfg.train_dataloader.dataset.type == 'MemAEOfficialClipDataset'
    assert cfg.train_dataloader.dataset.data_root == 'data/memae_video/UCSD_P2_256'
    assert cfg.train_dataloader.dataset.split == 'train'
    assert cfg.train_dataloader.sampler.type == 'MemAEOfficialOrderSampler'
    assert cfg.train_dataloader.sampler.epochs == 100
    assert cfg.train_dataloader.sampler.seed == 1
    assert cfg.train_dataloader.sampler.round_up is False
    assert cfg.train_dataloader.sampler.in_channels == 1
    assert cfg.train_dataloader.sampler.mem_dim == 2000
    assert cfg.train_dataloader.sampler.shrink_thres == 0.0025
    assert cfg.test_dataloader.dataset.split == 'test'
    assert cfg.test_evaluator.type == 'MemAEVideoMetric'
    assert cfg.randomness.seed == 1
    assert cfg.randomness.deterministic is True


def test_memae_ucsdped1_official_config_matches_video_strict_settings():
    cfg = Config.fromfile(ROOT / 'configs' / 'memae' / 'memae_ucsdped1_256_official.py')
    assert cfg.train_dataloader.dataset.type == 'MemAEOfficialClipDataset'
    assert cfg.train_dataloader.dataset.data_root == 'data/memae_video/UCSD_P1_256'
    assert cfg.train_dataloader.dataset.dataset_name == 'UCSDped1'
    assert cfg.train_dataloader.sampler.type == 'MemAEOfficialOrderSampler'
    assert cfg.train_dataloader.sampler.seed == 1
    assert cfg.model.in_channels == 1
    assert cfg.test_evaluator.type == 'MemAEVideoMetric'
    assert cfg.randomness.seed == 1


def test_memae_avenue_official_config_matches_video_strict_settings():
    cfg = Config.fromfile(ROOT / 'configs' / 'memae' / 'memae_avenue_256_official.py')
    assert cfg.train_dataloader.dataset.type == 'MemAEOfficialClipDataset'
    assert cfg.train_dataloader.dataset.data_root == 'data/memae_video/Avenue_256'
    assert cfg.train_dataloader.dataset.dataset_name == 'Avenue'
    assert cfg.train_dataloader.sampler.type == 'MemAEOfficialOrderSampler'
    assert cfg.train_dataloader.sampler.seed == 1
    assert cfg.model.in_channels == 1
    assert cfg.test_evaluator.type == 'MemAEVideoMetric'
    assert cfg.randomness.seed == 1
