"""Tests for dataset transforms: LoadImage, LoadMask, ResizeAD, NormalizeAD, PackADInputs."""

import numpy as np
import pytest

import baoiad  # noqa: F401

from baoiad.datasets.transforms import (
    CenterCrop,
    GenerateRDPPNoise,
    LoadImage,
    LoadMask,
    NormalizeAD,
    OpenCLIPPreprocessAD,
    ResizeAD,
    ThresholdMask,
)


class TestLoadImage:
    def test_load_image(self, tmp_path):
        import cv2

        img = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
        path = str(tmp_path / 'test.png')
        cv2.imwrite(path, img)

        t = LoadImage()
        results = t.transform({'img_path': path})
        assert results['img'].shape == (64, 48, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (64, 48)

    def test_load_image_keep_bgr_copy(self, tmp_path):
        import cv2

        bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        bgr[..., 0] = 11
        bgr[..., 1] = 22
        bgr[..., 2] = 33
        path = str(tmp_path / 'bgr.png')
        cv2.imwrite(path, bgr)

        t = LoadImage(to_rgb=False, keep_bgr_copy=True)
        results = t.transform({'img_path': path})

        assert np.array_equal(results['img'], bgr)
        assert np.array_equal(results['ori_img_bgr'], bgr)

    def test_load_image_float32(self, tmp_path):
        import cv2

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'f.png'), img)

        t = LoadImage(to_float32=True)
        results = t.transform({'img_path': str(tmp_path / 'f.png')})
        assert results['img'].dtype == np.float32

    def test_load_missing_raises(self):
        t = LoadImage()
        with pytest.raises(FileNotFoundError):
            t.transform({'img_path': '/nonexistent/img.png'})


class TestLoadMask:
    def test_with_mask(self, tmp_path):
        import cv2

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        path = str(tmp_path / 'mask.png')
        cv2.imwrite(path, mask)

        t = LoadMask()
        results = t.transform({
            'gt_mask_path': path,
            'img_shape': (32, 32),
        })
        assert results['gt_mask'].dtype == np.float32
        assert results['gt_mask'].max() == 1.0
        assert results['gt_mask'].shape == (32, 32)

    def test_without_mask(self):
        t = LoadMask()
        results = t.transform({
            'gt_mask_path': '',
            'img_shape': (64, 64),
        })
        assert results['gt_mask'].shape == (64, 64)
        assert results['gt_mask'].sum() == 0

    def test_load_mask_can_preserve_grayscale_values(self, tmp_path):
        import cv2

        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 128
        path = str(tmp_path / 'mask_gray.png')
        cv2.imwrite(path, mask)

        t = LoadMask(to_binary=False)
        results = t.transform({
            'gt_mask_path': path,
            'img_shape': (16, 16),
        })
        assert results['gt_mask'].dtype == np.float32
        assert results['gt_mask'].max() == pytest.approx(128 / 255.0)


class TestResizeAD:
    @pytest.mark.parametrize('size', [64, 128, 256, (128, 64)])
    def test_resize(self, size):
        t = ResizeAD(size=size)
        img = np.random.rand(256, 256, 3).astype(np.float32)
        mask = np.zeros((256, 256), dtype=np.float32)
        results = t.transform({'img': img, 'gt_mask': mask})
        expected = (size, size) if isinstance(size, int) else size
        assert results['img'].shape[:2] == expected
        assert results['gt_mask'].shape == expected
        assert results['img_shape'] == expected

    def test_resize_without_mask(self):
        t = ResizeAD(size=64)
        results = t.transform({'img': np.random.rand(128, 128, 3).astype(np.float32)})
        assert results['img'].shape[:2] == (64, 64)

    def test_resize_grayscale_image(self):
        t = ResizeAD(size=32)
        results = t.transform({'img': np.random.randint(0, 255, (64, 64), dtype=np.uint8)})
        assert results['img'].shape == (32, 32)

    def test_resize_preserves_channels(self):
        t = ResizeAD(size=32)
        results = t.transform({'img': np.random.rand(64, 64, 3).astype(np.float32)})
        assert results['img'].shape == (32, 32, 3)

    def test_resize_keep_ratio_matches_torchvision_style(self):
        t = ResizeAD(size=128, keep_ratio=True)
        img = np.random.randint(0, 255, (60, 120, 3), dtype=np.uint8)
        mask = np.zeros((60, 120), dtype=np.float32)
        results = t.transform({'img': img, 'gt_mask': mask})
        assert results['img'].shape[:2] == (128, 256)
        assert results['gt_mask'].shape == (128, 256)
        assert results['img_shape'] == (128, 256)

    def test_resize_mask_with_bilinear_keeps_soft_values(self):
        t = ResizeAD(size=8, backend='pillow', mask_interpolation='bilinear')
        img = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[1:3, 1:3] = 1.0

        results = t.transform({'img': img, 'gt_mask': mask})

        assert results['gt_mask'].shape == (8, 8)
        assert float(results['gt_mask'].max()) <= 1.0
        assert np.any((results['gt_mask'] > 0.0) & (results['gt_mask'] < 1.0))

    def test_resize_keep_ratio_plus_center_crop_matches_torchvision(self):
        from PIL import Image
        from torchvision.transforms import CenterCrop as TVCenterCrop
        from torchvision.transforms import Resize

        rng = np.random.default_rng(23)
        img = rng.integers(0, 255, size=(100, 160, 3), dtype=np.uint8)
        mask = np.zeros((100, 160), dtype=np.float32)

        resize = ResizeAD(size=128, keep_ratio=True, official_pil=True)
        crop = CenterCrop(96)
        normalize = NormalizeAD()
        ours = normalize.transform(crop.transform(resize.transform({
            'img': img.copy(),
            'gt_mask': mask.copy(),
        })))

        pil_img = Image.fromarray(img)
        tensor = Resize(128, antialias=True)(pil_img)
        tensor = TVCenterCrop(96)(tensor)
        tensor = np.asarray(tensor).astype(np.float32)
        tensor = (tensor - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array(
            [58.395, 57.12, 57.375], dtype=np.float32
        )

        np.testing.assert_allclose(
            ours['img'],
            tensor,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_resize_pillow_matches_torchvision_pil_resize(self):
        from PIL import Image
        from torchvision.transforms import Resize

        img = np.random.default_rng(7).integers(0, 255, size=(53, 79, 3), dtype=np.uint8)
        ours = ResizeAD(size=(64, 64), backend='pillow', official_pil=True).transform({'img': img.copy()})['img']
        reference = np.asarray(Resize((64, 64), antialias=True)(Image.fromarray(img)))

        np.testing.assert_array_equal(ours, reference)


class TestNormalizeAD:
    def test_imagenet_normalize(self):
        t = NormalizeAD()
        img = np.ones((32, 32, 3), dtype=np.float32) * 123.675
        results = t.transform({'img': img})
        # First channel: (123.675 - 123.675) / 58.395 ≈ 0
        assert abs(results['img'][:, :, 0].mean()) < 1e-5

    def test_output_is_float(self):
        t = NormalizeAD()
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        results = t.transform({'img': img})
        assert results['img'].dtype == np.float32

    def test_zero_image(self):
        t = NormalizeAD()
        results = t.transform({'img': np.zeros((16, 16, 3), dtype=np.float32)})
        # Should not produce NaN
        assert not np.isnan(results['img']).any()

    def test_normalize_multiple_keys(self):
        t = NormalizeAD(
            mean=(0.5, 0.5, 0.5),
            std=(0.25, 0.25, 0.25),
            keys=('img', 'img_noise'),
        )
        results = t.transform({
            'img': np.ones((4, 4, 3), dtype=np.float32) * 0.5,
            'img_noise': np.ones((4, 4, 3), dtype=np.float32) * 0.75,
        })
        assert np.allclose(results['img'], 0.0)
        assert np.allclose(results['img_noise'], 1.0)


class TestThresholdMask:
    def test_threshold_mask_binarizes_soft_values(self):
        t = ThresholdMask(threshold=0.5)
        mask = np.array([[0.2, 0.5], [0.8, 0.49]], dtype=np.float32)
        results = t.transform({'gt_mask': mask})
        assert np.array_equal(results['gt_mask'], np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))


class TestGenerateRDPPNoise:
    def test_generates_noise_branch(self):
        t = GenerateRDPPNoise()
        img = np.zeros((256, 256, 3), dtype=np.float32)
        results = t.transform({'img': img})
        assert 'img_noise' in results
        assert results['img_noise'].shape == img.shape
        assert results['img_noise'].dtype == np.float32
        assert np.count_nonzero(results['img_noise'] - results['img']) > 0


class TestOpenCLIPPreprocessAD:
    def test_matches_reference_val_preprocess(self):
        from PIL import Image
        from torchvision.transforms import CenterCrop as TVCenterCrop
        from torchvision.transforms import InterpolationMode, Normalize, Resize, ToTensor

        rng = np.random.default_rng(7)
        img = rng.integers(0, 255, size=(71, 93, 3), dtype=np.uint8)
        mask = np.zeros((71, 93), dtype=np.float32)
        mask[12:38, 21:56] = 1.0

        transform = OpenCLIPPreprocessAD(size=64)
        ours = transform.transform({'img': img.copy(), 'gt_mask': mask.copy()})

        expected_img = Normalize(
            mean=OpenCLIPPreprocessAD.OPENAI_DATASET_MEAN,
            std=OpenCLIPPreprocessAD.OPENAI_DATASET_STD,
        )(
            ToTensor()(
                TVCenterCrop((64, 64))(
                    Resize((64, 64), interpolation=InterpolationMode.BICUBIC, antialias=True)(
                        Image.fromarray(img).convert('RGB')
                    )
                )
            )
        )
        expected_mask = ToTensor()(
            TVCenterCrop((64, 64))(
                Resize((64, 64), interpolation=InterpolationMode.BILINEAR, antialias=True)(
                    Image.fromarray((mask * 255.0).astype(np.uint8))
                )
            )
        ).squeeze(0)

        np.testing.assert_allclose(
            ours['img'],
            expected_img.permute(1, 2, 0).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            ours['gt_mask'],
            expected_mask.numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        assert ours['img_shape'] == (64, 64)
