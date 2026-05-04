"""Tests for DeSTSegDetector."""

from unittest import TestCase

import torch
from mmengine.optim import OptimWrapper, OptimWrapperDict

import baoiad  # noqa: F401
from baoiad.engine.optimizers.destseg_optim_wrapper_constructor import DeSTSegOptimWrapperConstructor
from baoiad.registry import MODELS
from baoiad.structures import ADDataSample


def _make_train_batch(batch_size, height=64, width=64):
    aug = torch.randn(batch_size, 3, height, width)
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{i}.png',
            'defect_type': 'synthetic',
            'img_origin': torch.randn(3, height, width),
            'img_aug': aug[i].clone(),
        })
        mask = torch.zeros(height, width)
        mask[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = 1.0
        sample.gt_label = 1
        sample.gt_mask = mask
        samples.append(sample)
    return aug, samples


def _make_predict_samples(batch_size, height=64, width=64):
    samples = []
    for i in range(batch_size):
        sample = ADDataSample()
        sample.set_metainfo({
            'cls_name': 'bottle',
            'img_path': f'/fake/{i}.png',
            'defect_type': 'good',
        })
        sample.gt_label = 0
        sample.gt_mask = torch.zeros(height, width)
        samples.append(sample)
    return samples


def _clone_params(module):
    return [param.detach().clone() for param in module.parameters()]


def _any_param_changed(before, module):
    after = [param.detach() for param in module.parameters()]
    return any(not torch.allclose(prev, curr) for prev, curr in zip(before, after))


class TestDeSTSegDetector(TestCase):
    def setUp(self):
        self.cfg = dict(
            type='DeSTSegDetector',
            backbone='resnet18',
            teacher_pretrained=False,
            de_st_steps=1000,
        )

    def test_forward_tensor(self):
        model = MODELS.build(self.cfg)
        model.eval()
        out = model(torch.randn(2, 3, 64, 64), mode='tensor')
        assert out is not None
        assert len(out) == 3

    def test_forward_loss_student_phase(self):
        model = MODELS.build(self.cfg)
        model.train()
        inputs, data_samples = _make_train_batch(2)
        out = model(inputs, data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert torch.isfinite(out['loss'])

    def test_forward_loss_segmentation_phase(self):
        model = MODELS.build(self.cfg)
        model.train()
        model.set_iter_info(1000, 5000)
        inputs, data_samples = _make_train_batch(2)
        out = model(inputs, data_samples, mode='loss')
        assert isinstance(out, dict)
        assert 'loss' in out
        assert torch.isfinite(out['loss'])

    def test_forward_predict(self):
        model = MODELS.build(self.cfg)
        model.eval()
        data_samples = _make_predict_samples(2)
        out = model(torch.randn(2, 3, 64, 64), data_samples, mode='predict')
        assert isinstance(out, list)
        assert len(out) == 2
        assert torch.isfinite(torch.tensor([sample.pred_score for sample in out])).all()

    def test_set_iter_info_switches_to_segmentation_phase(self):
        model = MODELS.build(self.cfg)
        model.train()
        model.set_iter_info(0, 5000)
        assert model._phase == 'student'
        assert all(not param.requires_grad for param in model.segmentation_net.parameters())

        model.set_iter_info(1000, 5000)
        assert model._phase == 'segmentation'
        assert all(not param.requires_grad for param in model.student_net.parameters())
        assert any(param.requires_grad for param in model.segmentation_net.parameters())

    def test_train_step_with_multi_optimizer_across_phases(self):
        model = MODELS.build(self.cfg)
        model.train()
        inputs, data_samples = _make_train_batch(2)
        optim_wrapper = OptimWrapperDict(
            student=OptimWrapper(
                torch.optim.SGD(model.student_net.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
            ),
            segmentation=OptimWrapper(
                torch.optim.SGD(
                    [
                        dict(params=model.segmentation_net.res.parameters(), lr=0.1),
                        dict(params=model.segmentation_net.head.parameters(), lr=0.01),
                    ],
                    lr=0.01,
                    momentum=0.9,
                    weight_decay=1e-4,
                )
            ),
        )

        model.set_iter_info(0, 5000)
        outputs_student = model.train_step(dict(inputs=inputs, data_samples=data_samples), optim_wrapper)
        assert 'loss' in outputs_student
        assert torch.isfinite(outputs_student['loss'])

        model.set_iter_info(1000, 5000)
        outputs_seg = model.train_step(dict(inputs=inputs, data_samples=data_samples), optim_wrapper)
        assert 'loss' in outputs_seg
        assert torch.isfinite(outputs_seg['loss'])

    def test_train_step_updates_only_student_params_in_student_phase(self):
        model = MODELS.build(self.cfg)
        model.train()
        inputs, data_samples = _make_train_batch(2)
        optim_wrapper = OptimWrapperDict(
            student=OptimWrapper(
                torch.optim.SGD(model.student_net.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
            ),
            segmentation=OptimWrapper(
                torch.optim.SGD(
                    [
                        dict(params=model.segmentation_net.res.parameters(), lr=0.1),
                        dict(params=model.segmentation_net.head.parameters(), lr=0.01),
                    ],
                    lr=0.01,
                    momentum=0.9,
                    weight_decay=1e-4,
                )
            ),
        )

        student_before = _clone_params(model.student_net)
        segmentation_before = _clone_params(model.segmentation_net)

        model.set_iter_info(0, 5000)
        model.train_step(dict(inputs=inputs, data_samples=data_samples), optim_wrapper)

        assert _any_param_changed(student_before, model.student_net)
        assert not _any_param_changed(segmentation_before, model.segmentation_net)

    def test_train_step_updates_only_segmentation_params_in_segmentation_phase(self):
        model = MODELS.build(self.cfg)
        model.train()
        inputs, data_samples = _make_train_batch(2)
        optim_wrapper = OptimWrapperDict(
            student=OptimWrapper(
                torch.optim.SGD(model.student_net.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
            ),
            segmentation=OptimWrapper(
                torch.optim.SGD(
                    [
                        dict(params=model.segmentation_net.res.parameters(), lr=0.1),
                        dict(params=model.segmentation_net.head.parameters(), lr=0.01),
                    ],
                    lr=0.01,
                    momentum=0.9,
                    weight_decay=1e-4,
                )
            ),
        )

        student_before = _clone_params(model.student_net)
        segmentation_before = _clone_params(model.segmentation_net)

        model.set_iter_info(1000, 5000)
        model.train_step(dict(inputs=inputs, data_samples=data_samples), optim_wrapper)

        assert not _any_param_changed(student_before, model.student_net)
        assert _any_param_changed(segmentation_before, model.segmentation_net)

    def test_constructor_built_wrappers_update_real_model_params(self):
        model = MODELS.build(self.cfg)
        model.train()
        inputs, data_samples = _make_train_batch(2)

        constructor = DeSTSegOptimWrapperConstructor(
            dict(
                student=dict(optimizer=dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=1e-4)),
                segmentation=dict(
                    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
                    res_lr=0.1,
                    head_lr=0.01,
                ),
            )
        )
        optim_wrapper = constructor(model)

        model.set_iter_info(1000, 5000)
        segmentation_before = _clone_params(model.segmentation_net)
        model.train_step(dict(inputs=inputs, data_samples=data_samples), optim_wrapper)

        assert _any_param_changed(segmentation_before, model.segmentation_net)
