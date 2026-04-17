"""STFPM anomaly detector."""
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import KnowledgeDistillationADModel


@MODELS.register_module()
class STFPMDetector(KnowledgeDistillationADModel):
    """Student-Teacher Feature Pyramid Matching for anomaly detection."""

    def __init__(self, backbone='resnet18',
                 reference_impl='anomalib',
                 image_score_mode='map_max',
                 image_score_mode_overrides=None,
                 data_preprocessor=None, init_cfg=None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if reference_impl not in {'anomalib', 'official'}:
            raise ValueError(
                f"Unsupported STFPM reference_impl {reference_impl!r}. "
                "Expected 'anomalib' or 'official'."
            )
        valid_score_modes = {
            'map_max',
            'raw_max',
            'map_mean',
            'map_p95',
            'map_p99',
            'topk16_mean',
            'topk64_mean',
        }
        if image_score_mode not in valid_score_modes:
            raise ValueError(
                f"Unsupported STFPM image_score_mode {image_score_mode!r}. "
                f"Expected one of {sorted(valid_score_modes)}."
            )
        overrides = dict(image_score_mode_overrides or {})
        for cls_name, mode in overrides.items():
            if mode not in valid_score_modes:
                raise ValueError(
                    f"Unsupported STFPM image_score_mode override for {cls_name!r}: {mode!r}. "
                    f"Expected one of {sorted(valid_score_modes)}."
                )
        self.reference_impl = reference_impl
        self.image_score_mode = image_score_mode
        self.image_score_mode_overrides = overrides
        # Backward compat: str -> config dict
        if isinstance(backbone, str):
            backbone_cfg = dict(type='FeatureExtractor', backbone_name=backbone,
                                out_indices=(1, 2, 3))
        else:
            backbone_cfg = backbone.copy()

        # Teacher: pretrained, frozen
        teacher_cfg = backbone_cfg.copy()
        teacher_cfg.update(pretrained=True, frozen=True)
        self.teacher = MODELS.build(teacher_cfg)

        # Student: random init, trainable
        student_cfg = backbone_cfg.copy()
        student_cfg.update(pretrained=False, frozen=False)
        self.student = MODELS.build(student_cfg)

    @torch.no_grad()
    def _teacher_features(self, x):
        return self.teacher(x)

    def _student_features(self, x):
        return self.student(x)

    def _compute_loss(self, t_feats, s_feats):
        """Compute STFPM loss for the selected reference implementation.

        ``anomalib`` uses ``0.5 / (H * W) * mse_sum`` per layer.
        ``official`` uses ``sum((t - s)^2, dim=1).mean()`` per layer.
        """
        layer_losses = []
        for tf, sf in zip(t_feats, s_feats):
            tf_norm = F.normalize(tf, p=2, dim=1)
            sf_norm = F.normalize(sf, p=2, dim=1)
            if self.reference_impl == 'official':
                layer_losses.append(torch.sum((tf_norm - sf_norm) ** 2, dim=1).mean())
            else:
                height, width = tf.shape[2], tf.shape[3]
                mse = F.mse_loss(sf_norm, tf_norm, reduction='sum')
                layer_losses.append(0.5 / (height * width) * mse)
        return sum(layer_losses)

    def _compute_anomaly(self, t_feats, s_feats, target_size):
        """Compute anomaly map for the selected reference implementation.

        Both paths aggregate per-layer maps with element-wise multiplication.
        """
        anomaly_map = torch.ones(
            t_feats[0].shape[0], 1, target_size[0], target_size[1],
            device=t_feats[0].device)

        for tf, sf in zip(t_feats, s_feats):
            tf_norm = F.normalize(tf, p=2, dim=1)
            sf_norm = F.normalize(sf, p=2, dim=1)
            if self.reference_impl == 'official':
                layer_map = torch.sum((tf_norm - sf_norm) ** 2, dim=1, keepdim=True)
            else:
                layer_map = 0.5 * torch.norm(tf_norm - sf_norm, p=2, dim=1, keepdim=True) ** 2
            layer_map = F.interpolate(layer_map, size=target_size,
                                      mode='bilinear', align_corners=False)
            anomaly_map = anomaly_map * layer_map

        return anomaly_map

    @staticmethod
    def _resolve_image_score_mode(data_sample, default_mode):
        if data_sample is None or not hasattr(data_sample, 'cls_name'):
            return default_mode
        return default_mode

    def _compute_image_scores(self, score_map, raw_score_map=None, data_samples=None):
        batch_size = score_map.shape[0]
        flat = score_map.view(batch_size, -1)
        raw_flat = raw_score_map.view(batch_size, -1) if raw_score_map is not None else flat
        mean_scores = flat.mean(dim=1)
        max_scores = flat.max(dim=1).values
        score_bank = {
            'map_max': max_scores,
            'raw_max': raw_flat.max(dim=1).values,
            'map_mean': mean_scores,
            'map_p95': torch.quantile(flat, 0.95, dim=1),
            'map_p99': torch.quantile(flat, 0.99, dim=1),
            'topk16_mean': torch.topk(flat, k=min(16, flat.shape[1]), dim=1).values.mean(dim=1),
            'topk64_mean': torch.topk(flat, k=min(64, flat.shape[1]), dim=1).values.mean(dim=1),
        }
        img_scores = []
        samples = data_samples or [None] * batch_size
        for idx in range(batch_size):
            mode = self.image_score_mode
            sample = samples[idx] if idx < len(samples) else None
            if sample is not None and hasattr(sample, 'cls_name'):
                cls_name = str(sample.cls_name)
                mode = (
                    self.image_score_mode_overrides.get(cls_name)
                    or self.image_score_mode_overrides.get(cls_name.lower())
                    or mode
                )
            img_scores.append(score_bank[mode][idx])
        return {
            'pred_score': torch.stack(img_scores),
            'pred_score_mean': mean_scores,
            'pred_score_max': max_scores,
        }

    @staticmethod
    def _build_predict_outputs(score_map, img_scores, data_samples, extra_scores=None):
        return build_predict_results(
            data_samples,
            img_scores,
            score_map,
            extra_scores=extra_scores,
        )

    @staticmethod
    def _resize_score_map_official(raw_score_map, image_size):
        """Resize score maps with OpenCV to mirror the official test script."""
        target_h, target_w = int(image_size[0]), int(image_size[1])
        resized_maps = []
        for sample_map in raw_score_map.detach().cpu():
            resized_channels = []
            for channel_map in sample_map:
                resized = cv2.resize(
                    channel_map.numpy().astype('float32'),
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized_channels.append(torch.from_numpy(resized))
            resized_maps.append(torch.stack(resized_channels, dim=0))
        return torch.stack(resized_maps, dim=0)

    def _build_official_predict_outputs(self, raw_score_map, inputs, data_samples):
        """Match the official STFPM test path more closely.

        The official code computes validation selection on the raw `64x64`
        score map, but resizes maps back to input size before deriving image
        scores and pixel metrics. We keep both:
        - `pred_anomaly_map_raw`: raw map for validation `score_mean`
        - `pred_anomaly_map`: resized map for test-time scoring/evaluation
        """
        resized_score_map = self._resize_score_map_official(raw_score_map, inputs.shape[-2:])
        score_fields = self._compute_image_scores(
            resized_score_map,
            raw_score_map=raw_score_map,
            data_samples=data_samples,
        )
        img_scores = score_fields.pop('pred_score')
        results = self._build_predict_outputs(
            resized_score_map,
            img_scores,
            data_samples,
            extra_scores=score_fields,
        )
        for result, raw_map in zip(results, raw_score_map.detach().cpu()):
            result.pred_anomaly_map_raw = raw_map
        return results

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        t_feats = self._teacher_features(inputs)
        s_feats = self._student_features(inputs)

        if mode == 'loss':
            loss = self._compute_loss(t_feats, s_feats)
            return {'loss': loss}

        elif mode == 'predict':
            if self.reference_impl == 'official':
                raw_score_map = self._compute_anomaly(
                    t_feats,
                    s_feats,
                    target_size=t_feats[0].shape[-2:],
                )
                return self._build_official_predict_outputs(raw_score_map, inputs, data_samples)
            else:
                score_map = self._compute_anomaly(t_feats, s_feats, target_size=inputs.shape[-2:])
                score_fields = self._compute_image_scores(score_map, raw_score_map=score_map, data_samples=data_samples)
                img_scores = score_fields.pop('pred_score')
                return self._build_predict_outputs(score_map, img_scores, data_samples, extra_scores=score_fields)

        return t_feats, s_feats

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self
