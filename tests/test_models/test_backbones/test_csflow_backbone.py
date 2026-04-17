"""Tests for CSFlowFeatureExtractor."""

from unittest.mock import patch

import torch
from torchvision.models import efficientnet_b5

from baoiad.models.backbones.csflow_backbone import CSFlowFeatureExtractor


def _efficientnet_b5_no_weights(*args, **kwargs):
    """Return a deterministic local backbone without downloading weights."""
    return efficientnet_b5(weights=None)


def test_feature_slice_matches_features_6_8_node():
    with patch('torchvision.models.efficientnet_b5', new=_efficientnet_b5_no_weights):
        extractor = CSFlowFeatureExtractor(n_scales=1, input_size=(64, 64), frozen=True)

    reference_backbone = efficientnet_b5(weights=None)
    reference_backbone.features[:7].load_state_dict(extractor.features.state_dict())
    reference_backbone.eval()

    captured = {}

    def _capture_output(_module, _inputs, output):
        captured['features_6_8'] = output.detach()

    hook = reference_backbone.features[6][8].register_forward_hook(_capture_output)
    inputs = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        extracted = extractor(inputs)[0]
        reference_backbone.features[:7](inputs)

    hook.remove()

    assert 'features_6_8' in captured
    assert torch.equal(extracted, captured['features_6_8'])
    assert tuple(extracted.shape) == (1, 304, 2, 2)


def test_frozen_extractor_stays_in_eval_mode():
    with patch('torchvision.models.efficientnet_b5', new=_efficientnet_b5_no_weights):
        extractor = CSFlowFeatureExtractor(n_scales=1, input_size=(64, 64), frozen=True)

    extractor.train()

    assert extractor.training is False
    assert not any(parameter.requires_grad for parameter in extractor.parameters())
