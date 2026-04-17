"""Tests for the strict RD++ train loop."""

from baoiad.engine.loops.rdpp_train_loop import DEFAULT_RDPP_CATEGORY_EPOCHS, RDPPTrainLoop


def test_rdpp_train_loop_resolves_official_category_epochs():
    assert RDPPTrainLoop.resolve_max_epochs(['bottle'], DEFAULT_RDPP_CATEGORY_EPOCHS, 300) == 200
    assert RDPPTrainLoop.resolve_max_epochs(['carpet'], DEFAULT_RDPP_CATEGORY_EPOCHS, 300) == 10
    assert RDPPTrainLoop.resolve_max_epochs(['unknown'], DEFAULT_RDPP_CATEGORY_EPOCHS, 300) == 300
    assert RDPPTrainLoop.resolve_max_epochs(['bottle', 'cable'], DEFAULT_RDPP_CATEGORY_EPOCHS, 300) == 300
