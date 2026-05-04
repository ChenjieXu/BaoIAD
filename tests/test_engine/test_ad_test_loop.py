"""Tests for ADTestLoop."""

import baoiad  # noqa: F401

from baoiad.engine.loops.ad_test_loop import ADTestLoop
from baoiad.registry import LOOPS


class TestADTestLoop:
    def test_registered(self):
        assert LOOPS.get('ADTestLoop') is ADTestLoop
