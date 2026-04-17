"""Tests for the AnomalyDINO variant comparison tool."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_payload(path: Path, category: str, img: float, pxl: float) -> None:
    payload = {
        'anomalydino': {
            category: {
                'image_auroc': img,
                'pixel_auroc': pxl,
            },
            '_average': {
                'image_auroc': img,
                'pixel_auroc': pxl,
                'num_categories': 1,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding='utf-8')


def test_anomalydino_compare_variants(tmp_path: Path) -> None:
    baseline = tmp_path / 'baseline.json'
    variant = tmp_path / 'variant.json'
    output = tmp_path / 'compare.json'
    _write_payload(baseline, 'screw', 0.88, 0.96)
    _write_payload(variant, 'screw', 0.90, 0.95)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'tools' / 'anomalydino_compare_variants.py'),
            '--baseline',
            str(baseline),
            '--variant',
            f'candidate={variant}',
            '--output',
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )

    payload = json.loads(output.read_text())
    screw = payload['variants']['candidate']['per_category']['screw']
    assert screw['image_delta'] == 0.02
    assert screw['pixel_delta'] == -0.01
