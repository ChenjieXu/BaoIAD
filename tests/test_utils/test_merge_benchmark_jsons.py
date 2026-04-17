"""Tests for benchmark shard merging."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_merge_benchmark_jsons(tmp_path):
    shard1 = tmp_path / 'part1.json'
    shard2 = tmp_path / 'part2.json'
    output = tmp_path / 'merged.json'

    shard1.write_text(json.dumps({
        'glass': {
            'bottle': {'image_auroc': 1.0, 'pixel_auroc': 0.9},
        }
    }))
    shard2.write_text(json.dumps({
        'glass': {
            'cable': {'image_auroc': 0.8, 'pixel_auroc': 0.7},
        }
    }))

    subprocess.check_call([
        sys.executable,
        str(ROOT / 'tools' / 'merge_benchmark_jsons.py'),
        '--method', 'glass',
        '--inputs', str(shard1), str(shard2),
        '--output', str(output),
    ], cwd=ROOT)

    payload = json.loads(output.read_text())
    assert payload['glass']['_average']['image_auroc'] == 0.9
    assert payload['glass']['_average']['pixel_auroc'] == 0.8
    assert payload['glass']['_average']['num_categories'] == 2


def test_merge_benchmark_jsons_prefers_later_retry_shards(tmp_path):
    shard1 = tmp_path / 'part1.json'
    shard2 = tmp_path / 'retry.json'
    output = tmp_path / 'merged.json'

    shard1.write_text(json.dumps({
        'memseg': {
            'bottle': {'image_auroc': 0.4, 'pixel_auroc': 0.5},
            'cable': {'image_auroc': 0.6, 'pixel_auroc': 0.7},
        }
    }))
    shard2.write_text(json.dumps({
        'memseg': {
            'bottle': {'image_auroc': 0.9, 'pixel_auroc': 0.95},
        }
    }))

    subprocess.check_call([
        sys.executable,
        str(ROOT / 'tools' / 'merge_benchmark_jsons.py'),
        '--method', 'memseg',
        '--inputs', str(shard1), str(shard2),
        '--output', str(output),
    ], cwd=ROOT)

    payload = json.loads(output.read_text())
    assert payload['memseg']['bottle']['image_auroc'] == 0.9
    assert payload['memseg']['bottle']['pixel_auroc'] == 0.95
    assert payload['memseg']['cable']['image_auroc'] == 0.6
    assert payload['memseg']['_average']['image_auroc'] == 0.75
    assert payload['memseg']['_average']['num_categories'] == 2
