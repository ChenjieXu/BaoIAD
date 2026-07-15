"""Tests for the version compatibility release gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tools" / "check_version_compatibility.py"
MANIFEST = ROOT / "docs" / "alignment" / "v1_0_0_compatibility.json"


def _run_checker(manifest: Path = MANIFEST) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--repo-root",
            str(ROOT),
            "--manifest",
            str(manifest),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_repository_version_compatibility_contract_passes():
    result = _run_checker()
    report = json.loads(result.stdout)

    assert result.returncode == 0, report["errors"]
    assert report["ok"] is True
    assert report["live_tag"] == {
        "status": "verified",
        "peeled_commit": "697fc4304cc76876d397067e2706ed771f62e708",
    }


def test_tampered_current_version_fails_closed(tmp_path):
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    document["current"]["package_version"] = "9.9.9"
    tampered = tmp_path / "v1_0_0_compatibility.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")

    result = _run_checker(tampered)
    report = json.loads(result.stdout)

    assert result.returncode == 1
    assert report["ok"] is False
    assert "current.package_version must be '1.1.0'" in report["errors"]
    assert report["live_tag"]["status"] == "not-run"
