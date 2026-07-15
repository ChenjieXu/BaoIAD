"""Tests for release-candidate checks not covered by the public diff policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "check_release_candidate.py"
SPEC = importlib.util.spec_from_file_location("check_release_candidate", SCRIPT)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_secret_patterns_fail_closed_without_echoing_secret(tmp_path):
    secret = "gh" + "p_" + "A" * 24
    path = tmp_path / "settings.py"
    path.write_text(f"credential = {secret!r}\n", encoding="utf-8")

    errors = CHECKER.validate_secret_patterns(tmp_path, ["settings.py"], 1024)

    assert errors == ["secret pattern github-token in settings.py:1"]
    assert secret not in errors[0]


@pytest.mark.parametrize(
    "relative",
    ["deploy.sh", ".env", ".credentials", "Dockerfile", "payload.bin"],
)
def test_secret_scan_covers_text_without_an_extension_allowlist(tmp_path, relative):
    secret = "gh" + "p_" + "B" * 24
    (tmp_path / relative).write_text(f"export API_TOKEN={secret}\n", encoding="utf-8")

    errors = CHECKER.validate_secret_patterns(tmp_path, [relative], 1024)

    assert errors == [f"secret pattern github-token in {relative}:1"]
    assert secret not in errors[0]


@pytest.mark.parametrize(
    "content",
    [
        b"\0ghp_" + b"C" * 24,
        b"\xffghp_" + b"D" * 24,
    ],
    ids=["nul-binary", "invalid-utf8-binary"],
)
def test_secret_scan_safely_skips_binary_files(tmp_path, content):
    (tmp_path / "artifact").write_bytes(content)

    assert CHECKER.validate_secret_patterns(tmp_path, ["artifact"], 1024) == []


def test_secret_scan_fails_closed_on_oversized_text_without_leaking(tmp_path):
    secret = "gh" + "p_" + "E" * 24
    (tmp_path / "large.env").write_text(
        "padding=" + "x" * 64 + f"\ntoken={secret}\n", encoding="utf-8"
    )

    errors = CHECKER.validate_secret_patterns(tmp_path, ["large.env"], 32)

    assert errors == ["secret scan refuses changed file over 32 bytes: large.env"]
    assert secret not in errors[0]


def test_secret_scan_fails_closed_on_changed_symlink_without_following(tmp_path):
    secret = "gh" + "p_" + "F" * 24
    outside = tmp_path.parent / "outside-secret.env"
    outside.write_text(f"token={secret}\n", encoding="utf-8")
    (tmp_path / "linked.env").symlink_to(outside)

    errors = CHECKER.validate_secret_patterns(tmp_path, ["linked.env"], 1024)

    assert errors == ["secret scan refuses changed symlink: linked.env"]
    assert secret not in errors[0]


def test_secret_scan_requires_an_explicit_positive_maximum(tmp_path):
    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        CHECKER.validate_secret_patterns(tmp_path, [], 0)


def test_missing_local_markdown_and_rst_links_fail(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(
        "[present](present.md) [missing](missing.md)\n", encoding="utf-8"
    )
    (docs / "present.md").write_text("# Present\n", encoding="utf-8")
    (docs / "index.rst").write_text(
        ".. toctree::\n\n   present\n   absent\n", encoding="utf-8"
    )

    errors = CHECKER.validate_local_links(
        tmp_path, ["docs/README.md", "docs/present.md", "docs/index.rst"]
    )

    assert "broken local link in docs/README.md:1: missing.md" in errors
    assert "broken local link in docs/index.rst:4: absent" in errors
    assert not any("present" in error for error in errors)


def test_changed_file_size_applies_to_existing_files(tmp_path):
    path = tmp_path / "existing.bin"
    path.write_bytes(b"x" * 17)

    errors = CHECKER.validate_changed_file_sizes(tmp_path, ["existing.bin"], 16)

    assert errors == ["changed file exceeds 16 bytes: existing.bin (17 bytes)"]


def test_local_link_cannot_escape_repository(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("[escape](../../outside.md)\n", encoding="utf-8")

    errors = CHECKER.validate_local_links(tmp_path, ["docs/README.md"])

    assert errors == ["broken local link in docs/README.md:1: ../../outside.md"]
