#!/usr/bin/env python3
"""Run BaoIAD's fail-closed release-candidate verification gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "tools" / "public_release_policy.json"
SECRET_PATTERNS = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    (
        "credential-assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|password|passwd|secret)\b\s*[:=]\s*"
            r"[\"'][^\"'\s]{12,}[\"']"
        ),
    ),
)
MARKDOWN_INLINE_LINK = re.compile(
    r"!?\[[^\]]*\]\(\s*(?:<(?P<angled>[^>]+)>|(?P<plain>[^)\s]+))"
)
MARKDOWN_REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
HTML_LINK = re.compile(r"\b(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
RST_EXPLICIT_LINK = re.compile(r"`[^`<]*<([^>]+)>`_")
RST_DIRECTIVE = re.compile(
    r"^\s*\.\.\s+(?:image|figure|include|literalinclude)::\s+(\S+)",
    re.MULTILINE,
)
RST_DOC_ROLE = re.compile(r":doc:`(?:[^`<]*<)?([^>`]+)>?`")


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_json_command(
    root: Path, command: list[str], name: str
) -> tuple[dict[str, Any], list[str]]:
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    errors: list[str] = []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "ok": False,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
        errors.append(f"{name} did not emit valid JSON")
    if result.returncode != 0 or payload.get("ok") is not True:
        details = payload.get("errors")
        if isinstance(details, list):
            errors.extend(f"{name}: {detail}" for detail in details)
        elif not errors:
            errors.append(f"{name} exited {result.returncode}")
    return payload, errors


def _run_command(root: Path, command: list[str], name: str) -> tuple[dict, list[str]]:
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    step = {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "output": result.stdout[-4000:],
    }
    return step, [] if result.returncode == 0 else [
        f"{name} exited {result.returncode}"
    ]


def _read_changed_text_for_secret_scan(
    root: Path, relative: str, max_bytes: int
) -> tuple[str | None, str | None]:
    """Read one changed regular text file without following symlinks."""
    path = root / relative
    if not os.path.lexists(path):
        return None, None

    try:
        metadata = path.lstat()
    except OSError:
        return None, f"secret scan cannot inspect changed path: {relative}"
    if stat.S_ISLNK(metadata.st_mode):
        return None, f"secret scan refuses changed symlink: {relative}"
    if not stat.S_ISREG(metadata.st_mode):
        return None, f"secret scan refuses non-regular changed path: {relative}"
    if metadata.st_size > max_bytes:
        return (
            None,
            f"secret scan refuses changed file over {max_bytes} bytes: {relative}",
        )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None, f"secret scan cannot open changed file: {relative}"
    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            return None, f"secret scan refuses non-regular changed path: {relative}"
        if (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        ) != (metadata.st_dev, metadata.st_ino):
            return None, f"changed path changed during secret scan: {relative}"
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(max_bytes + 1)
    except OSError:
        return None, f"secret scan cannot read changed file: {relative}"
    finally:
        os.close(descriptor)

    if len(content) > max_bytes:
        return (
            None,
            f"secret scan refuses changed file over {max_bytes} bytes: {relative}",
        )
    if b"\0" in content:
        return None, None
    try:
        return content.decode("utf-8", errors="strict"), None
    except UnicodeDecodeError:
        return None, None


def validate_secret_patterns(
    root: Path, paths: Iterable[str], max_bytes: int
) -> list[str]:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    errors: list[str] = []
    for relative in sorted(set(paths)):
        text, read_error = _read_changed_text_for_secret_scan(root, relative, max_bytes)
        if read_error is not None:
            errors.append(read_error)
            continue
        if text is None:
            continue
        for name, pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"secret pattern {name} in {relative}:{line}")
    return errors


def validate_changed_file_sizes(
    root: Path, paths: Iterable[str], max_bytes: int
) -> list[str]:
    errors: list[str] = []
    for relative in sorted(set(paths)):
        path = root / relative
        if not os.path.lexists(path):
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            errors.append(f"changed path is a symlink: {relative}")
        elif not stat.S_ISREG(metadata.st_mode):
            errors.append(f"changed path is not a regular file: {relative}")
        elif metadata.st_size > max_bytes:
            errors.append(
                f"changed file exceeds {max_bytes} bytes: {relative} "
                f"({metadata.st_size} bytes)"
            )
    return errors


def _extract_link_targets(path: Path, text: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    patterns = (
        MARKDOWN_REFERENCE_LINK,
        HTML_LINK,
        RST_EXPLICIT_LINK,
        RST_DIRECTIVE,
        RST_DOC_ROLE,
    )
    if path.suffix.lower() == ".md":
        for match in MARKDOWN_INLINE_LINK.finditer(text):
            target = match.group("angled") or match.group("plain")
            matches.append((text.count("\n", 0, match.start()) + 1, target))
    for pattern in patterns:
        for match in pattern.finditer(text):
            matches.append((text.count("\n", 0, match.start()) + 1, match.group(1)))

    if path.suffix.lower() == ".rst":
        lines = text.splitlines()
        in_toctree = False
        directive_indent = 0
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith(".. toctree::"):
                in_toctree = True
                directive_indent = len(line) - len(line.lstrip())
                continue
            if not in_toctree:
                continue
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= directive_indent:
                in_toctree = False
                continue
            if stripped.startswith(":"):
                continue
            target = (
                stripped.rsplit("<", 1)[-1].rstrip(">") if "<" in stripped else stripped
            )
            matches.append((index, target))
    return matches


def _is_local_target(target: str) -> bool:
    target = target.strip().strip("<>")
    if not target or target.startswith("#") or "{" in target:
        return False
    parsed = urlsplit(target)
    return not parsed.scheme and not parsed.netloc


def _link_exists(root: Path, source: Path, target: str) -> bool:
    parsed = urlsplit(target.strip().strip("<>"))
    raw_path = unquote(parsed.path)
    if not raw_path:
        return True
    candidate = (
        root / raw_path.lstrip("/")
        if raw_path.startswith("/")
        else source.parent / raw_path
    )
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return False
    variants = [candidate]
    if not candidate.suffix:
        variants.extend(
            [
                candidate.with_suffix(".md"),
                candidate.with_suffix(".rst"),
                candidate / "index.md",
                candidate / "index.rst",
            ]
        )
    return any(variant.exists() for variant in variants)


def validate_local_links(root: Path, paths: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for relative in sorted(set(paths)):
        path = root / relative
        if not path.is_file() or path.suffix.lower() not in {".md", ".rst"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"cannot read documentation link source {relative}: {exc}")
            continue
        for line, target in _extract_link_targets(path, text):
            if _is_local_target(target) and not _link_exists(root, path, target):
                errors.append(f"broken local link in {relative}:{line}: {target}")
    return errors


def _tracked_documents(root: Path, changed_paths: Iterable[str]) -> set[str]:
    result = _git(root, "ls-files", "-z", "--", "*.md", "*.rst")
    tracked = {
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    }
    return tracked | {
        path for path in changed_paths if Path(path).suffix.lower() in {".md", ".rst"}
    }


def _validate_cff_version(root: Path) -> list[str]:
    errors: list[str] = []
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    required_lines = {
        "cff-version: 1.2.0": "CFF schema version",
        f'version: "{version}"': "CFF/package version",
        'repository-code: "https://github.com/Baosight-xVue/BaoIAD"': "CFF repository",
        'value: "10.5281/zenodo.20067087"': "CFF concept DOI",
    }
    for line, label in required_lines.items():
        if line not in cff:
            errors.append(f"{label} invariant is missing")
    if "date-released:" in cff:
        errors.append("CFF date-released must remain unset before the release archive")
    return errors


def _build_artifacts(root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="baoiad-release-build-") as temp_text:
        temp = Path(temp_text)
        source = temp / "source"
        dist = temp / "dist"
        ignore = shutil.ignore_patterns(
            ".git",
            ".omx",
            ".venv",
            "*.egg-info",
            "__pycache__",
            "build",
            "data",
        )
        shutil.copytree(root, source, ignore=ignore)
        dist.mkdir()
        uv = shutil.which("uv")
        command = (
            [uv, "build", "--out-dir", str(dist)]
            if uv
            else [
                sys.executable,
                "-m",
                "build",
                "--outdir",
                str(dist),
            ]
        )
        result = subprocess.run(
            command,
            cwd=source,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        step: dict[str, Any] = {
            "ok": result.returncode == 0,
            "command": command,
            "returncode": result.returncode,
            "output": result.stdout[-4000:],
        }
        if result.returncode != 0:
            return step, [f"distribution build exited {result.returncode}"]

        wheels = list(dist.glob("*.whl"))
        sdists = list(dist.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            return step, ["distribution build must emit one wheel and one sdist"]
        with zipfile.ZipFile(wheels[0]) as archive:
            metadata_name = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            wheel_metadata = Parser().parsestr(archive.read(metadata_name).decode())
        with tarfile.open(sdists[0], "r:gz") as archive:
            pkg_info_name = next(
                name for name in archive.getnames() if name.endswith("/PKG-INFO")
            )
            extracted = archive.extractfile(pkg_info_name)
            assert extracted is not None
            sdist_metadata = Parser().parsestr(extracted.read().decode())
        with (root / "pyproject.toml").open("rb") as stream:
            expected_version = tomllib.load(stream)["project"]["version"]
        for label, metadata in (("wheel", wheel_metadata), ("sdist", sdist_metadata)):
            if metadata["Version"] != expected_version:
                errors.append(f"{label} version does not match {expected_version}")
            if metadata["License-Expression"] != "Apache-2.0":
                errors.append(f"{label} License-Expression is not Apache-2.0")
            if metadata.get_all("License-File") != ["LICENSE"]:
                errors.append(f"{label} does not include License-File: LICENSE")
        step["ok"] = not errors
        step["artifacts"] = [wheels[0].name, sdists[0].name]
        return step, errors


def validate_release_candidate(root: Path = ROOT, static_only: bool = False) -> dict:
    root = root.resolve()
    errors: list[str] = []
    steps: dict[str, Any] = {}
    public_report, public_errors = _run_json_command(
        root,
        [
            sys.executable,
            str(root / "tools" / "check_public_release.py"),
            "--repo-root",
            str(root),
            "--policy",
            str(root / "tools" / "public_release_policy.json"),
            "--json",
        ],
        "public-release policy",
    )
    steps["public_release"] = public_report
    errors.extend(public_errors)
    changed_paths = public_report.get("changed_paths", [])
    if not isinstance(changed_paths, list):
        changed_paths = []
        errors.append("public-release policy did not report changed_paths")
    policy = json.loads((root / "tools" / "public_release_policy.json").read_text())
    max_bytes = policy["max_added_file_bytes"]

    secret_errors = validate_secret_patterns(root, changed_paths, max_bytes)
    size_errors = validate_changed_file_sizes(root, changed_paths, max_bytes)
    link_errors = validate_local_links(root, _tracked_documents(root, changed_paths))
    cff_errors = _validate_cff_version(root)
    steps["secret_scan"] = {"ok": not secret_errors, "errors": secret_errors}
    steps["changed_file_sizes"] = {"ok": not size_errors, "errors": size_errors}
    steps["local_links"] = {"ok": not link_errors, "errors": link_errors}
    steps["cff_version"] = {"ok": not cff_errors, "errors": cff_errors}
    errors.extend(secret_errors + size_errors + link_errors + cff_errors)

    if not static_only:
        json_checks = {
            "version_compatibility": [
                sys.executable,
                str(root / "tools" / "check_version_compatibility.py"),
                "--repo-root",
                str(root),
                "--json",
            ],
        }
        for name, command in json_checks.items():
            step, step_errors = _run_json_command(root, command, name)
            steps[name] = step
            errors.extend(step_errors)

        commands = {
            "method_inventory": [
                sys.executable,
                str(root / "tools" / "check_method_inventory.py"),
            ],
            "release_compliance": [
                sys.executable,
                str(root / "tools" / "check_release_compliance.py"),
                "--release-gate",
            ],
            "metadata_invariants": [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_release/test_metadata_invariants.py",
            ],
        }
        for name, command in commands.items():
            step, step_errors = _run_command(root, command, name)
            steps[name] = step
            errors.extend(step_errors)

        build_step, build_errors = _build_artifacts(root)
        steps["build"] = build_step
        errors.extend(build_errors)
        with tempfile.TemporaryDirectory(prefix="baoiad-release-docs-") as docs_temp:
            for language, source in (("en", "docs/en"), ("zh_cn", "docs/zh_cn")):
                command = [
                    sys.executable,
                    "-m",
                    "sphinx",
                    "-E",
                    "-W",
                    "--keep-going",
                    "-b",
                    "html",
                    source,
                    str(Path(docs_temp) / language),
                ]
                step, step_errors = _run_command(root, command, f"docs_{language}")
                steps[f"docs_{language}"] = step
                errors.extend(step_errors)

    return {
        "ok": not errors,
        "profile": "static-only" if static_only else "full-release-candidate",
        "complete": not static_only,
        "errors": errors,
        "steps": steps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run diff, allowlist, secret, size, link, and CFF checks only.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = validate_release_candidate(args.repo_root, args.static_only)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["ok"]:
        profile = report["profile"]
        print(f"release candidate verification: PASS ({profile})")
    else:
        print("release candidate verification: FAIL", file=sys.stderr)
        for error in report["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
