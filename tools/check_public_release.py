#!/usr/bin/env python3
"""Validate the public-release diff against a fail-closed repository policy.

The comparison covers the base ref through the index and working tree, plus
untracked files. On a clean release commit this is equivalent to base-to-HEAD;
before commit it also prevents staged or local changes from escaping review.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


class ReleasePolicyError(ValueError):
    """Raised when the release policy itself is invalid."""


def _git(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReleasePolicyError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _nul_fields(data: bytes) -> list[str]:
    return [
        field.decode("utf-8", errors="surrogateescape")
        for field in data.split(b"\0")
        if field
    ]


def _diff_paths(repo_root: Path, base_ref: str) -> set[str]:
    fields = _nul_fields(
        _git(
            repo_root,
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--find-copies",
            "--find-copies-harder",
            base_ref,
            "--",
        )
    )
    paths: set[str] = set()
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        path_count = 2 if status[:1] in {"R", "C"} else 1
        if index + path_count > len(fields):
            raise ReleasePolicyError(
                f"malformed git diff --name-status output near {status!r}"
            )
        paths.update(fields[index : index + path_count])
        index += path_count

    paths.update(
        _nul_fields(_git(repo_root, "ls-files", "--others", "--exclude-standard", "-z"))
    )
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleasePolicyError(f"cannot read policy {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReleasePolicyError("release policy must be a JSON object")
    return data


def _string_list(policy: dict[str, Any], key: str) -> list[str]:
    value = policy.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ReleasePolicyError(f"{key} must be a list of non-empty strings")
    return value


def _load_allowlist(repo_root: Path, relative_path: str) -> set[str]:
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ReleasePolicyError(
            f"allowlist_file must be repository-relative: {relative_path}"
        )
    path = repo_root / relative_path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReleasePolicyError(f"cannot read allowlist {path}: {exc}") from exc

    entries: list[str] = []
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("/") or ".." in Path(entry).parts:
            raise ReleasePolicyError(
                f"allowlist path must be repository-relative: {entry}"
            )
        if any(character in entry for character in "*?["):
            raise ReleasePolicyError(f"allowlist glob is not permitted: {entry}")
        entries.append(entry)

    if len(entries) != len(set(entries)):
        raise ReleasePolicyError("allowlist contains duplicate paths")
    return set(entries)


def _matches_glob(path: str, pattern: str) -> bool:
    variants = {pattern}
    pending = [pattern]
    while pending:
        candidate = pending.pop()
        marker = candidate.find("**/")
        if marker < 0:
            continue
        zero_directory_variant = candidate[:marker] + candidate[marker + 3 :]
        if zero_directory_variant not in variants:
            variants.add(zero_directory_variant)
            pending.append(zero_directory_variant)
    return any(fnmatch.fnmatchcase(path, variant) for variant in variants)


def _index_entries(repo_root: Path) -> dict[str, list[tuple[str, str, int]]]:
    entries: dict[str, list[tuple[str, str, int]]] = {}
    for field in _nul_fields(_git(repo_root, "ls-files", "--stage", "-z")):
        try:
            metadata, path = field.split("\t", 1)
            mode, object_id, stage_text = metadata.split()
            stage_number = int(stage_text)
        except (ValueError, TypeError) as exc:
            raise ReleasePolicyError(
                f"malformed git ls-files --stage output: {field!r}"
            ) from exc
        entries.setdefault(path, []).append((mode, object_id, stage_number))
    return entries


def _validate_exceptions(
    raw_exceptions: Any, default_max_bytes: int
) -> tuple[dict[str, int], list[str]]:
    errors: list[str] = []
    if not isinstance(raw_exceptions, list):
        raise ReleasePolicyError("added_file_size_exceptions must be a list")

    exceptions: dict[str, int] = {}
    required = {"path", "max_bytes", "reason", "approved_by"}
    for index, item in enumerate(raw_exceptions):
        if not isinstance(item, dict):
            errors.append(f"size exception #{index + 1} must be an object")
            continue
        missing = sorted(required - item.keys())
        if missing:
            errors.append(
                f"size exception #{index + 1} is missing: {', '.join(missing)}"
            )
            continue
        path = item["path"]
        max_bytes = item["max_bytes"]
        reason = item["reason"]
        approved_by = item["approved_by"]
        if not isinstance(path, str) or not path or any(ch in path for ch in "*?["):
            errors.append(f"size exception #{index + 1} has an invalid literal path")
            continue
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes <= default_max_bytes
        ):
            errors.append(
                f"size exception {path} max_bytes must exceed {default_max_bytes}"
            )
            continue
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"size exception {path} requires a reason")
            continue
        if not isinstance(approved_by, str) or not approved_by.strip():
            errors.append(f"size exception {path} requires approved_by")
            continue
        if path in exceptions:
            errors.append(f"duplicate size exception: {path}")
            continue
        exceptions[path] = max_bytes
    return exceptions, errors


def validate_release(repo_root: Path, policy_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    policy_path = policy_path.resolve()
    policy = _read_json(policy_path)
    if policy.get("version") != 1:
        raise ReleasePolicyError("unsupported or missing release policy version")

    base_ref = policy.get("base_ref")
    release_branch = policy.get("release_branch")
    allowlist_file = policy.get("allowlist_file")
    max_added_bytes = policy.get("max_added_file_bytes")
    if not isinstance(base_ref, str) or not base_ref:
        raise ReleasePolicyError("base_ref must be a non-empty string")
    if not isinstance(release_branch, str) or not release_branch:
        raise ReleasePolicyError("release_branch must be a non-empty string")
    if not isinstance(allowlist_file, str) or not allowlist_file:
        raise ReleasePolicyError("allowlist_file must be a non-empty string")
    if (
        not isinstance(max_added_bytes, int)
        or isinstance(max_added_bytes, bool)
        or max_added_bytes <= 0
    ):
        raise ReleasePolicyError("max_added_file_bytes must be a positive integer")

    banned_prefixes = _string_list(policy, "banned_tracked_prefixes")
    banned_paths = set(_string_list(policy, "banned_tracked_paths"))
    banned_globs = _string_list(policy, "banned_tracked_globs")
    banned_suffixes = _string_list(policy, "banned_added_suffixes")
    size_exceptions, errors = _validate_exceptions(
        policy.get("added_file_size_exceptions"), max_added_bytes
    )

    _git(repo_root, "rev-parse", "--verify", f"{base_ref}^{{commit}}")
    current_branch = _git(repo_root, "branch", "--show-current").decode().strip()
    if current_branch != release_branch:
        errors.append(
            f"release branch mismatch: expected {release_branch}, got "
            f"{current_branch or '<detached>'}"
        )
    changed_paths = _diff_paths(repo_root, base_ref)
    allowlist = _load_allowlist(repo_root, allowlist_file)

    unexpected = sorted(changed_paths - allowlist)
    stale = sorted(allowlist - changed_paths)
    if unexpected:
        errors.append("unexpected diff paths: " + ", ".join(unexpected))
    if stale:
        errors.append("allowlisted paths absent from diff: " + ", ".join(stale))

    base_paths = set(
        _nul_fields(_git(repo_root, "ls-tree", "-r", "-z", "--name-only", base_ref))
    )
    index_entries = _index_entries(repo_root)
    tracked_paths = set(index_entries)
    unresolved_index_paths: set[str] = set()
    for path, entries in sorted(index_entries.items()):
        if len(entries) != 1 or entries[0][2] != 0:
            errors.append(f"tracked path has unresolved index stages: {path}")
            unresolved_index_paths.add(path)
    unstaged_paths = set(
        _nul_fields(_git(repo_root, "diff", "--name-only", "-z", "--"))
    )
    present_changed_paths = {
        path
        for path in changed_paths
        if path in tracked_paths or os.path.lexists(repo_root / path)
    }
    candidate_paths = tracked_paths | present_changed_paths

    for path in sorted(candidate_paths):
        if path in banned_paths:
            errors.append(f"banned tracked path: {path}")
        for prefix in banned_prefixes:
            if path.startswith(prefix):
                errors.append(f"banned tracked prefix {prefix}: {path}")
        for pattern in banned_globs:
            if _matches_glob(path, pattern):
                errors.append(f"banned tracked glob {pattern}: {path}")

    added_paths = sorted(
        path for path in present_changed_paths if path not in base_paths
    )
    for path in added_paths:
        if any(path.endswith(suffix) for suffix in banned_suffixes):
            errors.append(f"banned added-file suffix: {path}")

        sizes: dict[str, int] = {}
        entries = index_entries.get(path, [])
        if entries and path not in unresolved_index_paths:
            mode, object_id, _ = entries[0]
            if mode not in {"100644", "100755"}:
                errors.append(f"added path has unsupported git mode {mode}: {path}")
            size_text = _git(repo_root, "cat-file", "-s", object_id).decode().strip()
            sizes["index"] = int(size_text)
        if entries:
            if path in unstaged_paths:
                errors.append(f"added path has staged/worktree divergence: {path}")

        worktree_path = repo_root / path
        if os.path.lexists(worktree_path):
            metadata = worktree_path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                errors.append(f"added symlink is not permitted: {path}")
            elif stat.S_ISREG(metadata.st_mode):
                sizes["worktree"] = metadata.st_size
            else:
                errors.append(f"added path is not a regular file: {path}")

        allowed_size = size_exceptions.get(path, max_added_bytes)
        oversized = {
            source: size for source, size in sizes.items() if size > allowed_size
        }
        if oversized:
            evidence = ", ".join(
                f"{source}={size}" for source, size in sorted(oversized.items())
            )
            errors.append(
                f"added file exceeds {allowed_size} bytes: {path} ({evidence})"
            )

    for path in sorted(size_exceptions):
        if path not in added_paths:
            errors.append(f"size exception does not target an added file: {path}")

    return {
        "ok": not errors,
        "base_ref": base_ref,
        "changed_paths": sorted(changed_paths),
        "added_paths": added_paths,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--policy",
        type=Path,
        default=default_root / "tools/public_release_policy.json",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = validate_release(args.repo_root, args.policy)
    except ReleasePolicyError as exc:
        report = {"ok": False, "errors": [str(exc)]}

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["ok"]:
        print(
            "public release policy: PASS "
            f"({len(report['changed_paths'])} changed, "
            f"{len(report['added_paths'])} added)"
        )
    else:
        print("public release policy: FAIL", file=sys.stderr)
        for error in report["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
