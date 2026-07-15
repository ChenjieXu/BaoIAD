#!/usr/bin/env python3
"""Validate the v1.0.0 to v1.1.0 compatibility contract."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "alignment" / "v1_0_0_compatibility.json"
EXPECTED_TAG = "v1.0.0"
EXPECTED_SHA = "697fc4304cc76876d397067e2706ed771f62e708"
EXPECTED_BASELINE_VERSION = "0.1.0"
EXPECTED_CURRENT_VERSION = "1.1.0"
EXPECTED_ROOTS = {"baoiad", "configs", "tools"}
EXPECTED_MIGRATIONS = {"checkpoint", "data_root", "regad", "vitad"}


def _dependency(project: dict, name: str) -> str | None:
    for requirement in project.get("dependencies", []):
        package = re.split(r"[<>=!~;\s\[]", requirement, maxsplit=1)[0]
        if package == name:
            return requirement
    return None


def _package_version(path: Path) -> str | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return None


def _cli_surface(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    positionals: set[str] = set()
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if first.value.startswith("-"):
            flags.add(first.value)
        else:
            positionals.add(first.value)
    return positionals, flags


def _validate_schema(document: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["manifest root must be an object"]
    required = {
        "schema_version",
        "artifact",
        "baseline",
        "current",
        "retained_surface",
        "migrations",
        "known_breaks",
        "release_rationale",
    }
    missing = required - document.keys()
    if missing:
        errors.append(f"manifest missing keys: {sorted(missing)}")
        return errors
    if document["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if not isinstance(document["artifact"], str) or not document["artifact"].strip():
        errors.append("artifact must be a non-empty string")

    baseline = document["baseline"]
    current = document["current"]
    retained = document["retained_surface"]
    for name, section in (
        ("baseline", baseline),
        ("current", current),
        ("retained_surface", retained),
    ):
        if not isinstance(section, dict):
            errors.append(f"{name} must be an object")
    if errors:
        return errors

    expected_baseline = {
        "git_tag": EXPECTED_TAG,
        "peeled_commit": EXPECTED_SHA,
        "package_version": EXPECTED_BASELINE_VERSION,
        "requires_python": ">=3.9",
        "core_dependency": "mmcv>=2.0",
    }
    for key, value in expected_baseline.items():
        if baseline.get(key) != value:
            errors.append(f"baseline.{key} must be {value!r}")
    expected_current = {
        "package_version": EXPECTED_CURRENT_VERSION,
        "requires_python": ">=3.10",
        "core_dependency": "mmcv-lite>=2.0",
    }
    for key, value in expected_current.items():
        if current.get(key) != value:
            errors.append(f"current.{key} must be {value!r}")
    for section_name, section in (("baseline", baseline), ("current", current)):
        extras = section.get("optional_extras")
        if not isinstance(extras, list) or not all(
            isinstance(item, str) for item in extras
        ):
            errors.append(f"{section_name}.optional_extras must be a string list")
        elif extras != sorted(set(extras)):
            errors.append(f"{section_name}.optional_extras must be sorted and unique")

    roots = retained.get("roots")
    if not isinstance(roots, list) or set(roots) != EXPECTED_ROOTS:
        errors.append(f"retained_surface.roots must be {sorted(EXPECTED_ROOTS)}")
    cli = retained.get("cli")
    if not isinstance(cli, list) or not cli:
        errors.append("retained_surface.cli must be a non-empty list")
    else:
        for index, entry in enumerate(cli):
            if not isinstance(entry, dict):
                errors.append(f"retained_surface.cli[{index}] must be an object")
                continue
            if set(entry) != {"path", "positionals", "flags"}:
                errors.append(
                    f"retained_surface.cli[{index}] must contain path, positionals, flags"
                )
            for key in ("positionals", "flags"):
                values = entry.get(key)
                if not isinstance(values, list) or not all(
                    isinstance(item, str) for item in values
                ):
                    errors.append(
                        f"retained_surface.cli[{index}].{key} must be a string list"
                    )
                elif values != sorted(set(values)):
                    errors.append(
                        f"retained_surface.cli[{index}].{key} must be sorted and unique"
                    )
    registry = retained.get("registry")
    if not isinstance(registry, dict) or registry.get("mode") != "lazy":
        errors.append("retained_surface.registry must declare lazy mode")

    migrations = document["migrations"]
    if not isinstance(migrations, list):
        errors.append("migrations must be a list")
    else:
        areas = {item.get("area") for item in migrations if isinstance(item, dict)}
        if areas != EXPECTED_MIGRATIONS:
            errors.append(f"migration areas must be {sorted(EXPECTED_MIGRATIONS)}")
        for index, item in enumerate(migrations):
            if not isinstance(item, dict):
                errors.append(f"migrations[{index}] must be an object")
                continue
            for key in ("area", "from", "to", "compatibility"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    errors.append(f"migrations[{index}].{key} must be non-empty")
            evidence = item.get("evidence")
            if not isinstance(evidence, dict) or set(evidence) != {"path", "contains"}:
                errors.append(f"migrations[{index}].evidence is invalid")

    for key in ("known_breaks", "release_rationale"):
        if not isinstance(document[key], list) or not document[key]:
            errors.append(f"{key} must be a non-empty list")
    return errors


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return None


def _validate_live_tag(document: dict, root: Path) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    inside = _run_git(root, "rev-parse", "--is-inside-work-tree")
    if inside is None or inside.returncode != 0:
        return errors, {"status": "skipped", "reason": "git repository unavailable"}
    tag = _run_git(
        root, "rev-parse", "--verify", "--quiet", f"refs/tags/{EXPECTED_TAG}"
    )
    if tag is None or tag.returncode != 0:
        return errors, {
            "status": "skipped",
            "reason": f"tag {EXPECTED_TAG} unavailable",
        }

    peeled = _run_git(root, "rev-parse", f"{EXPECTED_TAG}^{{}}")
    assert peeled is not None
    actual_sha = peeled.stdout.strip()
    if peeled.returncode != 0 or actual_sha != EXPECTED_SHA:
        errors.append(
            f"{EXPECTED_TAG} peeled SHA is {actual_sha!r}, expected {EXPECTED_SHA}"
        )

    roots = document["retained_surface"]["roots"]
    for retained_root in roots:
        tree = _run_git(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            f"{EXPECTED_TAG}^{{}}",
            "--",
            retained_root,
        )
        assert tree is not None
        if tree.returncode != 0:
            errors.append(f"could not inspect {retained_root} at {EXPECTED_TAG}")
            continue
        missing = [
            path for path in tree.stdout.splitlines() if not (root / path).is_file()
        ]
        if missing:
            errors.append(
                f"retained root {retained_root} deleted tagged paths: {missing}"
            )

    tagged_pyproject = _run_git(root, "show", f"{EXPECTED_TAG}:pyproject.toml")
    assert tagged_pyproject is not None
    if tagged_pyproject.returncode != 0:
        errors.append(f"could not read pyproject.toml from {EXPECTED_TAG}")
    else:
        tagged_project = tomllib.loads(tagged_pyproject.stdout)["project"]
        baseline = document["baseline"]
        checks = {
            "package_version": tagged_project.get("version"),
            "requires_python": tagged_project.get("requires-python"),
            "core_dependency": _dependency(tagged_project, "mmcv"),
            "optional_extras": sorted(tagged_project.get("optional-dependencies", {})),
        }
        for key, actual in checks.items():
            if baseline.get(key) != actual:
                errors.append(
                    f"baseline.{key} does not match {EXPECTED_TAG}: "
                    f"manifest={baseline.get(key)!r}, tag={actual!r}"
                )
    return errors, {"status": "verified", "peeled_commit": actual_sha}


def validate(document: object, root: Path = ROOT) -> tuple[list[str], dict[str, str]]:
    errors = _validate_schema(document)
    if errors or not isinstance(document, dict):
        return errors, {"status": "not-run", "reason": "schema validation failed"}

    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    current = document["current"]
    current_checks = {
        "package_version": project.get("version"),
        "requires_python": project.get("requires-python"),
        "core_dependency": _dependency(project, "mmcv-lite"),
        "optional_extras": sorted(project.get("optional-dependencies", {})),
    }
    for key, actual in current_checks.items():
        if current.get(key) != actual:
            errors.append(
                f"current.{key} does not match pyproject.toml: "
                f"manifest={current.get(key)!r}, source={actual!r}"
            )
    package_version = _package_version(root / "baoiad" / "__init__.py")
    if package_version != current["package_version"]:
        errors.append(
            "current.package_version does not match baoiad/__init__.py: "
            f"manifest={current['package_version']!r}, source={package_version!r}"
        )

    retained = document["retained_surface"]
    for retained_root in retained["roots"]:
        if not (root / retained_root).is_dir():
            errors.append(f"retained root is missing: {retained_root}")
    for entry in retained["cli"]:
        path = root / entry["path"]
        if not path.is_file():
            errors.append(f"retained CLI path is missing: {entry['path']}")
            continue
        positionals, flags = _cli_surface(path)
        missing_positionals = set(entry["positionals"]) - positionals
        missing_flags = set(entry["flags"]) - flags
        if missing_positionals:
            errors.append(
                f"{entry['path']} missing legacy positionals: {sorted(missing_positionals)}"
            )
        if missing_flags:
            errors.append(
                f"{entry['path']} missing legacy flags: {sorted(missing_flags)}"
            )

    registry = retained["registry"]
    registry_path = root / registry["registry_module"]
    registration_path = root / registry["explicit_registration_entrypoint"]
    if not registry_path.is_file() or "locations=" not in registry_path.read_text(
        encoding="utf-8"
    ):
        errors.append("lazy registry locations are not retained")
    if not registration_path.is_file() or "register_all_modules()" not in (
        registration_path.read_text(encoding="utf-8")
    ):
        errors.append("explicit registry entrypoint is not retained")

    for migration in document["migrations"]:
        evidence = migration["evidence"]
        path = root / evidence["path"]
        if not path.is_file():
            errors.append(f"migration evidence path is missing: {evidence['path']}")
        elif evidence["contains"] not in path.read_text(encoding="utf-8"):
            errors.append(
                f"migration evidence token missing from {evidence['path']}: "
                f"{evidence['contains']!r}"
            )

    live_errors, live_tag = _validate_live_tag(document, root)
    errors.extend(live_errors)
    return errors, live_tag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        document = json.loads(args.manifest.read_text(encoding="utf-8"))
        errors, live_tag = validate(document, args.repo_root.resolve())
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        errors = [f"could not validate compatibility manifest: {exc}"]
        live_tag = {"status": "not-run", "reason": "manifest load failed"}

    report = {"ok": not errors, "errors": errors, "live_tag": live_tag}
    if args.json:
        print(json.dumps(report, indent=2))
    elif errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
    else:
        print("PASS v1.0.0 to v1.1.0 compatibility validation")
        print(f"live tag: {live_tag['status']}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
