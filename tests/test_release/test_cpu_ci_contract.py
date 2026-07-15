"""Fail-closed invariants for the public, CPU-only GitHub Actions lane."""

from __future__ import annotations

from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
CONSTRAINTS_PATH = ROOT / ".github" / "constraints" / "ci-cpu.txt"
CHECKOUT_ACTION = "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5"
SETUP_PYTHON_ACTION = "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065"
P0_CONTEXTS = {
    "lint",
    "release-policy",
    "core-offline (3.10)",
    "core-offline (3.12)",
    "docs-en",
    "docs-zh",
}


def _workflow() -> dict:
    # BaseLoader keeps the YAML 1.1 spelling ``on`` as a string and is enough
    # for structural contract checks; GitHub still parses the actual workflow.
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _run_script(job: dict) -> str:
    return "\n".join(
        step.get("run", "") for step in job["steps"] if isinstance(step, dict)
    )


def test_cpu_workflow_has_minimal_read_only_trigger_surface() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = _workflow()

    assert set(workflow["on"]) == {"pull_request", "push"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert "pull_request_target" not in text
    assert "secrets." not in text
    assert "write" not in text.lower()


def test_core_and_package_jobs_gate_python_310_and_312_fail_fast() -> None:
    jobs = _workflow()["jobs"]
    for job_name in ("core", "package"):
        strategy = jobs[job_name]["strategy"]
        assert strategy["fail-fast"] == "true"
        assert strategy["matrix"]["python-version"] == ["3.10", "3.12"]
        assert jobs[job_name]["runs-on"] == "ubuntu-latest"

    core_script = _run_script(jobs["core"])
    for command_fragment in (
        "tools/check_install.py --offline --json",
        "tools/check_method_inventory.py",
        "tools/check_version_compatibility.py --json",
        "tests/test_checkpoint_policy.py",
        "tests/test_datasets/test_dataset_taxonomy.py",
        "tests/test_release",
        "not optional and not network and not gpu and not slow",
    ):
        assert command_fragment in core_script


def test_p0_context_names_match_release_process_and_ci_contract() -> None:
    jobs = _workflow()["jobs"]
    assert jobs["lint"]["name"] == "lint"
    assert jobs["release-policy"]["name"] == "release-policy"
    assert jobs["core"]["name"] == "core-offline (${{ matrix.python-version }})"
    assert {
        entry["check-name"] for entry in jobs["docs"]["strategy"]["matrix"]["include"]
    } == {"docs-en", "docs-zh"}

    for relative in (
        "docs/en/notes/release_process.md",
        "docs/zh_cn/notes/release_process.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert all(f"`{context}`" in text for context in P0_CONTEXTS)


def test_release_policy_job_runs_static_fail_closed_checks() -> None:
    script = _run_script(_workflow()["jobs"]["release-policy"])
    for command_fragment in (
        "git update-ref refs/remotes/upstream/master",
        "tools/check_public_release.py",
        "tools/check_release_candidate.py --static-only",
        "tools/check_release_compliance.py",
        "tools/check_method_inventory.py",
    ):
        assert command_fragment in script


def test_public_optional_extras_are_isolated_in_the_matrix() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        optional = set(tomllib.load(stream)["project"]["optional-dependencies"])
    expected = optional - {"dev"}

    matrix = _workflow()["jobs"]["optional-extras"]["strategy"]["matrix"]["include"]
    assert {entry["extra"] for entry in matrix} == expected
    assert all(entry["modules"].strip() for entry in matrix)


def test_docs_are_bilingual_and_warning_strict() -> None:
    docs = _workflow()["jobs"]["docs"]
    sources = {entry["source"] for entry in docs["strategy"]["matrix"]["include"]}
    assert sources == {"docs/en", "docs/zh_cn"}
    assert {entry["check-name"] for entry in docs["strategy"]["matrix"]["include"]} == {
        "docs-en",
        "docs-zh",
    }
    script = _run_script(docs)
    assert "-E -W --keep-going -b html" in script


def test_cpu_framework_constraints_are_exact_and_platform_neutral() -> None:
    constraints = {
        line.strip()
        for line in CONSTRAINTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert constraints == {
        "torch==2.7.1",
        "torchvision==0.22.1",
        "mmengine==0.10.7",
        "mmcv-lite==2.2.0",
    }
    assert not any(";" in requirement for requirement in constraints)
    assert "download.pytorch.org/whl/cpu" in WORKFLOW_PATH.read_text(encoding="utf-8")


def test_actions_are_commit_sha_pinned() -> None:
    actions = [
        step["uses"]
        for job in _workflow()["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    ]
    assert actions
    assert set(actions) == {CHECKOUT_ACTION, SETUP_PYTHON_ACTION}
    checkout_steps = [
        step
        for job in _workflow()["jobs"].values()
        for step in job["steps"]
        if step.get("uses") == CHECKOUT_ACTION
    ]
    assert checkout_steps
    assert all(
        step.get("with", {}).get("persist-credentials") == "false"
        for step in checkout_steps
    )


def test_pytest_markers_define_cpu_and_external_boundaries() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        pytest_config = tomllib.load(stream)["tool"]["pytest"]["ini_options"]
    markers = {entry.split(":", 1)[0] for entry in pytest_config["markers"]}
    assert markers == {"offline", "network", "gpu", "slow", "optional"}
    assert pytest_config["addopts"] == "--strict-markers"
