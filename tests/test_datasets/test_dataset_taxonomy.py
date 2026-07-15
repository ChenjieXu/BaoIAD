"""Release invariants for public dataset taxonomy claims."""

import ast
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).with_name("dataset_taxonomy.json")


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _all_categories(dataset: dict[str, Any]) -> tuple[str, ...]:
    source = ROOT / dataset["dataset_module"]
    tree = ast.parse(source.read_text(encoding="utf-8"))
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == dataset["dataset_class"]
    )
    assignment = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ALL_CATEGORIES"
            for target in node.targets
        )
    )
    categories = ast.literal_eval(assignment.value)
    return tuple(categories)


def _configured_dataset_types(config_path: str) -> set[str]:
    source = ROOT / config_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    configured_types: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "dict":
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "type"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                and keyword.value.value.endswith("Dataset")
            ):
                configured_types.add(keyword.value.value)
    return configured_types


def _markdown_row(document: str, display_name: str) -> list[str]:
    pattern = rf"^\|\s*{re.escape(display_name)}\s*\|(.+)$"
    match = re.search(pattern, document, flags=re.MULTILINE)
    assert match is not None, f"Missing taxonomy row for {display_name}"
    return [cell.strip() for cell in match.group(0).strip("|").split("|")]


def _leading_count(cell: str) -> int:
    match = re.match(r"^(\d+)", cell)
    assert match is not None, f"Expected a leading integer count in {cell!r}"
    return int(match.group(1))


def test_fixture_has_three_sourced_taxonomy_scopes() -> None:
    fixture = _load_fixture()

    assert fixture["schema_version"] == 1
    assert set(fixture["definitions"]) == {
        "object_categories",
        "defect_categories",
        "baoiad_entries",
    }
    assert len(fixture["datasets"]) == 10
    assert len({entry["key"] for entry in fixture["datasets"]}) == 10

    for dataset in fixture["datasets"]:
        for scope in ("object_categories", "defect_categories", "baoiad_entries"):
            assert scope in dataset
            source_path, _, symbol = dataset[scope]["source"].partition("::")
            assert symbol
            assert (ROOT / source_path).is_file()

        defect_count = dataset["defect_categories"]["count"]
        assert defect_count is None or defect_count >= 1
        if defect_count is None:
            assert dataset["defect_categories"]["representation"] in {
                "runtime-discovered",
                "metadata-discovered",
                "binary-adapter-label",
            }


def test_object_counts_match_all_categories_without_rewriting_constants() -> None:
    for dataset in _load_fixture()["datasets"]:
        categories = _all_categories(dataset)
        assert len(categories) == dataset["object_categories"]["count"]
        assert len(categories) == len(set(categories))


def test_each_public_dataset_has_one_base_config_dataset_pair() -> None:
    datasets = _load_fixture()["datasets"]

    assert sum(entry["baoiad_entries"]["count"] for entry in datasets) == 10
    for dataset in datasets:
        assert dataset["baoiad_entries"]["count"] == 1
        assert _configured_dataset_types(dataset["base_config"]) == {
            dataset["dataset_class"]
        }


def test_dataset_zoo_tables_match_fixture() -> None:
    fixture = _load_fixture()
    english = (ROOT / "docs/en/dataset_zoo.md").read_text(encoding="utf-8")
    chinese = (ROOT / "docs/zh_cn/dataset_zoo.md").read_text(encoding="utf-8")

    for dataset in fixture["datasets"]:
        expected = dataset["object_categories"]["count"]
        english_row = _markdown_row(english, dataset["display_name"])
        chinese_row = _markdown_row(chinese, dataset["display_name"])
        assert _leading_count(english_row[5]) == expected
        assert _leading_count(chinese_row[1]) == expected
        assert _leading_count(english_row[6]) == dataset["baoiad_entries"]["count"]
        assert _leading_count(chinese_row[2]) == dataset["baoiad_entries"]["count"]
        assert english_row[7] == "Not statically enumerated"
        assert chinese_row[3] == "未静态穷举"


def test_prepare_dataset_object_counts_match_fixture() -> None:
    document = (ROOT / "docs/en/user_guides/prepare_dataset.md").read_text(
        encoding="utf-8"
    )
    fixture = _load_fixture()["datasets"]

    for index, dataset in enumerate(fixture):
        heading = f"### {dataset['display_name']}"
        start = document.index(heading)
        if index + 1 < len(fixture):
            end = document.index(f"### {fixture[index + 1]['display_name']}", start)
        else:
            end = document.index("## Custom Datasets", start)
        section = document[start:end]
        match = re.search(
            r"^- \*\*BaoIAD object entries\*\*: (\d+)", section, re.MULTILINE
        )
        assert match is not None, f"Missing BaoIAD object count for {heading}"
        assert int(match.group(1)) == dataset["object_categories"]["count"]

    assert "- **Categories**:" not in document
