"""Regression tests for public release metadata."""

from __future__ import annotations

import runpy
import re
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:  # pragma: no cover - build env fallback
        from setuptools._vendor import tomli as tomllib

import baoiad


ROOT = Path(__file__).resolve().parents[2]
VERSION = "1.1.0"
REPOSITORY = "https://github.com/Baosight-xVue/BaoIAD"
CONCEPT_DOI = "10.5281/zenodo.20067087"
OLD_VERSION_DOI = "10.5281/zenodo.20067088"


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def _citation_metadata() -> dict:
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def _bibtex_block(readme: str) -> str:
    match = re.search(r"```bibtex\n(?P<body>.*?)\n```", readme, flags=re.DOTALL)
    assert match is not None
    return match.group("body")


def test_version_is_identical_across_public_metadata():
    project = _project_metadata()
    citation = _citation_metadata()
    docs_en = runpy.run_path(str(ROOT / "docs" / "en" / "conf.py"))
    docs_zh = runpy.run_path(str(ROOT / "docs" / "zh_cn" / "conf.py"))

    assert project["version"] == VERSION
    assert baoiad.__version__ == VERSION
    assert docs_en["release"] == VERSION
    assert docs_zh["release"] == VERSION
    assert citation["version"] == VERSION
    assert "date-released" not in citation


def test_project_identity_uses_canonical_organization_urls():
    project = _project_metadata()

    assert [author["name"] for author in project["authors"]] == [
        "Chenjie Xu",
        "Yang Zhang",
        "Tianyun Hu",
        "Bing Hu",
    ]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["urls"] == {
        "Homepage": REPOSITORY,
        "Repository": REPOSITORY,
        "Documentation": "https://baoiad.readthedocs.io/en/latest/",
        "Issues": f"{REPOSITORY}/issues",
        "Changelog": f"{REPOSITORY}/blob/master/docs/en/notes/changelog.md",
    }
    assert not any(
        classifier.startswith("License ::") for classifier in project["classifiers"]
    )

    extras = project["optional-dependencies"]
    for unsupported in ("mamba", "mmpretrain", "faiss-gpu", "imgaug"):
        assert unsupported not in extras


def test_citation_uses_only_the_verified_concept_doi():
    citation = _citation_metadata()
    identifiers = citation["identifiers"]

    assert citation["repository-code"] == REPOSITORY
    assert identifiers == [
        {
            "type": "doi",
            "value": CONCEPT_DOI,
            "description": "Zenodo concept DOI representing all versions of BaoIAD.",
        }
    ]
    assert citation["authors"][0]["orcid"] == ("https://orcid.org/0009-0006-8268-7732")
    assert OLD_VERSION_DOI not in (ROOT / "CITATION.cff").read_text(encoding="utf-8")


def test_readmes_publish_release_citation_and_dataset_counts():
    readme_en = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_zh = (ROOT / "README_zh-CN.md").read_text(encoding="utf-8")

    for readme in (readme_en, readme_zh):
        assert "zenodo.org/badge" not in readme.lower()
        assert f"version      = {{{VERSION}}}" in readme
        assert "publisher    =" not in readme
        assert f"doi          = {{{CONCEPT_DOI}}}" in readme
        assert OLD_VERSION_DOI not in readme

    assert _bibtex_block(readme_en) == _bibtex_block(readme_zh)

    assert "version-specific DOI for v1.1.0 will be added" in readme_en
    assert "v1.1.0 的版本专属 DOI 将在" in readme_zh
    assert "| MVTec AD 2 | 8 |" in readme_en
    assert "| Kolektor | 1 (adapter) |" in readme_en
    assert "| VAD | 1 (adapter) |" in readme_en
    assert "| RealIAD | 30 |" in readme_en
    assert "| MVTec AD 2 | 8 |" in readme_zh
    assert "| Kolektor | 1（适配器） |" in readme_zh
    assert "| VAD | 1（适配器） |" in readme_zh
    assert "| RealIAD | 30 |" in readme_zh


def test_changelogs_keep_release_identity_pending_until_publication():
    changelog_en = (ROOT / "docs" / "en" / "notes" / "changelog.md").read_text(
        encoding="utf-8"
    )
    changelog_zh = (ROOT / "docs" / "zh_cn" / "notes" / "changelog.md").read_text(
        encoding="utf-8"
    )

    assert "## v1.1.0 — Unreleased" in changelog_en
    assert (
        "release date, source commit, and version-specific Zenodo DOI" in changelog_en
    )
    assert "## v1.1.0 — 未发布" in changelog_zh
    assert "发布日期、源代码提交和 Zenodo 版本专属 DOI" in changelog_zh
