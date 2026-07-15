from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_public_release.py"
SPEC = importlib.util.spec_from_file_location("check_public_release", SCRIPT_PATH)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class PublicReleasePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tempdir.name)
        self._git("init", "-q")
        self._git("config", "user.name", "BaoIAD release test")
        self._git("config", "user.email", "release-test@example.invalid")
        (self.repo / "base.txt").write_text("base\n", encoding="utf-8")
        (self.repo / "conflict.txt").write_text("base\n", encoding="utf-8")
        (self.repo / "large.bin").write_bytes(b"x" * 64)
        (self.repo / "README.md").write_text(
            "# BaoIAD\n",
            encoding="utf-8",
        )
        (self.repo / "README_zh-CN.md").write_text(
            "# BaoIAD\n", encoding="utf-8"
        )
        (self.repo / "CITATION.cff").write_text(
            "message: Cite Chenjie Xu when using BaoIAD.\n"
            "authors:\n  - given-names: Chenjie\n    family-names: Xu\n"
            "repository-code: https://github.com/Baosight-xVue/BaoIAD\n",
            encoding="utf-8",
        )
        docs_en = self.repo / "docs" / "en"
        docs_en.mkdir(parents=True)
        (docs_en / "get_started.md").write_text(
            "git clone https://github.com/Baosight-xVue/BaoIAD.git\n",
            encoding="utf-8",
        )
        docs_zh = self.repo / "docs" / "zh_cn"
        docs_zh.mkdir(parents=True)
        (docs_zh / "get_started.md").write_text(
            "git clone https://github.com/Baosight-xVue/BaoIAD.git\n",
            encoding="utf-8",
        )
        alignment = self.repo / "docs" / "alignment"
        alignment.mkdir(parents=True)
        (alignment / "README.md").write_text(
            "# Alignment records\n\nValidation depth varies by method.\n",
            encoding="utf-8",
        )
        (alignment / "status.json").write_text("{}\n", encoding="utf-8")
        self.allowlist = self.repo / "allowlist.txt"
        self.allowlist.write_text("# release diff\n", encoding="utf-8")
        self.policy_path = self.repo / "policy.json"
        self.policy = {
            "version": 1,
            "base_ref": "HEAD",
            "release_branch": "test-release",
            "allowlist_file": "allowlist.txt",
            "max_added_file_bytes": 16,
            "added_file_size_exceptions": [],
            "banned_tracked_prefixes": ["forbidden/"],
            "banned_tracked_paths": ["secret.txt"],
            "banned_tracked_globs": ["**/*.pid"],
            "banned_added_suffixes": [".pth"],
        }
        self._write_policy()
        self._git("add", ".")
        self._git("commit", "-qm", "base")
        self._git("switch", "-qc", "test-release")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _git(self, *args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=self.repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_policy(self) -> None:
        self.policy_path.write_text(
            json.dumps(self.policy, indent=2) + "\n", encoding="utf-8"
        )

    def _write_allowlist(self, *paths: str) -> None:
        content = "# exact paths\n" + "".join(f"{path}\n" for path in paths)
        self.allowlist.write_text(content, encoding="utf-8")

    def _validate(self) -> dict[str, object]:
        return CHECKER.validate_release(self.repo, self.policy_path)

    def test_exact_diff_passes(self) -> None:
        (self.repo / "allowed.txt").write_text("ok\n", encoding="utf-8")
        self._write_allowlist("allowlist.txt", "allowed.txt")
        self._git("add", "allowlist.txt")

        report = self._validate()

        self.assertTrue(report["ok"], report["errors"])

    def test_unexpected_and_stale_allowlist_paths_fail(self) -> None:
        (self.repo / "surprise.txt").write_text("no\n", encoding="utf-8")
        self._write_allowlist("allowlist.txt", "stale.txt")

        report = self._validate()

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn("unexpected diff paths: surprise.txt", errors)
        self.assertIn("allowlisted paths absent from diff: stale.txt", errors)

    def test_banned_prefix_and_glob_fail(self) -> None:
        forbidden = self.repo / "forbidden"
        forbidden.mkdir()
        (forbidden / "result.txt").write_text("no\n", encoding="utf-8")
        logs = self.repo / "logs"
        logs.mkdir()
        (logs / "worker.pid").write_text("123\n", encoding="utf-8")
        self._write_allowlist(
            "allowlist.txt", "forbidden/result.txt", "logs/worker.pid"
        )

        report = self._validate()

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn("banned tracked prefix forbidden/", errors)
        self.assertIn("banned tracked glob **/*.pid", errors)

    def test_rename_target_cannot_bypass_added_file_size_gate(self) -> None:
        self._git("mv", "large.bin", "moved.bin")
        self._write_allowlist("allowlist.txt", "large.bin", "moved.bin")

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertIn(
            "added file exceeds 16 bytes: moved.bin (index=64, worktree=64)",
            report["errors"],
        )

    def test_staged_blob_cannot_hide_behind_smaller_worktree_file(self) -> None:
        staged = self.repo / "staged.bin"
        staged.write_bytes(b"x" * 64)
        self._git("add", "staged.bin")
        staged.write_bytes(b"x")
        self._write_allowlist("allowlist.txt", "staged.bin")

        report = self._validate()

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn(
            "changed tracked path has staged/worktree divergence: staged.bin", errors
        )
        self.assertIn("staged.bin (index=64)", errors)

    def test_tracked_document_cannot_hide_staged_content_in_worktree(self) -> None:
        readme = self.repo / "README.md"
        readme.write_text(
            "# BaoIAD\n\nUnder review.\n\n"
            "https://github.com/ChenjieXu/BaoIAD\n",
            encoding="utf-8",
        )
        self._git("add", "README.md")
        readme.write_text(
            "# BaoIAD\n",
            encoding="utf-8",
        )
        self._write_allowlist("allowlist.txt", "README.md")

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertIn(
            "changed tracked path has staged/worktree divergence: README.md",
            report["errors"],
        )

    def test_staged_deletion_cannot_hide_behind_worktree_recreation(self) -> None:
        self._git("rm", "README.md")
        (self.repo / "README.md").write_text(
            "# BaoIAD\n",
            encoding="utf-8",
        )
        self._write_allowlist("allowlist.txt", "README.md")
        self._git("add", "allowlist.txt")

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertIn(
            "staged deletion has worktree recreation: README.md", report["errors"]
        )
        self.assertIn("missing required public documents: README.md", report["errors"])

    def test_broken_symlink_is_rejected(self) -> None:
        (self.repo / "bad.pth").symlink_to("missing-target")
        self._write_allowlist("allowlist.txt", "bad.pth")

        report = self._validate()

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn("banned added-file suffix: bad.pth", errors)
        self.assertIn("added symlink is not permitted: bad.pth", errors)

    def test_double_star_glob_matches_zero_directory_levels(self) -> None:
        configs = self.repo / "configs"
        configs.mkdir()
        path = configs / "curricad_probe.py"
        path.write_text("research\n", encoding="utf-8")
        self.policy["banned_tracked_globs"] = ["configs/**/curricad_*.py"]
        self._write_policy()
        self._write_allowlist(
            "allowlist.txt", "configs/curricad_probe.py", "policy.json"
        )

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertIn(
            "banned tracked glob configs/**/curricad_*.py",
            "\n".join(report["errors"]),
        )

    def test_copy_source_and_target_are_both_in_exact_diff(self) -> None:
        shutil.copyfile(self.repo / "large.bin", self.repo / "copied.bin")
        self._git("add", "copied.bin")
        self._write_allowlist("allowlist.txt", "large.bin", "copied.bin")

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertEqual(
            report["changed_paths"],
            ["allowlist.txt", "copied.bin", "large.bin"],
        )
        self.assertIn(
            "added file exceeds 16 bytes: copied.bin (index=64, worktree=64)",
            report["errors"],
        )

    def test_size_exception_requires_complete_approval_schema(self) -> None:
        (self.repo / "allowed.txt").write_bytes(b"x" * 64)
        self.policy["added_file_size_exceptions"] = [{"path": "allowed.txt"}]
        self._write_policy()
        self._write_allowlist("allowlist.txt", "allowed.txt", "policy.json")

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertIn("size exception #1 is missing", "\n".join(report["errors"]))

    def test_release_branch_must_match_policy(self) -> None:
        self.policy["release_branch"] = "other-release"
        self._write_policy()
        self._write_allowlist("allowlist.txt", "policy.json")

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertIn(
            "release branch mismatch: expected other-release, got test-release",
            report["errors"],
        )

    def test_readmes_reject_review_status_and_personal_repository_url(self) -> None:
        (self.repo / "README.md").write_text(
            "# BaoIAD\n\nUnder-Review at NeurIPS.\n\n"
            "https://github.com/ChenjieXu/BaoIAD\n",
            encoding="utf-8",
        )
        self._write_allowlist("allowlist.txt", "README.md")

        report = self._validate()

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn("public README contains review-status text: README.md", errors)
        self.assertIn(
            "public README contains a non-organization BaoIAD URL: README.md",
            errors,
        )
        self.assertIn(
            "public document contains internal marker chenjiexu: README.md", errors
        )

    def test_chinese_readme_rejects_review_status(self) -> None:
        (self.repo / "README_zh-CN.md").write_text(
            "# BaoIAD\n\n论文审稿中，项目评审中。\n",
            encoding="utf-8",
        )
        self._write_allowlist("allowlist.txt", "README_zh-CN.md")

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertIn(
            "public README contains review-status text: README_zh-CN.md",
            report["errors"],
        )

    def test_citation_and_get_started_require_organization_repository_url(self) -> None:
        (self.repo / "CITATION.cff").write_text(
            "authors:\n  - given-names: Chenjie\n    family-names: Xu\n",
            encoding="utf-8",
        )
        (self.repo / "docs" / "en" / "get_started.md").write_text(
            "Clone the public repository.\n", encoding="utf-8"
        )
        (self.repo / "docs" / "zh_cn" / "get_started.md").write_text(
            "克隆公开仓库。\n", encoding="utf-8"
        )
        self._write_allowlist(
            "allowlist.txt",
            "CITATION.cff",
            "docs/en/get_started.md",
            "docs/zh_cn/get_started.md",
        )

        report = self._validate()

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn(
            "public document does not use "
            "https://github.com/Baosight-xVue/BaoIAD: CITATION.cff",
            errors,
        )
        self.assertIn(
            "public document does not use "
            "https://github.com/Baosight-xVue/BaoIAD: docs/en/get_started.md",
            errors,
        )
        self.assertIn(
            "public document does not use "
            "https://github.com/Baosight-xVue/BaoIAD: docs/zh_cn/get_started.md",
            errors,
        )

    def test_public_documents_reject_internal_paths_users_and_proxy(self) -> None:
        alignment_readme = self.repo / "docs" / "alignment" / "README.md"
        alignment_readme.write_text(
            "/mnt/data /Users/person /home/person xuchenjie chenjiexu\n"
            "manuscript evidence workspace https://gh-proxy.com/upstream\n",
            encoding="utf-8",
        )
        self._write_allowlist("allowlist.txt", "docs/alignment/README.md")

        report = self._validate()

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        for marker in (
            "/mnt/",
            "/Users/",
            "/home/",
            "xuchenjie",
            "chenjiexu",
            "manuscript evidence workspace",
            "gh-proxy.com",
        ):
            self.assertIn(
                f"public document contains internal marker {marker}: "
                "docs/alignment/README.md",
                errors,
            )

    def test_author_display_name_is_not_treated_as_an_internal_username(self) -> None:
        (self.repo / "CITATION.cff").write_text(
            "authors:\n  - given-names: Chenjie\n    family-names: Xu\n"
            "repository-code: https://github.com/Baosight-xVue/BaoIAD\n",
            encoding="utf-8",
        )
        self._write_allowlist("allowlist.txt", "CITATION.cff")
        self._git("add", "CITATION.cff", "allowlist.txt")

        report = self._validate()

        self.assertTrue(report["ok"], report["errors"])

    def test_alignment_documents_reject_internal_evidence_markers(self) -> None:
        alignment_readme = self.repo / "docs" / "alignment" / "README.md"
        alignment_readme.write_text(
            ".refs/upstream runs/alignment runs/benchmark playbook agent handoff\n",
            encoding="utf-8",
        )
        self._write_allowlist("allowlist.txt", "docs/alignment/README.md")

        report = self._validate()

        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        for marker in (
            ".refs/",
            "runs/alignment",
            "runs/benchmark",
            "playbook",
            "agent handoff",
        ):
            self.assertIn(
                "alignment document contains internal evidence marker "
                f"{marker}: docs/alignment/README.md",
                errors,
            )

    def test_alignment_json_is_not_scanned_as_public_documentation(self) -> None:
        status = self.repo / "docs" / "alignment" / "status.json"
        status.write_text(
            '{"internal_note": ".refs/upstream and agent handoff"}\n',
            encoding="utf-8",
        )
        self._write_allowlist("allowlist.txt", "docs/alignment/status.json")
        self._git("add", "docs/alignment/status.json", "allowlist.txt")

        report = self._validate()

        self.assertTrue(report["ok"], report["errors"])

    def test_alignment_index_rejects_blanket_strict_alignment_claim(self) -> None:
        alignment_readme = self.repo / "docs" / "alignment" / "README.md"
        alignment_readme.write_text(
            "This directory records strict-alignment evidence for the 37 methods.\n",
            encoding="utf-8",
        )
        self._write_allowlist("allowlist.txt", "docs/alignment/README.md")

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertIn(
            "docs/alignment/README.md makes a blanket strict-alignment claim "
            "for the 37-method inventory",
            report["errors"],
        )

    def test_unresolved_index_stages_are_rejected_for_tracked_files(self) -> None:
        base_object = (
            subprocess.run(
                ["git", "rev-parse", "HEAD:conflict.txt"],
                cwd=self.repo,
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .strip()
        )

        def write_blob(content: bytes) -> str:
            return (
                subprocess.run(
                    ["git", "hash-object", "-w", "--stdin"],
                    cwd=self.repo,
                    check=True,
                    input=content,
                    stdout=subprocess.PIPE,
                )
                .stdout.decode()
                .strip()
            )

        ours_object = write_blob(b"ours\n")
        theirs_object = write_blob(b"theirs\n")
        self._git("update-index", "--force-remove", "conflict.txt")
        index_info = (
            f"100644 {base_object} 1\tconflict.txt\n"
            f"100644 {ours_object} 2\tconflict.txt\n"
            f"100644 {theirs_object} 3\tconflict.txt\n"
        ).encode()
        subprocess.run(
            ["git", "update-index", "--index-info"],
            cwd=self.repo,
            check=True,
            input=index_info,
        )
        self._write_allowlist("allowlist.txt", "conflict.txt")

        report = self._validate()

        self.assertFalse(report["ok"])
        self.assertIn(
            "tracked path has unresolved index stages: conflict.txt",
            report["errors"],
        )

    def test_allowlist_file_must_stay_inside_repository(self) -> None:
        self.policy["allowlist_file"] = "../outside.txt"
        self._write_policy()

        with self.assertRaisesRegex(
            CHECKER.ReleasePolicyError, "allowlist_file must be repository-relative"
        ):
            self._validate()


if __name__ == "__main__":
    unittest.main()
