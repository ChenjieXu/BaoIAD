"""Tests for the public-release compliance inventory."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_checker():
    path = ROOT / "tools" / "check_release_compliance.py"
    spec = importlib.util.spec_from_file_location("release_compliance_checker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_repository_compliance_inventory_is_internally_consistent():
    checker = _load_checker()

    assert checker.validate_all(ROOT) == []


def test_canonical_apache_license_is_required(tmp_path):
    checker = _load_checker()
    (tmp_path / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")

    errors = checker.validate_license(tmp_path)

    assert len(errors) == 1
    assert "unmodified Apache-2.0 text" in errors[0]


def test_method_status_must_match_exact_37_method_inventory():
    checker = _load_checker()
    document = _read_json("docs/alignment/method_status.json")
    document["methods"] = document["methods"][:-1]

    errors = checker.validate_method_status_document(
        document,
        checker._load_inventory(ROOT),
    )

    assert any("slug set mismatch" in error for error in errors)


def test_public_evidence_cannot_be_claimed_when_raw_artifacts_are_absent():
    checker = _load_checker()
    document = _read_json("docs/alignment/method_status.json")
    document["methods"][0]["validation"]["public_evidence"] = True

    errors = checker.validate_method_status_document(
        document,
        checker._load_inventory(ROOT),
    )

    assert any("public_evidence cannot be true" in error for error in errors)


def test_method_paper_url_must_match_the_verified_primary_source():
    checker = _load_checker()
    document = _read_json("docs/alignment/method_status.json")
    document["methods"][0]["paper_url"] = "https://arxiv.org/abs/0000.00000"

    errors = checker.validate_method_status_document(
        document,
        checker._load_inventory(ROOT),
    )

    assert any(
        "paper_url must match verified primary source" in error for error in errors
    )


def test_method_status_is_bound_to_config_class_source_and_runtime_evidence():
    checker = _load_checker()
    document = _read_json("docs/alignment/method_status.json")
    item = next(entry for entry in document["methods"] if entry["slug"] == "glass")
    item["registry_name"] = "BogusDetector"
    item["detector_module"] = "baoiad.models.detectors.does_not_exist"
    item["source"] = {
        "url": "https://example.com/bogus",
        "revision": "bogus",
        "traceability": "public_revision",
    }
    item["validation"]["runtime_state"] = "not_assessed"

    errors = checker.validate_method_status_document(
        document,
        checker._load_inventory(ROOT),
        ROOT,
    )

    assert any(
        "registry_name must match config model.type" in error for error in errors
    )
    assert any(
        "detector_module does not resolve to a file" in error for error in errors
    )
    assert any(
        "source metadata must match the audited freeze" in error for error in errors
    )
    assert any(
        "runtime_state must match the audited freeze" in error for error in errors
    )


def test_non_approved_method_license_review_must_block_release():
    checker = _load_checker()
    document = _read_json("docs/alignment/method_status.json")
    document["methods"][0]["license_review"]["release_blocking"] = False

    errors = checker.validate_method_status_document(
        document,
        checker._load_inventory(ROOT),
    )

    assert any(
        "non-approved license review must block release" in error for error in errors
    )


def test_approved_method_license_review_needs_evidence():
    checker = _load_checker()
    document = _read_json("docs/alignment/method_status.json")
    review = document["methods"][0]["license_review"]
    review["status"] = "approved"
    review["release_blocking"] = False

    errors = checker.validate_method_status_document(
        document,
        checker._load_inventory(ROOT),
    )

    assert any("approved license review needs evidence" in error for error in errors)


def test_unresolved_provenance_must_block_release():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    document["entries"][0]["release_blocking"] = False

    errors = checker.validate_provenance_document(document, ROOT)

    assert any("unresolved item must block release" in error for error in errors)


def test_provenance_requires_a_scope_for_every_covered_path():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    document["entries"][0]["ranges"] = []

    errors = checker.validate_provenance_document(document, ROOT)

    assert any("ranges must be non-empty strings" in error for error in errors)


def test_provenance_audit_projection_prevents_distinct_source_entry_loss():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    document["entries"] = [
        entry for entry in document["entries"] if entry["id"] != "TP-CODE-CFA-COORDCONV"
    ]

    errors = checker.validate_provenance_document(document, ROOT)

    assert any("audited manifest projection changed" in error for error in errors)


def test_provenance_rejects_fake_derived_kinds_and_ranges():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    item = next(
        entry for entry in document["entries"] if entry["id"] == "TP-CODE-CUTPASTE"
    )
    item["kind"] = "referenced_external_weight_code"
    item["ranges"] = [f"{item['paths'][0]}:1banana"]

    errors = checker.validate_provenance_document(document, ROOT)

    assert any("invalid provenance kind" in error for error in errors)
    assert any("invalid range or whole-file scope" in error for error in errors)


def test_provenance_cannot_be_approved_with_unresolved_license():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    item = next(
        entry
        for entry in document["entries"]
        if entry["license"]["status"] == "needs_external_review"
    )
    item["reviewer_signoff"]["status"] = "approved"

    errors = checker.validate_provenance_document(document, ROOT)

    assert any(
        "approved signoff requires confirmed license" in error for error in errors
    )


def test_approved_code_provenance_requires_frozen_source_ranges_and_notices():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    item = next(
        entry for entry in document["entries"] if entry["id"] == "TP-CODE-WINCLIP"
    )
    item["reviewer_signoff"].update(
        status="approved",
        evidence_reference="LEGAL-FAKE",
    )
    item["release_blocking"] = False

    errors = checker.validate_provenance_document(document, ROOT)

    assert any(
        "approved code/test provenance needs a pinned source revision" in error
        for error in errors
    )
    assert any(
        "approved provenance needs obligations evidence" in error for error in errors
    )
    assert any(
        "approved attribution needs repository license notice files" in error
        for error in errors
    )

    pending_range_item = next(
        entry for entry in document["entries"] if entry["id"] == "TP-CODE-ADACLIP"
    )
    pending_range_item["reviewer_signoff"].update(
        status="approved",
        evidence_reference="LEGAL-FAKE",
    )
    pending_range_item["release_blocking"] = False

    pending_range_errors = checker.validate_provenance_document(document, ROOT)

    assert any(
        "approved provenance cannot retain pending range placeholders" in error
        for error in pending_range_errors
    )


def test_incompatible_code_requires_fail_closed_disposition():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    item = next(
        entry
        for entry in document["entries"]
        if entry["license"]["status"] == "incompatible"
    )
    item["disposition"] = "keep_with_attribution"

    errors = checker.validate_provenance_document(document, ROOT)

    assert any("must be removed, replaced, or rewritten" in error for error in errors)


def test_agpl_spdx_cannot_be_reclassified_as_confirmed():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    item = next(
        entry
        for entry in document["entries"]
        if entry["license"]["spdx"].startswith("AGPL")
    )
    item["license"]["status"] = "confirmed"
    item["disposition"] = "keep_with_attribution"
    item["reviewer_signoff"].update(
        status="approved",
        evidence_reference="LEGAL-FAKE",
    )
    item["release_blocking"] = False

    errors = checker.validate_provenance_document(document, ROOT)

    assert any("AGPL SPDX must remain incompatible" in error for error in errors)


def test_rejected_provenance_review_cannot_be_made_non_blocking():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    item = next(
        entry
        for entry in document["entries"]
        if entry["license"]["status"] == "confirmed"
    )
    item["reviewer_signoff"]["status"] = "rejected"
    item["release_blocking"] = False

    errors = checker.validate_provenance_document(document, ROOT)

    assert any("non-approved signoff must block release" in error for error in errors)


def test_provenance_requires_complete_non_empty_audit_fields():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    item = document["entries"][0]
    item["kind"] = ""
    item["obligations"] = []
    item["source"]["name"] = None
    item["license"]["evidence"] = ""
    item["reviewer_signoff"]["role"] = ""

    errors = checker.validate_provenance_document(document, ROOT)

    assert any("kind must be non-empty" in error for error in errors)
    assert any("obligations must be non-empty strings" in error for error in errors)
    assert any("source.name must be non-empty" in error for error in errors)
    assert any("license.evidence must be non-empty" in error for error in errors)
    assert any("reviewer role must be non-empty" in error for error in errors)


def test_direct_source_marker_catches_optional_the_wording():
    checker = _load_checker()

    assert checker.DIRECT_SOURCE_MARKER.search("from official implementation")
    assert checker.DIRECT_SOURCE_MARKER.search("from the official implementation")
    assert checker.DIRECT_SOURCE_MARKER.search("matches the reference DRAEM code")
    assert checker.DIRECT_SOURCE_MARKER.search("matches the reference implementation")
    assert checker.DIRECT_SOURCE_MARKER.search(
        "matches the optimizer shipped in the official Dinomaly repository"
    )
    assert checker.DIRECT_SOURCE_MARKER.search(
        "Perlin noise matching anomalib's torch implementation"
    )
    assert checker.DIRECT_SOURCE_MARKER.search("Official RD++ train loop helpers")
    assert checker.DIRECT_SOURCE_MARKER.search(
        "adapted from upstream official implementation"
    )
    assert checker.DIRECT_SOURCE_MARKER.search("matching the upstream training loop")


def test_external_weight_entry_does_not_satisfy_derived_code_coverage():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    path = "baoiad/models/detectors/efficientad.py"
    document["entries"] = [
        entry
        for entry in document["entries"]
        if not (
            path in entry["paths"] and entry["kind"] != "referenced_external_weight"
        )
    ]

    errors = checker.validate_provenance_document(document, ROOT)

    assert any(
        f"copied/ported/vendored marker lacks coverage: {path}" in error
        for error in errors
    )


def test_canonical_derived_detectors_cannot_be_omitted_from_provenance():
    checker = _load_checker()
    canonical_paths = {
        "baoiad/models/detectors/anomalyclip_official.py",
        "baoiad/models/detectors/anomalydino.py",
        "baoiad/models/detectors/dfkde.py",
    }

    for path in canonical_paths:
        document = _read_json(".github/release/provenance.json")
        document["entries"] = [
            entry for entry in document["entries"] if path not in entry["paths"]
        ]

        errors = checker.validate_provenance_document(document, ROOT)

        assert any(
            "required derived paths are not covered" in error and path in error
            for error in errors
        ), path


def test_secondary_source_entries_cannot_be_collapsed_into_primary_sources():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    identifier = "TP-CODE-HSUXU-FOCAL"
    document["entries"] = [
        entry for entry in document["entries"] if entry["id"] != identifier
    ]

    errors = checker.validate_provenance_document(document, ROOT)

    assert any(
        "required secondary-source entries are missing" in error and identifier in error
        for error in errors
    )


def test_external_artifacts_are_bound_to_exact_urls_and_fail_closed_state():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    identifier = "TP-WEIGHT-VITAD-DINO-SMALL"
    item = next(entry for entry in document["entries"] if entry["id"] == identifier)
    item["source"]["url"] = "https://example.invalid/random.pth"
    item["release_blocking"] = False

    errors = checker.validate_provenance_document(document, ROOT)

    assert any(
        "external artifact binding changed" in error and identifier in error
        for error in errors
    )


def test_required_external_artifact_cannot_be_omitted():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    identifier = "TP-DATA-IMAGENETTE"
    document["entries"] = [
        entry for entry in document["entries"] if entry["id"] != identifier
    ]

    errors = checker.validate_provenance_document(document, ROOT)

    assert any(
        "required external artifact entry is missing" in error and identifier in error
        for error in errors
    )


def test_external_artifact_can_reach_a_legitimate_approved_terminal_state():
    checker = _load_checker()
    document = _read_json(".github/release/provenance.json")
    identifier = "TP-DATA-IMAGENETTE"
    item = next(entry for entry in document["entries"] if entry["id"] == identifier)
    item["license"] = {
        "spdx": "LicenseRef-Imagenette-Approved-Terms",
        "status": "confirmed",
        "evidence": "LEGAL-IMAGENETTE-TERMS-2026",
    }
    item["reviewer_signoff"].update(
        status="approved",
        evidence_reference="APP-THIRD-PARTY-IMAGENETTE-2026",
    )
    item["obligations_evidence"] = ["SBOM-IMAGENETTE-REFERENCE-2026"]
    item["release_blocking"] = False

    errors = checker.validate_provenance_document(document, ROOT)

    forbidden = (
        "approved code/test provenance",
        "approved bundled code/test",
        "approved retained code/test",
        "external artifact binding changed",
        "resolved item cannot remain release-blocking",
    )
    assert not any(term in error for error in errors for term in forbidden), errors


def test_asset_manifest_is_bound_to_exact_file_hash():
    checker = _load_checker()
    document = _read_json("resources/asset_approvals.json")
    document["assets"][0]["sha256"] = "0" * 64

    errors = checker.validate_asset_approvals_document(document, ROOT)

    assert any("manifest sha256 does not match file" in error for error in errors)


def test_asset_manifest_is_bound_to_audited_media_and_origin_metadata():
    checker = _load_checker()
    document = _read_json("resources/asset_approvals.json")
    asset = document["assets"][0]
    asset["dimensions"] = "1x1"
    asset["origin"]["kind"] = "company_owned"
    asset["origin"]["creator_or_owner"] = "Unverified owner claim"

    errors = checker.validate_asset_approvals_document(document, ROOT)

    assert any("metadata changed from the audited freeze" in error for error in errors)
    assert any(
        "owner claim requires approved rights evidence" in error for error in errors
    )


def test_asset_cannot_resolve_without_rights_approval_evidence_and_scopes():
    checker = _load_checker()
    document = _read_json("resources/asset_approvals.json")
    asset = document["assets"][0]
    asset["rights"]["status"] = "approved"
    for approval in asset["approvals"].values():
        approval["status"] = "approved"

    errors = checker.validate_asset_approvals_document(document, ROOT)

    assert any(
        "approved rights need license basis and evidence" in error for error in errors
    )
    assert any(
        "approved technical approval needs evidence" in error for error in errors
    )
    assert any(
        "resolved asset scopes must be explicit booleans" in error for error in errors
    )


def test_asset_manifest_rejects_duplicate_paths_and_invalid_scope_values():
    checker = _load_checker()
    document = _read_json("resources/asset_approvals.json")
    document["assets"].append(copy.deepcopy(document["assets"][0]))
    document["assets"][0]["scopes"]["waic_event"] = "maybe"

    errors = checker.validate_asset_approvals_document(document, ROOT)

    assert any("duplicate path" in error for error in errors)
    assert any("scope values must be boolean or null" in error for error in errors)


def test_retained_asset_requires_github_repository_scope():
    checker = _load_checker()
    document = _read_json("resources/asset_approvals.json")
    asset = document["assets"][0]
    asset["rights"].update(
        status="approved",
        license_or_basis="company-owned artwork",
        evidence_reference="LEGAL-FAKE",
    )
    for approval in asset["approvals"].values():
        approval.update(status="approved", evidence_reference="APPROVAL-FAKE")
    asset["scopes"] = {scope: False for scope in asset["scopes"]}
    asset["scopes"]["waic_event"] = True
    asset["disposition"] = "approved_for_scopes"
    asset["release_blocking"] = False

    errors = checker.validate_asset_approvals_document(document, ROOT)

    assert any(
        "retained file requires github_repository scope" in error for error in errors
    )


def test_alignment_exception_cannot_be_downgraded_without_resolution_evidence():
    checker = _load_checker()
    document = _read_json("docs/alignment/exceptions.json")
    item = next(entry for entry in document["exceptions"] if entry["status"] == "open")
    item["status"] = "accepted_public_limitation"
    item["release_blocking"] = False
    item["resolution_evidence"] = "unverified text"

    errors = checker.validate_alignment_exceptions_document(document, ROOT)

    assert any("cannot be an accepted public limitation" in error for error in errors)


def test_machine_gated_alignment_items_cannot_close_on_free_form_text():
    checker = _load_checker()
    document = _read_json("docs/alignment/exceptions.json")
    item = next(
        entry for entry in document["exceptions"] if entry["id"] == "ALIGN-CLEAN-CLONE"
    )
    item["status"] = "resolved"
    item["release_blocking"] = False
    item["resolution_evidence"] = "unverified text"

    errors = checker.validate_alignment_exceptions_document(document, ROOT)

    assert (
        sum("requires a goal-specific machine gate" in error for error in errors) == 1
    )


def test_absent_evidence_exception_uses_the_alignment_marker_scan():
    checker = _load_checker()
    document = _read_json("docs/alignment/exceptions.json")
    item = next(
        entry
        for entry in document["exceptions"]
        if entry["id"] == "ALIGN-ABSENT-EVIDENCE"
    )
    item["status"] = "resolved"
    item["release_blocking"] = False
    item["resolution_evidence"] = (
        "G003 public documentation gate and alignment marker scan"
    )

    errors = checker.validate_alignment_exceptions_document(document, ROOT)

    assert not any("undistributed artifact markers remain" in error for error in errors)


def test_broken_link_exception_cannot_resolve_while_links_still_break(monkeypatch):
    checker = _load_checker()
    monkeypatch.setattr(
        checker,
        "_broken_alignment_links",
        lambda _root: ["docs/alignment/glass.md -> missing.md"],
    )
    document = _read_json("docs/alignment/exceptions.json")
    item = next(
        entry for entry in document["exceptions"] if entry["id"] == "ALIGN-BROKEN-LINKS"
    )
    item["status"] = "resolved"
    item["release_blocking"] = False
    item["resolution_evidence"] = "unverified text"

    errors = checker.validate_alignment_exceptions_document(document, ROOT)

    assert any("broken relative links remain" in error for error in errors)


def test_paper_link_exception_must_match_the_readme_scan(monkeypatch):
    checker = _load_checker()
    monkeypatch.setattr(
        checker,
        "_method_readme_paper_link_mismatches",
        lambda _root, _inventory: {"glass"},
    )
    document = _read_json("docs/alignment/exceptions.json")
    item = next(
        entry for entry in document["exceptions"] if entry["id"] == "ALIGN-PAPER-LINKS"
    )
    item["status"] = "open"
    item["release_blocking"] = True
    item["resolution_evidence"] = None

    errors = checker.validate_alignment_exceptions_document(document, ROOT)

    assert any("methods must match the README scan" in error for error in errors)


def test_alignment_method_sets_are_bound_to_method_status_states():
    checker = _load_checker()
    document = _read_json("docs/alignment/exceptions.json")
    clean_clone = next(
        entry for entry in document["exceptions"] if entry["id"] == "ALIGN-CLEAN-CLONE"
    )
    partial = next(
        entry
        for entry in document["exceptions"]
        if entry["id"] == "ALIGN-PARTIAL-VALIDATION"
    )
    clean_clone["methods"].remove("dinomaly")
    partial["methods"].remove("adaclip")

    errors = checker.validate_alignment_exceptions_document(document, ROOT)

    assert any("non-clean-clone runtime states" in error for error in errors)
    assert any("partial validation state set" in error for error in errors)


def test_pending_external_approval_must_fail_closed():
    checker = _load_checker()
    document = copy.deepcopy(_read_json("docs/release/external_approvals.json"))
    document["approvals"][0]["release_blocking"] = False

    errors = checker.validate_external_approvals_document(document)

    assert any("pending item must block release" in error for error in errors)


def test_rejected_external_approval_cannot_be_made_non_blocking():
    checker = _load_checker()
    document = copy.deepcopy(_read_json("docs/release/external_approvals.json"))
    document["approvals"][0]["status"] = "rejected"
    document["approvals"][0]["release_blocking"] = False

    errors = checker.validate_external_approvals_document(document)

    assert any("non-approved item must block release" in error for error in errors)


def test_approved_external_approval_requires_string_evidence_and_nonblocking_state():
    checker = _load_checker()
    document = copy.deepcopy(_read_json("docs/release/external_approvals.json"))
    approval = document["approvals"][0]
    approval["status"] = "approved"
    approval["evidence_reference"] = True

    errors = checker.validate_external_approvals_document(document)

    assert any(
        "approved item needs a string evidence reference" in error for error in errors
    )
    assert any(
        "approved item cannot remain release-blocking" in error for error in errors
    )


def test_terminal_approval_placeholders_are_rejected_across_manifests():
    checker = _load_checker()

    methods = _read_json("docs/alignment/method_status.json")
    method_review = methods["methods"][0]["license_review"]
    method_review.update(
        status="approved", evidence_reference="pending", release_blocking=False
    )
    method_errors = checker.validate_method_status_document(
        methods, checker._load_inventory(ROOT), ROOT
    )

    approvals = _read_json("docs/release/external_approvals.json")
    approvals["approvals"][0].update(
        status="approved", evidence_reference="pending", release_blocking=False
    )
    approval_errors = checker.validate_external_approvals_document(approvals)

    assets = _read_json("resources/asset_approvals.json")
    asset = assets["assets"][0]
    asset["rights"].update(
        status="approved",
        license_or_basis="pending",
        evidence_reference="pending",
    )
    for approval in asset["approvals"].values():
        approval.update(status="approved", evidence_reference="pending")
    asset_errors = checker.validate_asset_approvals_document(assets, ROOT)

    assert any("placeholders are not accepted" in error for error in method_errors)
    assert any("string evidence reference" in error for error in approval_errors)
    assert any(
        "approved rights need license basis and evidence" in error
        for error in asset_errors
    )


def test_external_approvals_reject_duplicate_ids():
    checker = _load_checker()
    document = copy.deepcopy(_read_json("docs/release/external_approvals.json"))
    document["approvals"].append(copy.deepcopy(document["approvals"][0]))

    errors = checker.validate_external_approvals_document(document)

    assert any("duplicate id" in error for error in errors)


def test_release_gate_accounts_for_pending_method_license_reviews():
    checker = _load_checker()

    blockers = checker.open_release_blockers(ROOT)

    assert any(
        blocker.startswith("method glass license review:") for blocker in blockers
    )


def test_release_gate_stays_closed_while_approvals_are_pending(capsys):
    checker = _load_checker()

    assert checker.main(["--release-gate"]) == 1

    output = capsys.readouterr().out
    assert "FAIL public release gate" in output
    assert "external approval APP-COMMUNITY-CONDUCT: pending" in output


def test_default_cli_accepts_truthfully_recorded_blockers(capsys):
    checker = _load_checker()

    assert checker.main([]) == 0

    output = capsys.readouterr().out
    assert "PASS release compliance inventory validation" in output
    assert "open release blockers: 113" in output
    assert "method records: 37" in output
