# BaoIAD v1.1.0 release notes (Unreleased)

## Draft state

This document is a draft for the target v1.1.0 release. BaoIAD v1.1.0 has not
been released. No release date, tag, GitHub Release URL, documentation
promotion, or version DOI is claimed here.

The final version remains subject to the release owner's recorded decision. If
compatibility review changes the approved version, the release owner must
update package, citation, documentation, and release metadata together before
publication.

## Highlights

- A canonical 37-method inventory now separates public method availability
  from alignment, runtime, and evidence status. The inventory is not a claim
  that all methods completed end-to-end execution.
- Public release checks bind method status, provenance, asset authorization,
  external approvals, exact diff paths, file-size limits, local links, and
  secret scanning to auditable repository records.
- CPU CI defines stable Python 3.10 and 3.12 gates, isolated optional extras,
  bilingual warning-strict documentation builds, and package smoke checks.
- Offline and optional-dependency boundaries are explicit. Data paths,
  checkpoint trust, network access, and missing external artifacts fail with
  actionable behavior instead of silently fabricating release evidence.
- Contributor, security, issue, pull-request, release, GPU-evidence, and
  hotfix governance are documented by role without publishing private contact
  details.

## Compatibility and migration

The audited [v1.0.0 compatibility contract](../alignment/v1_0_0_compatibility.json)
records the intentional boundaries for this target release:

- Python 3.10 or newer is required; the release gates cover Python 3.10 and
  3.12.
- The core dependency uses `mmcv-lite`; compiled MMCV operators are not a core
  package requirement.
- The legacy `faiss-gpu`, `imgaug`, `mamba`, and `mmpretrain` extras are no
  longer declared. Install only the public extras listed in `pyproject.toml`.
- Restricted checkpoint loading is the default. Loading a verified legacy
  pickle checkpoint requires the explicit `--trusted-checkpoint` opt-in.
- Strict RegAD support-set and exact-order ViTAD workflows require verified
  external artifacts that BaoIAD does not fabricate or redistribute.

Review the compatibility contract before upgrading existing automation,
especially Python environments, optional extras, checkpoint commands, and
strict-reproduction workflows.

## Validation boundary

CPU, static, package, documentation, and offline checks do not validate CUDA.
The separate [real-GPU validation contract](gpu_validation.md) requires a real
CUDA runtime, retained sanitized logs, key-method training and inference,
compiled-operator state, and measured peak VRAM for the exact candidate.

Until that evidence passes, the truthful state is **GPU not validated**. The
release must not imply validated GPU training, CUDA operators, peak VRAM, or
37-method end-to-end GPU execution. A release that excludes GPU validation
requires an explicit scope decision accepted by every go/no-go owner.

## Known limitations and support boundary

- Industrial datasets, third-party services, pretrained weights, official
  support sets, and exact-order files remain subject to their owners' access
  and redistribution terms.
- Method-level alignment and runtime limitations remain authoritative in the
  [method status inventory](../alignment/method_status.json) and linked method
  records.
- Network, slow, optional, and GPU checks are separate from the required
  offline CPU pull-request lane.
- BaoIAD support covers installation, configuration, documentation, and
  reproducible repository defects. It does not promise dataset licenses,
  third-party service availability, or unpublished checkpoints.
- Sensitive security details must not be posted in a public Issue. Follow
  [SECURITY.md](../../SECURITY.md); the public release remains blocked until
  its approved private reporting channel is active.

## Publication gate

The [external approval register](external_approvals.json), exact-commit checks,
owner approvals, and service-permission preflights must all pass before any
production action. Only after a recorded GO may approved owners create the
tag and GitHub Release, promote ReadTheDocs, or create the matching Zenodo
software version.

Published artifacts must contain audited source and release notes only. Do not
attach datasets, unaudited checkpoints, credentials, internal evidence, or
experiment archives. See the [support and hotfix runbook](support_hotfix_runbook.md)
for post-publication handling.
