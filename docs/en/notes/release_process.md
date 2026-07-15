# Release process

This document defines the proposed BaoIAD organization release process. It
does not assert that GitHub branch protection, private security reporting,
ReadTheDocs, Zenodo, or release permissions have already been configured.
Those production changes require the named external owners and recorded
approval evidence.

## Roles and review ownership

| Role | Required responsibility |
|---|---|
| Technical maintainer | Code/config correctness, test scope, compatibility, and method claims |
| Release owner | Exact-commit gate, release notes, tag/release execution, and rollback coordination |
| Legal/OSS owner | License, provenance, redistribution, and third-party disposition |
| Brand owner | Public identity, top-level README, media, and WAIC-facing wording |
| Security owner | Private reporting channel, security triage, and security-fix approval |
| Launch support owner and backup | Monitor installation, documentation, and reproducible bug reports during launch |

At least one reviewer other than the author approves every pull request.
Release-sensitive files additionally require the applicable role above.
`CODEOWNERS` must not be added until the organization supplies approved GitHub
users or teams and confirms that branch protection resolves them correctly.

## Required pull-request checks

The proposed P0 required status-check contract for `master` is:

1. `lint`
2. `release-policy`
3. `core-offline (3.10)`
4. `core-offline (3.12)`
5. `docs-en`
6. `docs-zh`

The workflow implementation must use stable names that match the branch-rule
contexts exactly. The core jobs use the committed CPU constraints and include
core import, method inventory, public-release checks, and data-independent CPU
tests. Sphinx warnings fail both documentation jobs.

Optional-extra/registry, clean-install, scheduled critical-link, network,
slow, and GPU jobs remain separately visible. Only stable offline CPU jobs may
be PR-required; a network or unavailable-GPU job must not become an accidental
merge blocker. Before branch protection is enabled, the Release owner records
the exact check contexts from a successful organization-repository run.

## Branch protection proposal

For `Baosight-xVue/BaoIAD:master`, an organization administrator should enable:

- pull requests required for changes;
- at least one approving reviewer;
- the P0 checks above required and up to date with the target branch;
- stale approvals dismissed after relevant changes;
- force pushes and branch deletion disabled;
- conversation resolution required;
- administrator bypass governed explicitly by organization policy and audited
  if enabled.

This repository does not apply those settings automatically. Enabling or
changing production branch rules requires organization-admin authority and a
post-change verification PR proving that direct pushes and failed checks are
blocked.

## GPU evidence is a separate gate

CPU release checks do not validate CUDA. A real-GPU evidence record includes:

- GPU model, driver, CUDA runtime, Python, PyTorch, TorchVision, the exact
  `mmcv` or `mmcv-lite` package/version, and commit;
- whether required compiled CUDA operators are present or not applicable;
- the selected key-method training and inference smoke commands and results;
- peak allocated/reserved GPU memory and any out-of-memory behavior;
- placeholder-only commands and retained logs that contain no credentials,
  raw dataset/work/checkpoint fields, private absolute paths, or file URIs.

Without a real CUDA device record, the state is **GPU not validated**. It is
never reported as green and the release must not claim validated GPU training,
CUDA operators, peak VRAM, or 37-method end-to-end execution. A Release owner
may only proceed with that state when the public release scope explicitly
excludes GPU validation and all other go/no-go owners accept the limitation.

See {doc}`Real-GPU release validation <gpu_validation>` for the manual workflow,
evidence contract, and G007 required gate.

## Exact-commit pre-tag gate

Run the complete gate on the exact merged commit intended for the tag:

1. All required CI checks are green and no code changes follow the run.
2. Python 3.10/3.12 clean installs use the committed CPU constraints.
3. Exact diff, allowlist, forbidden tracked paths, file-size gate, local links,
   secret scan, and research-worktree before/after evidence pass.
4. Method inventory, provenance, licensing, asset authorization, and public
   claims have no unresolved release-blocking item.
5. English/Chinese README and documentation rendering receive Technical and
   Brand approval.
6. Legal/OSS, Brand, Technical, Security, and Release approvals are recorded;
   tag/release, ReadTheDocs, and Zenodo permissions pass preflight.
7. Release notes, compatibility boundaries, support scope, launch coverage,
   and hotfix instructions are ready.

Pending entries in the
[external approval register](https://github.com/Baosight-xVue/BaoIAD/blob/master/docs/release/external_approvals.json) fail
safely by blocking release. In particular, a public release cannot pass the security gate while
`APP-SECURITY-CHANNEL` remains pending. Any yellow/red item produces no tag;
after a fix, rerun the entire gate on the new exact commit.

## Release execution

After a recorded go decision:

1. Verify package, CFF, documentation, GitHub Release, and Zenodo version
   metadata agree.
2. Create an annotated tag from the approved exact commit. Do not move or
   recreate an existing tag.
3. Publish only audited source and release notes; do not attach datasets,
   unaudited checkpoints, credentials, or experiment archives.
4. Trigger and verify stable/latest documentation and the matching Zenodo
   software record only through approved owner accounts.
5. Record the published URLs and launch support owner/backup in the release
   evidence without storing private contact details in the repository.

## Hotfix and rollback

- For an unreleased candidate, use a normal revert/fix pull request and rerun
  the exact-commit gate.
- For a published regression, branch from the immutable tag, prepare the
  smallest reviewed fix, reuse every P0 gate, and publish a new patch tag such
  as `v1.1.1`.
- Roll back `master` with a reviewed revert when needed; never force-push,
  delete a published tag, or retarget an existing release.
- Security hotfixes follow [SECURITY.md](https://github.com/Baosight-xVue/BaoIAD/blob/master/SECURITY.md) and require the
  approved private process before public disclosure.

See {doc}`the contribution guide <contributing>` for contributor-facing checks.

## Release materials

- {doc}`Draft v1.1.0 release notes <v1_1_0_release_notes>`
- {doc}`WAIC go/no-go checklist <waic_go_no_go>`
- {doc}`Support and hotfix runbook <support_hotfix_runbook>`
