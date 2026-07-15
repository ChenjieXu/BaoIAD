# Release support and hotfix runbook

## Purpose and activation boundary

This runbook defines role-based support for an approved BaoIAD public release.
It does not assign people, publish private contacts, create a service-level
commitment, or authorize production access.

Before publication, the restricted go/no-go evidence must name a launch
support owner and backup and confirm escalation paths for the Release,
Technical, Security, Brand, and Legal/OSS roles. Missing assignments are a
NO-GO. Tag, GitHub Release, ReadTheDocs, and Zenodo actions occur only after a
recorded GO and only through approved owner accounts.

## Roles

| Role | Support responsibility |
|---|---|
| Launch support owner | Coordinate intake, status, evidence, handoffs, and public updates |
| Launch support backup | Assume coordination when the primary owner is unavailable |
| Technical maintainer | Reproduce, classify, fix, review, and validate repository defects |
| Release owner | Protect the exact-commit gate, approve patch execution, and coordinate rollback |
| Security owner | Receive private security reports, control disclosure, and approve security fixes |
| Brand owner | Review public identity, media, event wording, and brand-impacting corrections |
| Legal/OSS owner | Review licensing, provenance, redistribution, and third-party incidents |

Public repository files must not contain personal phone numbers, private email
addresses, private chat links, credentials, or legal documents. The restricted
assignment record stores approved routing information.

## Supported intake

Public Issues may cover installation, configuration, documentation, and a
minimal reproducible repository defect. Requests for dataset access, dataset
licenses, third-party service availability, or unpublished checkpoints are
outside BaoIAD's support commitment and should be redirected to the relevant
owner without copying restricted content into the repository.

Suspected vulnerabilities, credentials, personal data, embargoed details, and
unpatched exploit information must not enter a public Issue or pull request.
Use the approved private process described by [SECURITY.md](../../SECURITY.md).
Until that process is active and tested, public release remains NO-GO.

Brand or legal concerns use the approved internal escalation route. Public
discussion should contain only the minimum non-sensitive status approved by
the responsible role.

## Incident flow

1. **Record:** Assign an opaque incident identifier and capture the affected
   public version, environment, reproduction, impact, and reporter-visible
   facts. Do not copy secrets, private assets, or licensed dataset content.
2. **Classify:** Route technical, security, brand, legal, documentation, and
   external-service ownership to the responsible role. Severity is based on
   user impact and exposure, not social-media volume.
3. **Reproduce:** Reproduce from the published immutable tag in an isolated
   environment. Record exact commands and distinguish CPU evidence from GPU
   evidence.
4. **Contain:** Publish only approved mitigations. Security containment and
   disclosure stay private until the Security owner authorizes publication.
5. **Fix:** Create the smallest reviewable fix from the published tag. Add a
   regression test and update English/Chinese documentation and release notes
   when user behavior changes.
6. **Verify:** Run every applicable P0 check and the exact-commit release gate.
   A GPU-related fix requires real-GPU evidence; CPU checks cannot substitute.
7. **Approve:** Obtain Technical and Release approval plus Security,
   Brand, or Legal/OSS approval when their scope is affected.
8. **Publish:** Only an approved Release owner may publish the next patch tag,
   GitHub Release, documentation update, and software archive record.
9. **Communicate:** Link the public fix and supported mitigation without
   exposing restricted evidence. Record follow-up actions and ownership in the
   restricted incident record.

No fixed response or remediation time is promised until the organization
approves and publishes one. The launch support owner records actual event
times for audit without inventing an SLA.

## Immutable tags and rollback

Published tags and GitHub Releases must be treated as immutable. Never
force-push, delete or move a published tag, replace its source commit, or
silently retarget a release asset.

- For an unreleased candidate, use a normal reviewed revert or fix and rerun
  the gate on the resulting new commit.
- For a published regression, branch from the affected immutable tag, apply
  the minimum approved change, and issue the next approved patch version.
- If `master` must be restored, merge a reviewed revert. Do not rewrite branch
  history.
- If documentation must be corrected, publish a reviewed new documentation
  build tied to the correction commit. Do not change the meaning of an old tag.
- If an archive or release record is wrong, preserve the historical record and
  add an approved correction or successor version according to the external
  service policy.

The patch candidate is a new exact commit. It must pass the same release
policy, compliance, package, documentation, security, and scope-appropriate
GPU gates as the original release. Its release notes identify the affected
version, user impact, mitigation, fixed behavior, and remaining limitations.

## Launch monitoring and handoff

During the approved publication window, the launch support owner and backup
monitor:

- organization-repository Issues and reproducible installation reports;
- the required GitHub Actions contexts and package smoke results;
- English and Chinese documentation builds and public source links;
- approved public release and archive pages;
- repeated reports that may indicate a security, licensing, dataset-rights,
  or compatibility incident.

Each handoff records the incident identifier, current public status, evidence
location, responsible role, next decision, and whether a user-facing update is
approved. Private contacts and sensitive evidence remain outside the public
repository.

After the publication window, unresolved incidents retain explicit owners.
The Release owner records a brief retrospective covering root cause, detection,
gate effectiveness, communication, and preventive actions without embedding
private evidence in the repository.
